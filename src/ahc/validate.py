"""Pre-flight schema validator. Run before EVERY upload.

A rejected file does not consume a run, but it does consume minutes, and the seven documented
rejection causes are all trivially checkable offline.

    python -m ahc.validate runs/sub.json --manifest manifest.json

Exits non-zero and prints a per-video table naming the offending field, matching the format
the arena itself returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load

CHECKS = """
Structural
  - top-level `predictions` present (the only strictly required top-level field)
  - per video: video_id, events, runtime_metadata all present
  - video_id matches the manifest exactly and appears exactly once
  - every manifest video is covered (an unanswered video is scored as normal)
  - file size <= 5 MB

Per event
  - class_name is one of the 11; "normal" is REJECTED - express normal as `events: []`
  - level 1: start_time_sec and end_time_sec are both null  (timestamps at L1 are rejected)
  - level 2/3: start_time_sec present and >= 0, end_time_sec > start_time_sec
  - explanation, if present, is 20-500 chars; otherwise omit the key entirely

runtime_metadata
  - present on every video, including videos with no events
  - if a model_runtimes entry gives total_time_ms, call_count and average_time_ms then
    |average - total/count| / average <= 2%
  - any call_times_ms has exactly call_count entries

Sanity (warn, do not fail)
  - a level 2/3 video whose single event spans >80% of its duration: the 0.5 IoU gate makes
    "the whole clip is anomalous" score far below a real attempt
  - more than ~8 events on one video: fragments cannot each match, and the extras count against
  - zero events predicted across ALL level 3 videos: public L3 has no normal videos at all, so
    total silence there is very likely a bug
"""


def validate(sub: dict, manifest: dict, cfg) -> list[dict]:
    """-> list of {video_id, field, problem, severity}. Empty list == safe to upload."""
    problems: list[dict] = []
    def add(video_id, field, problem, severity="error"):
        problems.append({"video_id": video_id, "field": field, "problem": problem,
                         "severity": severity})

    if not isinstance(sub, dict) or "predictions" not in sub:
        add("<file>", "predictions", "missing required top-level field")
        return problems
    predictions = sub["predictions"]
    if not isinstance(predictions, list):
        add("<file>", "predictions", "must be a list")
        return problems
    max_bytes = cfg.submission.max_bytes
    encoded_size = len(json.dumps(sub, ensure_ascii=False).encode("utf-8"))
    if encoded_size > max_bytes:
        add("<file>", "file_size", f"{encoded_size} bytes exceeds {max_bytes}")

    manifest_levels = _manifest_levels(manifest)
    known_classes = set(cfg.classes)
    seen = set()
    for index, prediction in enumerate(predictions):
        video_label = f"<prediction {index}>"
        if not isinstance(prediction, dict):
            add(video_label, "prediction", "must be an object")
            continue
        video_id = prediction.get("video_id", video_label)
        if "video_id" not in prediction:
            add(video_label, "video_id", "missing required field")
            continue
        if video_id in seen:
            add(video_id, "video_id", "duplicate video_id")
        seen.add(video_id)
        if video_id not in manifest_levels:
            add(video_id, "video_id", "not in manifest")
            continue
        for key in ("events", "runtime_metadata"):
            if key not in prediction:
                add(video_id, key, "missing required field")
        events = prediction.get("events", [])
        if not isinstance(events, list):
            add(video_id, "events", "must be a list")
            continue
        level = manifest_levels[video_id]
        for event_index, event in enumerate(events):
            prefix = f"events[{event_index}]"
            if not isinstance(event, dict):
                add(video_id, prefix, "must be an object")
                continue
            class_name = event.get("class_name")
            if class_name == "normal":
                add(video_id, f"{prefix}.class_name", "'normal' is forbidden; use events: []")
            elif class_name not in known_classes:
                add(video_id, f"{prefix}.class_name", "unknown class string")
            start, end = event.get("start_time_sec"), event.get("end_time_sec")
            if level == 1:
                if start is not None or end is not None:
                    add(video_id, prefix + ".start_time_sec/end_time_sec",
                        "timestamps must both be null for level 1")
            elif start is None or end is None:
                add(video_id, prefix + ".start_time_sec/end_time_sec", "timestamps are required for level 2/3")
            else:
                try:
                    if float(start) < 0:
                        add(video_id, prefix + ".start_time_sec", "must be >= 0")
                    if float(end) <= float(start):
                        add(video_id, prefix + ".end_time_sec", "must be greater than start_time_sec")
                except (TypeError, ValueError):
                    add(video_id, prefix + ".start_time_sec/end_time_sec", "must be numeric")
            if "explanation" in event:
                explanation = event["explanation"]
                if not isinstance(explanation, str) or not (cfg.submission.explanation_min_chars <= len(explanation) <= cfg.submission.explanation_max_chars):
                    add(video_id, prefix + ".explanation", "must be a 20-500 character string")
        metadata = prediction.get("runtime_metadata")
        if metadata is not None and not isinstance(metadata, dict):
            add(video_id, "runtime_metadata", "must be an object")
        elif isinstance(metadata, dict):
            for runtime_index, runtime in enumerate(metadata.get("model_runtimes", [])):
                if not isinstance(runtime, dict):
                    add(video_id, f"runtime_metadata.model_runtimes[{runtime_index}]", "must be an object")
                    continue
                required = {"total_time_ms", "call_count", "average_time_ms"}
                if required <= runtime.keys():
                    total, count, average = (runtime[key] for key in ("total_time_ms", "call_count", "average_time_ms"))
                    if not isinstance(count, (int, float)) or count <= 0 or not isinstance(average, (int, float)) or average == 0 or abs(average - total / count) / abs(average) > .02:
                        add(video_id, f"runtime_metadata.model_runtimes[{runtime_index}].average_time_ms", "must equal total_time_ms/call_count within 2%")
                if "call_times_ms" in runtime and "call_count" in runtime and len(runtime["call_times_ms"]) != runtime["call_count"]:
                    add(video_id, f"runtime_metadata.model_runtimes[{runtime_index}].call_times_ms", "length must equal call_count")
    for video_id in manifest_levels:
        if video_id not in seen:
            add(video_id, "video_id", "missing manifest video")
    return problems


def _manifest_levels(manifest: dict | pd.DataFrame) -> dict[str, int]:
    if isinstance(manifest, pd.DataFrame):
        return {str(row.video_id): int(row.level) for row in manifest[["video_id", "level"]].drop_duplicates().itertuples()}
    if isinstance(manifest, dict):
        items = manifest.get("videos", manifest)
        if isinstance(items, list):
            return {str(item["video_id"]): int(item["level"]) for item in items}
        return {str(video_id): int(value["level"] if isinstance(value, dict) else value)
                for video_id, value in items.items()}
    raise TypeError("manifest must map video_id to level")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    with Path(args.submission).open(encoding="utf-8") as handle:
        submission = json.load(handle)
    manifest_path = Path(args.manifest)
    if manifest_path.suffix.lower() == ".csv":
        manifest = pd.read_csv(manifest_path)
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    result = validate(submission, manifest, load(args.config))
    print(pd.DataFrame(result).to_string(index=False) if result else "Validation passed: 0 problems")
    if result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
