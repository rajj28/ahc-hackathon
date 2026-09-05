"""End-to-end local inference to an upload-safe submission."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import load
from .explain import attach
from .fallback import load_model as load_clip_model, window_events
from .features import build_cache
from .types import Event, VideoPred


def predict_video(path: str, video_id: str, level: int, models, cfg) -> VideoPred:
    """Predict one cached video; decode, feature extraction and head inference are timed."""
    started = time.perf_counter()
    with np.load(path) as cached:
        embedding = cached["emb"].astype(np.float32)
        timestamps = cached["ts"].astype(np.float32)
        extract_ms = float(cached.get("extract_time_ms", 0.0))
    if not len(embedding):
        return VideoPred(video_id, level, frames_processed=0,
                         end_to_end_internal_time_ms=extract_ms + (time.perf_counter() - started) * 1000)
    events: list[Event] = []
    if level == 1:
        with torch.no_grad():
            probabilities = torch.softmax(models["clip"](torch.from_numpy(embedding).cuda()), dim=0).cpu().numpy()
        class_index = int(np.argmax(probabilities[:11]))
        if float(probabilities[11]) < float(cfg.decide.l1_normal_threshold):
            events = [Event(cfg.classes[class_index], None, None, float(probabilities[class_index]))]
    else:
        # R4's temporal checkpoint failed the real-footage transfer gate. The calibrated
        # ClipHead window decoder is the conservative production path until it passes.
        raw_events = window_events(models["clip"], embedding, timestamps, cfg,
                                   float(cfg.fallback.threshold), cfg.fallback.window_sec,
                                   cfg.fallback.stride_sec, float(cfg.fallback.expand),
                                   bool(cfg.fallback.snap))
        events = [Event(item["class_name"], item["start_time_sec"], item["end_time_sec"], item["score"])
                  for item in raw_events]
    return VideoPred(video_id, level, events, len(embedding), len(events),
                     extract_ms + (time.perf_counter() - started) * 1000)


def load_manifest(path: str) -> dict:
    """Return {video_id: manifest entry}; manifest levels are the sole authority."""
    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".csv":
        import pandas as pd
        table = pd.read_csv(manifest_path)
        return {str(row.video_id): {"level": int(row.level)} for row in table.itertuples(index=False)}
    with manifest_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("videos", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        return {str(video_id): (value if isinstance(value, dict) else {"level": value})
                for video_id, value in entries.items()}
    return {str(item["video_id"]): item for item in entries}


def main(manifest_path: str, out_path: str, cfg, only: list[str] | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    requested = set(only) if only else None
    # The day-of pack arrives as MP4. Build its resumable eval cache before loading heads;
    # this covers decoding, preprocessing and frozen-encoder inference locally.
    build_cache("eval", cfg.features.encoder, cfg, manifest_path)
    models = {"clip": load_clip_model(cfg).eval()}
    predictions = []
    cache_root = Path(cfg.paths.root) / cfg.paths.cache / cfg.features.encoder
    for video_id, entry in manifest.items():
        if requested is not None and video_id not in requested:
            continue
        cache_path = cache_root / "eval" / f"{video_id}.npz"
        if not cache_path.exists():
            cache_path = cache_root / "test" / f"{video_id}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"missing embedding cache for {video_id}: {cache_path}")
        predictions.append(predict_video(str(cache_path), video_id, int(entry["level"]), models, cfg).to_dict())
    attach(predictions, cfg)
    submission = {"schema_version": cfg.submission.schema_version, "predictions": predictions}
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(submission, handle, indent=2)
    return submission


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    main(args.manifest, args.out, load(args.config), args.only)


if __name__ == "__main__":
    cli()
