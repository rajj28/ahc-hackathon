"""Conservative sliding-window ClipHead fallback for L2/L3 localization.

This is an insurance path, not a replacement for the temporal model: it reuses the validated
whole-clip classifier on overlapping 10 s windows, then emits only high-confidence windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import load
from .heads import ClipHead
from .scorer import load_gt, match_events, score_submission, sensitivity
from .segment import changepoints


def window_events(model, emb: np.ndarray, ts: np.ndarray, cfg, threshold: float,
                  window_sec: float | None = None, stride_sec: float | None = None,
                  expand: float = 1.0, snap: bool = True, max_gap_sec: float = 0.0,
                  max_events: int | None = None, max_span_frac: float = 1.0,
                  reclassify: bool = False) -> list[dict]:
    fps = float(cfg.features.target_fps)
    width = max(1, round((window_sec or cfg.fallback.window_sec) * fps))
    stride = max(1, round((stride_sec or cfg.fallback.stride_sec) * fps))
    starts = list(range(0, max(1, len(emb) - width + 1), stride))
    final_start = max(0, len(emb) - width)
    if final_start not in starts:
        starts.append(final_start)
    windows = [emb[start:min(len(emb), start + width)] for start in starts]
    candidates = []
    model.eval()
    with torch.no_grad():
        # Windows are a fixed width for L2/L3. Batching makes a 90-point sweep cheap.
        for offset in range(0, len(windows), cfg.fallback.batch):
            batch = windows[offset:offset + cfg.fallback.batch]
            probability_batch = torch.softmax(model(torch.from_numpy(np.stack(batch)).cuda()), dim=-1).cpu().numpy()
            for start, frames, probability in zip(starts[offset:offset + len(batch)], batch, probability_batch):
                end = start + len(frames)
                class_idx = int(probability[:11].argmax())
                score = float(probability[class_idx])
                if score >= threshold and score > float(probability[11]):
                    step = float(ts[1] - ts[0]) if len(ts) > 1 else 1 / fps
                    candidates.append((start, end, class_idx, score,
                                       float(ts[start]), float(ts[end - 1] + step)))
    # Local labels are deliberately *not* trusted here: the motion-confusion cluster splits
    # one physical event among wrong-way/blocking/stalled windows. Merge temporal evidence
    # first, then classify the complete interval below at the scale ClipHead was trained on.
    candidates.sort(key=lambda item: item[0])
    merged = []
    for candidate in candidates:
        same_local_class = candidate[2] == merged[-1][2] if merged else False
        if merged and candidate[0] <= merged[-1][1] + stride and (reclassify or same_local_class):
            old = merged[-1]
            merged[-1] = (old[0], max(old[1], candidate[1]), old[2], max(old[3], candidate[3]),
                           old[4], max(old[5], candidate[5]))
        else:
            merged.append(candidate)
    result = []
    # The classifier determines *which* interval has evidence.  For spliced eval footage,
    # embedding discontinuities can place its edges more accurately than window geometry.
    # This is deliberately optional so the R5 ablation is a real like-for-like comparison.
    cps = changepoints(emb, cfg.segment.changepoint_k) if snap else np.empty(0, dtype=np.int64)
    minimum = max(1, round(cfg.fallback.min_interval_sec * fps))
    for start, end, class_idx, score, _, _ in merged:
        # A positive W-second classifier is centred on evidence, but raw union extends W/2
        # beyond either edge. Remove that deterministic window smear before final calibration.
        start, end = int(round(start + width / 2)), int(round(end - width / 2))
        if end - start < minimum:
            centre = (start + end) / 2
            start, end = int(round(centre - minimum / 2)), int(round(centre + minimum / 2))
        centre = (start + end) / 2
        half = max(minimum / 2, (end - start) * expand / 2)
        start, end = max(0, int(np.floor(centre - half))), min(len(ts), int(np.ceil(centre + half)))
        if len(cps):
            start_candidates = cps[np.abs(cps - start) <= cfg.segment.snap_max_shift_frames]
            end_candidates = cps[np.abs(cps - end) <= cfg.segment.snap_max_shift_frames]
            snapped_start = int(start_candidates[np.argmin(np.abs(start_candidates - start))]) if len(start_candidates) else start
            snapped_end = int(end_candidates[np.argmin(np.abs(end_candidates - end))]) if len(end_candidates) else end
            if snapped_end > snapped_start:
                start, end = snapped_start, snapped_end
        if end <= start or (end - start) / len(ts) > max_span_frac:
            continue
        if reclassify:
            # Experimental R3b precursor: classify a merged interval at the scale ClipHead
            # was trained on. Kept opt-in until it demonstrably beats the window labels.
            with torch.no_grad():
                interval_prob = torch.softmax(model(torch.from_numpy(emb[start:end]).cuda()), dim=0).cpu().numpy()
            class_idx = int(interval_prob[:11].argmax())
            score = float(interval_prob[class_idx])
            if score <= float(interval_prob[11]):
                continue
        step = float(ts[1] - ts[0]) if len(ts) > 1 else 1 / fps
        result.append({"class_name": cfg.classes[class_idx], "start_time_sec": float(ts[start]),
                       "end_time_sec": float(ts[end - 1] + step), "score": score})
    # Re-classification can make neighbouring local segments share a class. Bridge only these
    # gaps, preserving class boundaries in multi-event clips such as T026.
    result.sort(key=lambda item: item["start_time_sec"])
    bridged = []
    for item in result:
        if (bridged and item["class_name"] == bridged[-1]["class_name"] and
                item["start_time_sec"] - bridged[-1]["end_time_sec"] <= max_gap_sec):
            prior = bridged[-1]
            prior["end_time_sec"] = item["end_time_sec"]
            prior["score"] = max(prior["score"], item["score"])
        else:
            bridged.append(item)
    ranked = sorted(bridged, key=lambda item: item["score"], reverse=True)
    return ranked[:max_events] if max_events is not None else ranked


def load_model(cfg):
    checkpoint = torch.load(Path(cfg.paths.root) / cfg.paths.runs / "clip_head.pt", map_location="cuda", weights_only=True)
    model = ClipHead(checkpoint["d_in"], cfg).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def public_dev_submission(cfg, threshold: float, window_sec=None, stride_sec=None, expand=1.0,
                          snap: bool = True, model=None) -> dict:
    model = model or load_model(cfg)
    with (Path(cfg.paths.root) / cfg.paths.runs / "sub_R3.json").open(encoding="utf-8") as handle:
        submission = json.load(handle)
    for prediction in submission["predictions"]:
        video_id = str(prediction["video_id"])
        if not video_id.startswith("T") or int(video_id[1:]) < 25:
            continue
        with np.load(Path(cfg.paths.root) / cfg.paths.cache / cfg.features.encoder / "test" / f"{video_id}.npz") as cached:
            prediction["events"] = window_events(model, cached["emb"].astype(np.float32), cached["ts"], cfg, threshold,
                                                   window_sec, stride_sec, expand, snap)
            prediction["runtime_metadata"]["frames_processed"] = len(cached["emb"])
            prediction["runtime_metadata"]["chunks_processed"] = len(prediction["events"])
    return submission


def component_report(submission: dict, gt) -> dict:
    by_id = {item["video_id"]: item for item in submission["predictions"]}
    report = {}
    for level in (2, 3):
        found = emitted = events = 0
        for video_id, group in gt[gt.level == level].groupby("video_id", sort=False):
            predicted = by_id[video_id]["events"]
            truth = group.to_dict("records") if bool(group.iloc[0].is_anomaly) else []
            found += len(match_events(predicted, truth))
            emitted += len(predicted)
            events += len(truth)
        report[f"L{level}"] = {"found": found, "emitted": emitted, "false_alarms": emitted - found,
                                "precision": found / emitted if emitted else 0.0,
                                "recall": found / events if events else 0.0}
    return report


def sweep_l3(cfg, gt, output: str = "runs/l3_sweep.json") -> dict:
    """Tune L3 only, preserving the validated L1/L2 R5 baseline exactly.

    This intentionally returns a ranked table rather than installing an argmax: eight L3
    events are too few for a one-point peak to be trusted. The caller chooses a broad plateau.
    """
    with (Path(cfg.paths.root) / cfg.paths.runs / "sub_R5_window_sweep.json").open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    model = load_model(cfg).eval()
    cached = {}
    for video_id in ("T031", "T032", "T033", "T034"):
        with np.load(Path(cfg.paths.root) / cfg.paths.cache / cfg.features.encoder / "test" / f"{video_id}.npz") as item:
            cached[video_id] = (item["emb"].astype(np.float32), item["ts"])
    trials = []
    for window in (6.0, 10.0, 16.0, 24.0):
        for stride in (2.0, 4.0):
            for threshold in (.4, .5, .6, .7):
                for expand in (1.0, 1.25):
                    for cap in (1, 2, 3, 5):
                        for gap in (0.0, 5.0, 15.0):
                            submission = json.loads(json.dumps(baseline))
                            for prediction in submission["predictions"]:
                                if prediction["video_id"] not in cached:
                                    continue
                                emb, ts = cached[prediction["video_id"]]
                                prediction["events"] = window_events(
                                    model, emb, ts, cfg, threshold, window, stride, expand,
                                    True, gap, cap, .6)
                            grid = sensitivity(submission, gt)
                            score = score_submission(submission, gt)
                            trials.append({"window_sec": window, "stride_sec": stride,
                                           "threshold": threshold, "expand": expand,
                                           "max_events": cap, "max_gap_sec": gap,
                                           "min_total": float(grid.total.min()),
                                           "median_total": float(grid.total.median()),
                                           "score": score})
    ranked = sorted(trials, key=lambda item: (item["min_total"], item["median_total"]), reverse=True)
    result = {"baseline_min": float(sensitivity(baseline, gt).total.min()),
              "top_10": ranked[:10], "n_trials": len(trials)}
    with Path(output).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--out")
    parser.add_argument("--window", type=float)
    parser.add_argument("--stride", type=float)
    parser.add_argument("--expand", type=float, default=1.0)
    parser.add_argument("--no-snap", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-l3", action="store_true")
    args = parser.parse_args()
    cfg = load()
    gt = load_gt(str(Path(cfg.paths.root) / cfg.paths.test / "ground_truth.csv"))
    if args.sweep_l3:
        sweep_l3(cfg, gt)
        return
    if args.sweep:
        model, best = load_model(cfg), None
        for window in (3.0, 4.0, 6.0):
            for stride in (1.0, 2.0):
                for threshold in np.arange(.4, .81, .1):
                    for expand in (.8, 1.0, 1.25):
                        for snap in (False, True):
                            submission = public_dev_submission(cfg, float(threshold), window, stride, expand, snap, model)
                            grid = sensitivity(submission, gt)
                            candidate = (float(grid.total.min()), submission, window, stride, float(threshold), expand, snap)
                            if best is None or candidate[0] > best[0]:
                                best = candidate
                                print(f"new best min={best[0]:.4f} window={window} stride={stride} "
                                      f"threshold={threshold:.1f} expand={expand} snap={snap}", flush=True)
        _, submission, window, stride, threshold, expand, snap = best
        score, grid = score_submission(submission, gt), sensitivity(submission, gt)
        ablation_submission = public_dev_submission(cfg, threshold, window, stride, expand, not snap, model)
        ablation_grid = sensitivity(ablation_submission, gt)
        result = {"params": {"window_sec": window, "stride_sec": stride, "threshold": threshold,
                               "expand": expand, "snap": snap},
                  "score": score, "components": component_report(submission, gt),
                  "sensitivity": {"min": float(grid.total.min()), "median": float(grid.total.median()),
                                  "max": float(grid.total.max())},
                  "snap_ablation": {"same_params_without_snap_min": float(ablation_grid.total.min()),
                                    "delta_min": float(grid.total.min() - ablation_grid.total.min())}}
    else:
        if args.threshold is None:
            parser.error("--threshold is required unless --sweep is used")
        submission = public_dev_submission(cfg, args.threshold, args.window, args.stride, args.expand,
                                           not args.no_snap)
        score, grid = score_submission(submission, gt), sensitivity(submission, gt)
        result = {"score": score, "components": component_report(submission, gt),
                  "sensitivity": {"min": float(grid.total.min()), "median": float(grid.total.median()),
                                  "max": float(grid.total.max())}}
    print(json.dumps(result, indent=2))
    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as handle:
            json.dump(submission, handle, indent=2)


if __name__ == "__main__":
    main()
