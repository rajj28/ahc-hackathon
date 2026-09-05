"""Local replica of the official AHC metric."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


def _events(video: object) -> list[dict]:
    return video.get("events", []) if isinstance(video, dict) else []


def _interval(event: dict) -> tuple[float, float]:
    return float(event["start_time_sec"]), float(event["end_time_sec"])


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Temporal intersection over union. Zero-length union -> 0.0."""
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def match_events(pred: list, gt: list, thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, descending IoU, with exact class equality."""
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred_event in enumerate(pred):
        for gt_index, gt_event in enumerate(gt):
            if pred_event.get("class_name") != gt_event.get("class_name"):
                continue
            try:
                overlap = iou(_interval(pred_event), _interval(gt_event))
            except (KeyError, TypeError, ValueError):
                overlap = 0.0
            if overlap >= thr:
                candidates.append((overlap, pred_index, gt_index))
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_pred, used_gt, result = set(), set(), []
    for _, pred_index, gt_index in candidates:
        if pred_index not in used_pred and gt_index not in used_gt:
            used_pred.add(pred_index)
            used_gt.add(gt_index)
            result.append((pred_index, gt_index))
    return result


def score_video_l23(pred: list, gt: list, level: int, w: tuple[float, float, float]) -> float:
    """Score one L2/L3 video under (alert, match-F1, timing-IoU) weights."""
    if not gt:
        return 1.0 if not pred else 0.0
    matches = match_events(pred, gt)
    matched = len(matches)
    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(gt)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = (sum(iou(_interval(pred[p]), _interval(gt[g])) for p, g in matches) / matched
                if matched else 0.0)
    return w[0] * float(bool(pred)) + w[1] * f1 + w[2] * mean_iou


def score_level1(pred_by_id: dict, gt_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return binary accuracy, class accuracy, and their equally weighted mean."""
    binary_correct, class_correct = [], []
    # ASSUMPTION: class accuracy is pooled over all L1 clips, including normal clips;
    # therefore an empty prediction correctly classifies a normal video as "normal".
    for video_id, group in gt_df[gt_df["level"] == 1].groupby("video_id", sort=False):
        gt_is_anomaly = bool(group.iloc[0]["is_anomaly"])
        gt_class = group.iloc[0]["class_name"] if gt_is_anomaly else "normal"
        events = _events(pred_by_id.get(video_id, {}))
        if events:
            best = max(events, key=lambda event: float(event.get("score", 0.0)))
            predicted_class = best.get("class_name")
        else:
            predicted_class = "normal"
        binary_correct.append((predicted_class != "normal") == gt_is_anomaly)
        class_correct.append(predicted_class == gt_class)
    binary = float(np.mean(binary_correct)) if binary_correct else 0.0
    classes = float(np.mean(class_correct)) if class_correct else 0.0
    return binary, classes, (binary + classes) / 2.0


def score_submission(pred_json: dict | str | Path, gt_df: pd.DataFrame,
                     w_l2=(0.2, 0.5, 0.3), w_l3=(0.2, 0.4, 0.4)) -> dict:
    """Return level scores, difficulty contributions, total, and answered count."""
    if isinstance(pred_json, (str, Path)):
        with Path(pred_json).open(encoding="utf-8") as handle:
            pred_json = json.load(handle)
    predictions = pred_json.get("predictions", pred_json)
    if isinstance(predictions, list):
        pred_by_id = {item.get("video_id"): item for item in predictions if isinstance(item, dict)}
    elif isinstance(predictions, dict):
        pred_by_id = predictions
    else:
        raise TypeError("submission predictions must be a list or mapping")

    _, _, l1 = score_level1(pred_by_id, gt_df)
    scores = {1: l1}
    # ASSUMPTION: the non-public L2/L3 mix uses the configurable defaults documented in
    # CODEX.md/configs/default.yaml; it is not inferred from the public development set.
    for level, weights in ((2, w_l2), (3, w_l3)):
        values = []
        for video_id, group in gt_df[gt_df["level"] == level].groupby("video_id", sort=False):
            gt = [] if not bool(group.iloc[0]["is_anomaly"]) else group.to_dict("records")
            values.append(score_video_l23(_events(pred_by_id.get(video_id, {})), gt, level, weights))
        scores[level] = float(np.mean(values)) if values else 0.0
    d1, d2, d3 = 25 * scores[1], 35 * scores[2], 40 * scores[3]
    return {"L1": scores[1], "L2": scores[2], "L3": scores[3], "D1": d1, "D2": d2,
            "D3": d3, "total": d1 + d2 + d3, "n_answered": len(pred_by_id)}


def sensitivity(pred_json: dict, gt_df: pd.DataFrame, grid: dict | None = None) -> pd.DataFrame:
    """Re-score all L2/L3 weighting combinations, followed by min/median/max total."""
    if grid is None:
        grid = {"l2": ((.2, .5, .3), (.3, .4, .3), (.34, .33, .33)),
                "l3": ((.2, .4, .4), (.2, .3, .5), (.3, .3, .4))}
    rows = []
    for w_l2 in grid["l2"]:
        for w_l3 in grid["l3"]:
            rows.append({"w_l2": tuple(w_l2), "w_l3": tuple(w_l3),
                         **score_submission(pred_json, gt_df, tuple(w_l2), tuple(w_l3))})
    frame = pd.DataFrame(rows)
    summary = {"w_l2": "summary", "w_l3": "summary", "total_min": frame["total"].min(),
               "total_median": frame["total"].median(), "total_max": frame["total"].max()}
    return pd.concat((frame, pd.DataFrame([summary])), ignore_index=True, sort=False)


def load_gt(path: str) -> pd.DataFrame:
    """Read a ground_truth.csv into the canonical scorer frame."""
    frame = pd.read_csv(path)
    required = {"video_id", "level", "is_anomaly", "class_name", "start_time_sec", "end_time_sec"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ground truth missing columns: {sorted(missing)}")
    frame = frame.loc[:, [column for column in frame.columns if column in required]].copy()
    frame["level"] = frame["level"].astype(int)
    frame["is_anomaly"] = frame["is_anomaly"].astype(str).str.lower().map({"true": True, "false": False})
    if frame["is_anomaly"].isna().any():
        raise ValueError("is_anomaly must be true or false")
    return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("ground_truth")
    args = parser.parse_args()
    print(score_submission(args.submission, load_gt(args.ground_truth)))
