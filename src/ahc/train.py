"""Training entry points. Both heads train on cached embeddings, so a full cycle is minutes.

    python -m ahc.train clip        # ClipHead      -> runs/clip_head.pt
    python -m ahc.train temporal    # TemporalHead  -> runs/temporal_head.pt

Every run writes runs/<name>.json with config, val metrics and the scorer.sensitivity() table,
so any two experiments are comparable after the fact.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import ClipDataset, SpliceDataset, build_real_holdout, build_synth_val, load_clip_table, make_donor_split
from .heads import ClipHead, TemporalHead, class_weights, losses
from .scorer import load_gt, match_events, score_submission


def _pad_collate(batch):
    embeddings, labels = zip(*batch)
    lengths = [len(embedding) for embedding in embeddings]
    padded = torch.zeros(len(batch), max(lengths), embeddings[0].shape[-1], dtype=torch.float32)
    mask = torch.zeros(len(batch), max(lengths), dtype=torch.bool)
    for index, embedding in enumerate(embeddings):
        padded[index, :len(embedding)] = embedding
        mask[index, :len(embedding)] = True
    return padded, mask, torch.tensor(labels, dtype=torch.long)


def _evaluate(model, loader, device):
    model.eval()
    true, predicted = [], []
    with torch.no_grad():
        for embeddings, mask, labels in loader:
            logits = model(embeddings.to(device), mask.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            true.extend(labels.tolist())
    return np.asarray(true), np.asarray(predicted)


def _write_r3_submission(model, cfg, device) -> dict:
    """Evaluate all public-dev L1 videos while leaving L2/L3 empty for this card."""
    import pandas as pd

    gt = load_gt(str(Path(cfg.paths.root) / cfg.paths.test / "ground_truth.csv"))
    # Use item access: ``DataFrameGroupBy.level`` resolves to pandas' grouping
    # metadata rather than the ground-truth column on current pandas versions.
    levels = gt.groupby("video_id")["level"].first().to_dict()
    videos = pd.read_csv(Path(cfg.paths.root) / cfg.paths.test / "videos.csv")
    predictions = []
    model.eval()
    for row in videos.itertuples(index=False):
        video_id = str(row.video_id)
        events = []
        frames_processed, elapsed_ms = 0, 0.0
        if levels.get(video_id) == 1:
            cache_path = Path(cfg.paths.root) / cfg.paths.cache / cfg.features.encoder / "test" / f"{video_id}.npz"
            with np.load(cache_path) as cached:
                embedding = cached["emb"].astype(np.float32)
            started = time.perf_counter()
            with torch.no_grad():
                probabilities = torch.softmax(model(torch.from_numpy(embedding).to(device)), dim=0).cpu().numpy()
            elapsed_ms = (time.perf_counter() - started) * 1000
            frames_processed = len(embedding)
            normal_probability = probabilities[11]
            class_index = int(np.argmax(probabilities[:11]))
            if normal_probability < cfg.decide.l1_normal_threshold:
                events = [{"class_name": cfg.classes[class_index], "start_time_sec": None,
                           "end_time_sec": None, "score": float(probabilities[class_index])}]
        predictions.append({"video_id": video_id, "events": events,
                            "runtime_metadata": {"frames_processed": frames_processed,
                                                 "chunks_processed": 1 if frames_processed else 0,
                                                 "end_to_end_internal_time_ms": elapsed_ms}})
    submission = {"schema_version": cfg.submission.schema_version, "predictions": predictions}
    output_path = Path(cfg.paths.root) / cfg.paths.runs / "sub_R3.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(submission, handle, indent=2)
    return score_submission(submission, gt)


def _naive_temporal_events(model, embedding, ts, cfg, device):
    """R4-only conservative decoder: no smoothing, snapping, or duration priors.

    R5 owns the full segmentation pipeline.  This deliberately simple decoder makes the R4
    transfer gate measure the head rather than post-processing cleverness.
    """
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(embedding).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    anomaly = probs[:, :11]
    labels, confidence = anomaly.argmax(axis=1), anomaly.max(axis=1)
    active = (confidence >= cfg.segment.hi) & (confidence > probs[:, 11])
    events, start = [], None
    for index in range(len(active) + 1):
        same = index < len(active) and start is not None and active[index] and labels[index] == labels[start]
        if start is None and index < len(active) and active[index]:
            start = index
        elif start is not None and not same:
            end = index
            end_sec = float(ts[end - 1] + (ts[1] - ts[0] if len(ts) > 1 else 1 / cfg.features.target_fps))
            if end_sec - float(ts[start]) >= cfg.segment.min_dur_sec:
                events.append({"class_name": cfg.classes[int(labels[start])],
                               "start_time_sec": float(ts[start]), "end_time_sec": end_sec,
                               "score": float(confidence[start:end].mean())})
            start = index if index < len(active) and active[index] else None
    return events


def _temporal_gate(model, videos, gt, cfg, device):
    """Report the dashboard-equivalent found/FA precision and normal-video alarm rate."""
    predictions, found, emitted, normal_alerts, normal_videos, anomalous_events = [], 0, 0, 0, 0, 0
    model.eval()
    for embedding, ts, video_id in videos:
        events = _naive_temporal_events(model, embedding, ts, cfg, device)
        predictions.append({"video_id": video_id, "events": events})
        group = gt[gt.video_id == video_id]
        is_anomaly = bool(group.iloc[0].is_anomaly)
        if is_anomaly:
            truth = group.to_dict("records")
            anomalous_events += len(truth)
            found += len(match_events(events, truth))
        else:
            normal_videos += 1
            normal_alerts += int(bool(events))
        emitted += len(events)
    submission = {"predictions": predictions}
    score = score_submission(submission, gt)
    false_alarms = emitted - found
    return {"precision": found / emitted if emitted else 0.0,
            "recall": found / anomalous_events if anomalous_events else 0.0,
            "false_alarm_rate": normal_alerts / normal_videos if normal_videos else 0.0,
            "found": found, "false_alarms": false_alarms, "emitted": emitted,
            "anomalous_events": anomalous_events, "normal_videos": normal_videos,
            "normal_video_alerts": normal_alerts, "scorer": score}


def train_clip(cfg):
    """ClipHead on ClipDataset. 12-way over whole trimmed clips.

    Val = the `val` donor pool (never spliced into training videos). Report balanced accuracy
    and the confusion matrix - the confusions that matter are fire/smoke and
    traffic_congestion/vehicle_blocking_traffic/stalled_or_broken_down_vehicle, which are the
    pairs most likely to cost L1 class accuracy.
    """
    torch.manual_seed(cfg.splice.seed)
    np.random.seed(cfg.splice.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = make_donor_split(load_clip_table(cfg), cfg.splice.val_donor_frac, cfg.splice.seed)
    fit_dataset, val_dataset = ClipDataset(table, cfg, "fit"), ClipDataset(table, cfg, "val")
    fit_loader = DataLoader(fit_dataset, batch_size=cfg.train.batch, shuffle=True,
                            collate_fn=_pad_collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch, shuffle=False,
                            collate_fn=_pad_collate, num_workers=0)
    model = ClipHead(fit_dataset.store.shape[1], cfg).to(device)
    counts = table[table.donor_split == "fit"].class_idx.value_counts().to_dict()
    weights = class_weights(counts, cfg.train.class_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    best_state, best_balanced, best_metrics = None, -1.0, None
    for epoch in range(cfg.train.epochs):
        model.train()
        for embeddings, mask, labels in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=bool(cfg.train.amp and device.type == "cuda")):
                loss = F.cross_entropy(model(embeddings.to(device), mask.to(device)), labels.to(device), weight=weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        true, predicted = _evaluate(model, val_loader, device)
        balanced = balanced_accuracy_score(true, predicted)
        print(f"clip epoch {epoch + 1}/{cfg.train.epochs}: val balanced accuracy {balanced:.4f}", flush=True)
        if balanced > best_balanced:
            best_balanced = balanced
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = (true, predicted)
    model.load_state_dict(best_state)
    true, predicted = best_metrics
    labels = list(range(12))
    matrix = confusion_matrix(true, predicted, labels=labels).tolist()
    report = {"val_balanced_accuracy": best_balanced, "labels": list(cfg.classes) + ["normal"],
              "confusion_matrix": matrix}
    run_root = Path(cfg.paths.root) / cfg.paths.runs
    torch.save({"state_dict": model.state_dict(), "d_in": fit_dataset.store.shape[1]}, run_root / "clip_head.pt")
    score = _write_r3_submission(model, cfg, device)
    report["public_dev_score"] = score
    with (run_root / "clip_head_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report


def train_temporal(cfg):
    """TemporalHead on SpliceDataset synthetic videos.

    Val is the FIXED synthetic set from `val` donors, scored through scorer.py so the number
    we optimize is the number the leaderboard pays. Early-stop on that, not on loss.
    AMP on, gradient clipping at 1.0. Variable-length sequences: bucket by length or pad with
    an attention mask - never truncate a 1258-frame L3 video to a fixed window.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.splice.seed)
    np.random.seed(cfg.splice.seed)
    table = make_donor_split(load_clip_table(cfg), cfg.splice.val_donor_frac, cfg.splice.seed)
    train_data = SpliceDataset(table, cfg, "fit", seed=cfg.splice.seed)
    synth_videos, synth_gt = build_synth_val(table, cfg)
    real_videos, real_gt = build_real_holdout(table, cfg)
    # A whole synthetic video is one transformer sequence.  Keeping batches at one avoids
    # padding a 240s L2 example up to a 650s L3 example and fits the 6GB target GPU.
    loader = DataLoader(train_data, batch_size=None, num_workers=0)
    model = TemporalHead(train_data.store.shape[1], cfg).to(device)
    counts = {index: 1 for index in range(12)}
    counts.update(table[table.donor_split == "fit"].class_idx.value_counts().to_dict())
    weights = class_weights(counts, cfg.train.class_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    history, best_state, best_gate = [], None, None
    fallback_state, fallback_gate = None, None
    for epoch in range(cfg.train.epochs):
        model.train()
        running, steps = 0.0, 0
        for emb, labels, boundary, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=bool(cfg.train.amp and device.type == "cuda")):
                frame_logits, boundary_logits = model(emb.to(device))
                loss, _ = losses(frame_logits, boundary_logits, labels.to(device), boundary.to(device), weights, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            steps += 1
        real_gate = _temporal_gate(model, real_videos, real_gt, cfg, device)
        synth_gate = _temporal_gate(model, synth_videos, synth_gt, cfg, device)
        metric = {"epoch": epoch + 1, "train_loss": running / max(steps, 1),
                  "real_holdout": real_gate, "synthetic_validation": synth_gate}
        history.append(metric)
        print(f"temporal epoch {epoch + 1}/{cfg.train.epochs}: loss {metric['train_loss']:.4f}; "
              f"real P={real_gate['precision']:.3f} R={real_gate['recall']:.3f} "
              f"FA-rate={real_gate['false_alarm_rate']:.3f}", flush=True)
        # Select by the actual scorer quantity, but only after the hard component floors
        # rule out silence / a statistically meaningless handful of emissions.
        qualifies = (real_gate["recall"] >= cfg.train.holdout_min_recall and
                     real_gate["emitted"] >= cfg.train.holdout_min_emitted)
        metric["qualifies_checkpoint_gate"] = qualifies
        score = real_gate["scorer"]["total"]
        if qualifies and (best_gate is None or score > best_gate["scorer"]["total"]):
            best_gate = real_gate
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if fallback_gate is None or real_gate["recall"] > fallback_gate["recall"]:
            fallback_gate = real_gate
            fallback_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    run_root = Path(cfg.paths.root) / cfg.paths.runs
    gate_reachable = best_state is not None
    model.load_state_dict(best_state if gate_reachable else fallback_state)
    torch.save({"state_dict": model.state_dict(), "d_in": train_data.store.shape[1]}, run_root / "temporal_head.pt")
    final_real = _temporal_gate(model, real_videos, real_gt, cfg, device)
    final_synth = _temporal_gate(model, synth_videos, synth_gt, cfg, device)
    report = {"history": history, "checkpoint_gate_reachable": gate_reachable,
              "checkpoint_selection": ("max_real_holdout_scorer_total_among_qualified_epochs"
                                       if gate_reachable else "max_real_holdout_recall_fallback"),
              "checkpoint_eligibility": {"min_recall": cfg.train.holdout_min_recall,
                                           "min_emitted": cfg.train.holdout_min_emitted},
              "selected_real_holdout": final_real,
              "selected_synthetic_validation": final_synth, "synthetic_validation_videos": len(synth_videos),
              "real_holdout_videos": len(real_videos),
              "real_holdout_anomalies": int(real_gt.is_anomaly.sum()),
              "real_holdout_normals": int((~real_gt.is_anomaly).sum())}
    with (run_root / "temporal_head_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("clip", "temporal"))
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    from .config import load
    if args.stage == "clip":
        train_clip(load(args.config))
    else:
        train_temporal(load(args.config))


if __name__ == "__main__":
    main()
