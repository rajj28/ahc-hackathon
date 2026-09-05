"""Frozen encoder registry + frame sampling + embedding cache.

Cache layout is a CONTRACT - other modules glob it:
  cache/{encoder}/train/{class_name}/{video_id}.npz
  cache/{encoder}/test/{video_id}.npz
  cache/{encoder}/eval/{video_id}.npz     # the private 28 on the day
Each .npz: emb float16 [N,D], ts float32 [N], dur float32, fps float32.

Extraction is 30-45 min over 3173 train clips, so it MUST be resumable: skip any video whose
.npz already exists. A crash at 17:30 must cost seconds, not a rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np


ENCODERS = ("siglip2", "clip_b16", "dinov2")


def load_encoder(name: str):
    """-> (encode_fn, mean, std, input_size). fp16, .cuda().eval(), inference under no_grad.

    siglip2  google/siglip2-base-patch16-224 vision tower, pooler_output
    clip_b16 open_clip ViT-B-16 laion2b_s34b_b88k, visual
    dinov2   facebook/dinov2-base, CLS token
    """
    import torch

    if name == "siglip2":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
        model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").cuda().eval().half()
        image_processor = getattr(processor, "image_processor", processor)
        mean = np.asarray(image_processor.image_mean, dtype=np.float32)
        std = np.asarray(image_processor.image_std, dtype=np.float32)
        size = image_processor.size
        input_size = int(size.get("height", size.get("shortest_edge")))

        @torch.no_grad()
        def encode(images):
            output = model.get_image_features(pixel_values=images)
            return output.pooler_output if hasattr(output, "pooler_output") else output
    elif name == "clip_b16":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k")
        model = model.cuda().eval().half()
        mean = np.asarray(model.visual.image_mean, dtype=np.float32)
        std = np.asarray(model.visual.image_std, dtype=np.float32)
        image_size = model.visual.image_size
        input_size = int(image_size[0] if isinstance(image_size, tuple) else image_size)

        @torch.no_grad()
        def encode(images):
            return model.encode_image(images)
    elif name == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
        image_processor = getattr(processor, "image_processor", processor)
        mean = np.asarray(image_processor.image_mean, dtype=np.float32)
        std = np.asarray(image_processor.image_std, dtype=np.float32)
        size = image_processor.size
        input_size = int(size.get("height", size.get("shortest_edge")))

        @torch.no_grad()
        def encode(images):
            return model(pixel_values=images).last_hidden_state[:, 0]
    else:
        raise ValueError(f"unknown encoder {name!r}; choices are {ENCODERS}")
    return encode, mean, std, input_size


def sample_frames(path: str, size: int, mean, std, target_fps: float):
    """-> (frames[N,3,S,S] float16, ts[N] float32, duration_sec, native_fps)

    Sequential cap.grab() + retrieve() only - never seek, it is slow and unreliable on these
    files. step = max(1, round(native_fps / target_fps)).
    Native fps in this dataset ranges 1.88 to 30.0; at 1.88 fps every frame is kept.
    Guard against fps <= 0 or NaN (fall back to 25.0).
    """
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise OSError(f"could not open video: {path}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = 25.0
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / native_fps if frame_count > 0 else 0.0
    step = max(1, round(native_fps / target_fps))
    samples, timestamps, index = [], [], 0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    try:
        while capture.grab():
            if index % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    index += 1
                    continue
                rgb = cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR),
                                   cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                samples.append(np.transpose((rgb - mean) / std, (2, 0, 1)).astype(np.float16))
                timestamps.append(index / native_fps)
            index += 1
    finally:
        capture.release()
    if index and duration <= 0:
        duration = index / native_fps
    return (np.stack(samples) if samples else np.empty((0, 3, size, size), dtype=np.float16),
            np.asarray(timestamps, dtype=np.float32), float(duration), float(native_fps))


def embed_video(path: str, encoder, cfg) -> dict:
    """Sample then batch through the encoder. -> {'emb','ts','dur','fps'}."""
    import torch

    encode, mean, std, input_size = encoder
    frames, timestamps, duration, native_fps = sample_frames(
        str(path), input_size, mean, std, cfg.features.target_fps)
    batches = []
    for start in range(0, len(frames), cfg.features.batch):
        batch = torch.from_numpy(frames[start:start + cfg.features.batch]).cuda(non_blocking=True)
        with torch.no_grad():
            batches.append(encode(batch).detach().float().cpu().numpy().astype(np.float16))
    embedding_dim = 0 if not batches else batches[0].shape[1]
    return {"emb": np.concatenate(batches, axis=0) if batches else np.empty((0, embedding_dim), np.float16),
            "ts": timestamps.astype(np.float32), "dur": np.float32(duration), "fps": np.float32(native_fps)}


def build_cache(split: str, encoder_name: str, cfg, manifest: str | None = None) -> None:
    """split in {train, test, eval}. `eval` reads the manifest for its video list.

    Skips existing .npz. Logs progress and ETA every 25 videos. A single video failing must
    log and continue, never abort the run.
    """
    if split not in {"train", "test", "eval"}:
        raise ValueError("split must be train, test, or eval")
    root = Path(cfg.paths.root)
    source_root = root / ("test" if split in {"test", "eval"} else "train")
    output_root = root / cfg.paths.cache / encoder_name / split
    if split == "train":
        records = []
        for class_name in cfg.classes + ["normal"]:
            table_path = source_root / class_name / "videos.csv"
            if not table_path.exists():
                continue
            import pandas as pd
            for row in pd.read_csv(table_path).itertuples(index=False):
                records.append((str(row.video_id), source_root / class_name / row.filename,
                                output_root / class_name / f"{row.video_id}.npz"))
    elif split == "test":
        import pandas as pd
        records = [(str(row.video_id), source_root / row.filename, output_root / f"{row.video_id}.npz")
                   for row in pd.read_csv(source_root / "videos.csv").itertuples(index=False)]
    else:
        if manifest is None:
            raise ValueError("eval cache requires a manifest path")
        manifest_path = Path(manifest)
        # Public rehearsal keeps manifest and videos under test/. On the day the delivered
        # pack may live elsewhere; prefer the manifest's sibling videos/ directory.
        if (manifest_path.parent / "videos").exists():
            source_root = manifest_path.parent
        if manifest_path.suffix.lower() == ".csv":
            import pandas as pd
            entries = pd.read_csv(manifest_path).to_dict("records")
        else:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest_data = json.load(handle)
            entries = manifest_data.get("videos", manifest_data) if isinstance(manifest_data, dict) else manifest_data
        if isinstance(entries, dict):
            entries = [{"video_id": video_id, **(value if isinstance(value, dict) else {})}
                       for video_id, value in entries.items()]
        records = []
        for entry in entries:
            video_id = str(entry["video_id"])
            filename = entry.get("filename", f"videos/{video_id}.mp4")
            records.append((video_id, source_root / filename, output_root / f"{video_id}.npz"))

    encoder = load_encoder(encoder_name)
    started = time.perf_counter()
    completed = 0
    for position, (video_id, input_path, output_path) in enumerate(records, 1):
        if output_path.exists():
            completed += 1
            continue
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            video_started = time.perf_counter()
            result = embed_video(input_path, encoder, cfg)
            result["extract_time_ms"] = np.float32((time.perf_counter() - video_started) * 1000)
            temporary_path = output_path.with_suffix(".tmp")
            with temporary_path.open("wb") as handle:
                np.savez_compressed(handle, **result)
            os.replace(temporary_path, output_path)
            completed += 1
        except Exception as exc:
            print(f"[{split}] {video_id}: FAILED: {exc}", flush=True)
        if position % 25 == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed / position * (len(records) - position)
            print(f"[{split}] {position}/{len(records)} complete; ETA {eta / 60:.1f} min", flush=True)
    print(f"[{split}] cache complete: {completed}/{len(records)}", flush=True)


def bakeoff(cfg, n_per_class: int = 30) -> "pd.DataFrame":
    """R2: compare encoders on a ~300-clip subset with a linear probe.

    Extract that subset under each encoder, fit logistic regression on pooled embeddings,
    report balanced accuracy on held-out clips + extraction wall-time. Pick on accuracy;
    break ties on speed, because RTF feeds the latency bonus. Cheap enough to be worth it,
    small enough not to cost an hour.
    """
    raise NotImplementedError("R2 is not part of R1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "test", "eval"))
    parser.add_argument("--encoder", default=None, choices=ENCODERS)
    parser.add_argument("--manifest")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    from .config import load
    cfg = load(args.config)
    build_cache(args.split, args.encoder or cfg.features.encoder, cfg, args.manifest)


if __name__ == "__main__":
    main()
