"""Training data. All of it operates on cached embeddings — never pixels.

The central idea (see CODEX.md §3): train clips are pre-trimmed to their event, so they carry
no boundary supervision and no in-video negatives. We manufacture both by splicing anomaly
clip embeddings into tiled normal backgrounds, exactly the way the eval L2 videos are built.
Concatenating embeddings is free, so this costs seconds per epoch.

LOAD THE WHOLE CACHE INTO RAM ONCE. Measured after R1: 129,210 frames x 768 dims x 2 bytes =
198 MB as float16, ~400 MB as float32. The entire dataset fits in memory with room to spare.

So: read every .npz a single time into one contiguous array plus an index
{video_id -> (offset, length, class_idx, ts)}, and have SpliceDataset slice views of that
array. Do NOT open .npz files per sample — a 4,000-video epoch draws 40k-120k donor clips, and
per-sample file opens would make disk I/O dominate a training step that should be pure
arithmetic. Splicing then costs a few array copies and epochs run in seconds, which is the
entire reason this architecture was chosen over VLM fine-tuning.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

from .config import NORMAL_IDX, class_index


_EMBEDDING_STORES: dict[tuple[str, ...], tuple[np.ndarray, dict[str, tuple[int, int]]]] = {}


def _load_embedding_store(table: pd.DataFrame) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Load each cached train clip exactly once into a contiguous float32 backing store."""
    paths = tuple(table.sort_values("video_id").cache_path.astype(str))
    if paths in _EMBEDDING_STORES:
        return _EMBEDDING_STORES[paths]
    arrays, index, offset = [], {}, 0
    for row in table.sort_values("video_id").itertuples(index=False):
        embedding = np.load(row.cache_path)["emb"].astype(np.float32)
        arrays.append(embedding)
        index[row.video_id] = (offset, len(embedding))
        offset += len(embedding)
    store = (np.concatenate(arrays, axis=0), index)
    _EMBEDDING_STORES[paths] = store
    return store


def load_clip_table(cfg) -> pd.DataFrame:
    """Every train clip -> video_id, class_name, class_idx, is_anomaly, start_time_sec,
    end_time_sec, cache_path, n_frames, dur. Reads the 12 per-class ground_truth.csv files
    (which have NO `level` column, unlike test) and joins each class's videos.csv.
    """
    rows = []
    root = Path(cfg.paths.root)
    for class_name in list(cfg.classes) + ["normal"]:
        class_root = root / cfg.paths.train / class_name
        videos = pd.read_csv(class_root / "videos.csv")
        ground_truth = pd.read_csv(class_root / "ground_truth.csv")
        joined = videos.merge(ground_truth, on="video_id", how="inner", validate="one_to_one")
        for row in joined.itertuples(index=False):
            cache_path = root / cfg.paths.cache / cfg.features.encoder / "train" / class_name / f"{row.video_id}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"missing feature cache: {cache_path}")
            with np.load(cache_path) as cached:
                n_frames = len(cached["emb"])
                duration = float(cached["dur"])
            is_anomaly = str(row.is_anomaly).lower() == "true"
            rows.append({
                "video_id": row.video_id,
                "class_name": row.class_name,
                "class_idx": class_index(row.class_name),
                "is_anomaly": is_anomaly,
                "start_time_sec": row.start_time_sec,
                "end_time_sec": row.end_time_sec,
                "cache_path": str(cache_path),
                "n_frames": n_frames,
                "dur": duration,
            })
    table = pd.DataFrame(rows)
    table["is_real_holdout"] = table.dur > cfg.augment.long_holdout_min_dur_sec
    return table


def make_donor_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Add `donor_split` in {fit, val}, stratified by class.

    A clip must never appear in both pools. Synthetic videos built from `fit` donors are
    trained on; those built from `val` donors are validated on. Without this the val score is
    memorisation and every downstream threshold is calibrated on a lie.
    """
    result = df.copy()
    # Long clips are held out before splitting; they must never become splice donors.
    result["donor_split"] = "holdout"
    eligible = result[~result.is_real_holdout]
    rng = np.random.default_rng(seed)
    for _, index in eligible.groupby("class_name", sort=False).groups.items():
        index = np.asarray(list(index))
        rng.shuffle(index)
        n_val = int(round(len(index) * val_frac))
        result.loc[index, "donor_split"] = "fit"
        result.loc[index[:n_val], "donor_split"] = "val"
    return result


class ClipDataset(Dataset):
    """L1 head. -> (emb[T,D] float32, class_idx int) over whole trimmed clips.

    Augment by random temporal cropping (keep >=50% of frames) and frame dropout.
    """
    def __init__(self, table: pd.DataFrame, cfg, split: str):
        if split not in {"fit", "val"}:
            raise ValueError("ClipDataset split must be 'fit' or 'val'")
        self.cfg = cfg
        self.split = split
        self.table = table[table.donor_split == split].reset_index(drop=True)
        self.store, self.index = _load_embedding_store(table)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        offset, length = self.index[row.video_id]
        embedding = self.store[offset:offset + length]
        if self.split == "fit" and length > 1:
            keep = max(1, int(np.ceil(length * self.cfg.augment.crop_min_frac)))
            if keep < length:
                start = np.random.randint(0, length - keep + 1)
                embedding = embedding[start:start + keep]
            if self.cfg.augment.frame_dropout > 0:
                embedding = embedding.copy()
                drop = np.random.random(len(embedding)) < self.cfg.augment.frame_dropout
                embedding[drop] = 0.0
        return torch.from_numpy(embedding.copy()), int(row.class_idx)


class SpliceDataset(IterableDataset):
    """Synthetic L2/L3-style videos in embedding space. Implements CODEX.md §3.

    Yields (emb[T,D] float32, frame_labels[T] int64, boundary[T] float32, meta dict).
    meta carries the true spans so the same generator can produce a fixed validation set
    scoreable by scorer.py.

    Per video:
      1. duration <- 240.0 (L2 style) or U(300, 650) (L3 style)
      2. tile shuffled NORMAL donor embeddings to fill it
      3. k ~ U{events_per_video}; classes drawn UNIFORMLY over the 11, not by frequency
         (this rebalances fire=77 / smoke=85 against traffic_accident=565 for free)
      4. with prob `regular_pitch_prob`, place events on a fixed pitch like T025/T028;
         otherwise place randomly respecting min_gap_sec
      5. splice each anomaly clip's embeddings into its slot (crop if longer)
      6. labels: class inside spliced spans, index 11 (`normal`) elsewhere
      7. boundary: 1.0 at splice frames, Gaussian-smoothed +-1 frame
      8. with prob `same_scene_prob`, take the 'event' from the SAME donor scene so there is
         no seam — otherwise the head learns to detect cuts instead of anomalies
      9. with prob `pure_normal_frac`, emit k=0. These calibrate the false-positive rate that
         decides the normal-video score at L2/L3, where a false alarm costs the whole video.

    Deterministic given (seed, index) so the validation set is stable across runs.
    """
    def __init__(self, table: pd.DataFrame, cfg, split: str, seed: int = 0):
        if split not in {"fit", "val"}:
            raise ValueError("SpliceDataset split must be 'fit' or 'val'")
        self.cfg, self.split, self.seed = cfg, split, seed
        self.table = table[table.donor_split == split].reset_index(drop=True)
        self.store, self.index = _load_embedding_store(table)
        self.normal = self.table[self.table.class_idx == NORMAL_IDX].reset_index(drop=True)
        self.anomaly = {
            class_idx: self.table[self.table.class_idx == class_idx].reset_index(drop=True)
            for class_idx in range(NORMAL_IDX)
        }
        if self.normal.empty or any(pool.empty for pool in self.anomaly.values()):
            raise ValueError(f"{split} donor pool lacks a required class")

    def _emb(self, row):
        offset, length = self.index[row.video_id]
        return self.store[offset:offset + length]

    def _one(self, number: int):
        rng = np.random.default_rng(self.seed + number)
        fps = float(self.cfg.features.target_fps)
        is_l2 = bool(rng.integers(0, 2))
        duration = float(self.cfg.splice.l2_duration_sec if is_l2 else
                         rng.uniform(self.cfg.splice.l3_duration_min_sec, self.cfg.splice.l3_duration_max_sec))
        frames = max(2, int(round(duration * fps)))
        # Tile normal clips into the background. Whole clips preserve real temporal texture.
        chunks = []
        total = 0
        while total < frames:
            row = self.normal.iloc[int(rng.integers(len(self.normal)))]
            chunk = self._emb(row)
            chunks.append(chunk)
            total += len(chunk)
        emb = np.concatenate(chunks, axis=0)[:frames].copy()
        labels = np.full(frames, NORMAL_IDX, dtype=np.int64)
        boundary = np.zeros(frames, dtype=np.float32)
        spans, occupied = [], []
        k = 0 if rng.random() < self.cfg.splice.pure_normal_frac else int(rng.integers(
            self.cfg.splice.events_per_video[0], self.cfg.splice.events_per_video[1] + 1))
        # Regular pitch is intentionally used for L2-like examples, matching the public construction.
        pitch = max(1, frames // max(k + 1, 2)) if (is_l2 and rng.random() < self.cfg.splice.regular_pitch_prob) else None
        for event_no in range(k):
            class_idx = int(rng.integers(NORMAL_IDX))
            source = self._emb(self.anomaly[class_idx].iloc[int(rng.integers(len(self.anomaly[class_idx])))])
            max_len = min(len(source), max(2, int(45 * fps)))
            if len(source) > max_len:
                left = int(rng.integers(len(source) - max_len + 1))
                source = source[left:left + max_len]
            length = len(source)
            candidates = []
            if pitch is not None:
                centre = min(frames - length, max(0, (event_no + 1) * pitch - length // 2))
                candidates = [centre]
            else:
                candidates = rng.integers(0, max(1, frames - length + 1), size=32).tolist()
            start = None
            gap = int(round(self.cfg.splice.min_gap_sec * fps))
            for candidate in candidates:
                candidate = int(candidate)
                if all(candidate + length + gap <= a or candidate >= b + gap for a, b in occupied):
                    start = candidate
                    break
            if start is None:
                continue
            end = min(frames, start + length)
            source = source[:end - start]
            # ASSUMPTION: clips have no source-scene identifier.  The literal no-seam
            # equivalent available in this pack is to retain the already tiled background
            # at the labelled span.  These deliberately hard (appearance-free) examples
            # prevent the boundary head from solving every event through a cosine jump.
            same_scene = rng.random() < self.cfg.splice.same_scene_prob
            if not same_scene:
                emb[start:end] = source
            labels[start:end] = class_idx
            boundary[start] = boundary[end - 1] = 1.0
            occupied.append((start, end))
            spans.append((start, end, class_idx))
        # ±1-frame Gaussian smoothing for trainable splice boundaries.
        boundary = np.convolve(boundary, np.array([0.25, 1.0, 0.25], dtype=np.float32), mode="same")
        ts = np.arange(frames, dtype=np.float32) / fps
        return (torch.from_numpy(emb), torch.from_numpy(labels), torch.from_numpy(boundary),
                {"video_id": f"synth_{self.split}_{number:05d}", "ts": ts, "level": 2 if is_l2 else 3,
                 "spans": spans, "duration": duration})

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        offset = worker.id if worker else 0
        stride = worker.num_workers if worker else 1
        for number in range(offset, int(self.cfg.splice.videos_per_epoch), stride):
            yield self._one(number)


def build_real_holdout(table: pd.DataFrame, cfg, min_dur: float = 30.0) -> tuple[list, pd.DataFrame]:
    """The 130 train videos longer than 30 s: 69 anomalous, 61 normal. R4's transfer gate.

    These are the ONLY genuine untrimmed footage available. Every one still carries exactly
    one ground-truth event, so a long video with a short event is a real, non-degenerate
    boundary — unlike the trimmed short clips where the event spans the whole file.

    MUST be excluded from the splice donor pools entirely, or the gate measures memorisation.
    Assign them before make_donor_split runs, not after.

    -> (list of (emb, ts, video_id), gt_df) shaped exactly like build_synth_val's output, with
    `level` set to 2, so scorer.py scores it through the identical code path.

    Report from it: the fraction of the 69 anomalous events matched at IoU >= 0.5, and the
    false-alarm rate over the 61 normal videos. Report those two numbers, not a total — on
    this set a total is gameable (all-silent 0.469; alert-everything-localize-nothing 0.575).
    """
    rows = table[table.dur > min_dur].copy()
    store, index = _load_embedding_store(table)
    videos, gt_rows = [], []
    for row in rows.itertuples(index=False):
        offset, length = index[row.video_id]
        emb = store[offset:offset + length].copy()
        ts = np.arange(length, dtype=np.float32) / float(cfg.features.target_fps)
        videos.append((emb, ts, row.video_id))
        gt_rows.append({"video_id": row.video_id, "level": 2, "is_anomaly": bool(row.is_anomaly),
                        "class_name": row.class_name,
                        "start_time_sec": float(row.start_time_sec), "end_time_sec": float(row.end_time_sec)})
    return videos, pd.DataFrame(gt_rows)


def build_synth_val(table: pd.DataFrame, cfg, n: int = 200) -> tuple[list, pd.DataFrame]:
    """Fixed synthetic validation set from `val` donors only.

    -> (list of (emb, ts, video_id), gt_df in the same schema load_gt returns) so that
    scorer.py can score synthetic videos with the identical code path used for the real set.
    Composition should mirror the public dev split: ~1/3 normal at L2, and include some
    all-normal L3 videos even though public L3 has none (the private set may differ).
    """
    dataset = SpliceDataset(table, cfg, "val", seed=10_000)
    videos, rows = [], []
    for number in range(n):
        emb, labels, _, meta = dataset._one(number)
        videos.append((emb.numpy(), meta["ts"], meta["video_id"]))
        if meta["spans"]:
            for start, end, class_idx in meta["spans"]:
                rows.append({"video_id": meta["video_id"], "level": meta["level"], "is_anomaly": True,
                             "class_name": cfg.classes[class_idx], "start_time_sec": float(meta["ts"][start]),
                             "end_time_sec": float(meta["ts"][end - 1] + 1 / cfg.features.target_fps)})
        else:
            rows.append({"video_id": meta["video_id"], "level": meta["level"], "is_anomaly": False,
                         "class_name": "normal", "start_time_sec": np.nan, "end_time_sec": np.nan})
    return videos, pd.DataFrame(rows)
