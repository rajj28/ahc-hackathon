"""Score curve -> intervals. Highest-leverage module: the IoU>=0.5 gate is won or lost here.

Geometry (settled, do not re-derive):
  - Prediction centered on GT, duration k*D  ->  IoU = min(k, 1/k).  Feasible k in [0.5, 2].
  - Equal duration, center offset d          ->  IoU = (D-d)/(D+d).  Needs d <= D/3.
  So: modest over-prediction beats under-prediction, and CENTER ACCURACY matters more than
  duration accuracy. Fragmenting one real event into several spans is strictly harmful.

Why changepoints matter: the eval L2 videos are built by splicing trimmed anomaly clips into
normal background (measured: T025 six 20s events on a 40s pitch, T028 four 5s on a 60s pitch,
all L2 videos exactly 240.0s). Splices are sharp discontinuities in embedding space, so
snapping span edges to them lands boundaries far more precisely than thresholding alone.
"""
import numpy as np

from .types import Event


def smooth(scores: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average, edge-padded. Length preserved."""
    if win <= 1:
        return np.asarray(scores, dtype=np.float32).copy()
    values = np.asarray(scores, dtype=np.float32)
    left, right = win // 2, win - 1 - win // 2
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.full(win, 1.0 / win, dtype=np.float32), mode="valid")


def changepoints(emb: np.ndarray, k: int) -> np.ndarray:
    """Candidate seam frames.

    Cosine distance between adjacent L2-normalized frames -> 1D signal -> local maxima ->
    top-k indices. Returns sorted frame indices.
    """
    if len(emb) < 3 or k <= 0:
        return np.empty(0, dtype=np.int64)
    unit = emb.astype(np.float32, copy=False)
    unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-8)
    distance = 1.0 - np.einsum("ij,ij->i", unit[1:], unit[:-1])
    # A changepoint at i is the edge between i-1 and i, hence the +1 index.
    peaks = np.flatnonzero((distance[1:-1] >= distance[:-2]) &
                           (distance[1:-1] >= distance[2:])) + 1
    if not len(peaks):
        return np.empty(0, dtype=np.int64)
    chosen = peaks[np.argsort(distance[peaks - 1])[-min(k, len(peaks)):]]
    return np.sort(chosen.astype(np.int64))


def hysteresis(p: np.ndarray, hi: float, lo: float) -> list[tuple[int, int]]:
    """Schmitt trigger. Open a span where p > hi, extend left and right while p > lo.

    Double-thresholding beats a single threshold because it keeps one event whole instead of
    chopping it into fragments at every dip below the line.
    """
    values = np.asarray(p, dtype=np.float32)
    result, start = [], None
    for index, value in enumerate(values):
        if start is None:
            if value > hi:
                start = index
        elif value <= lo:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, len(values)))
    return result


def snap_to_changepoints(spans: list, cps: np.ndarray, max_shift: int) -> list:
    """Move each span edge to the nearest changepoint within max_shift frames.

    Never let snapping collapse a span to zero length or cross a neighbouring span's edge.
    """
    if not spans or not len(cps):
        return list(spans)
    result = []
    for span in spans:
        start, end, *rest = span
        candidates = cps[np.abs(cps - start) <= max_shift]
        snapped_start = int(candidates[np.argmin(np.abs(candidates - start))]) if len(candidates) else start
        candidates = cps[np.abs(cps - end) <= max_shift]
        snapped_end = int(candidates[np.argmin(np.abs(candidates - end))]) if len(candidates) else end
        if snapped_end <= snapped_start:
            snapped_start, snapped_end = start, end
        result.append((snapped_start, snapped_end, *rest))
    return result


def bridge_gaps(spans: list, max_gap_sec: float, ts: np.ndarray) -> list:
    """Merge same-class spans separated by less than max_gap_sec.

    Only ever merges spans of the SAME class. T026 holds four different classes in one video
    and merging across them would destroy both events.
    """
    if not spans:
        return []
    limit = float(max_gap_sec)
    ordered = sorted(spans, key=lambda item: (item[2] if len(item) > 2 else "", item[0]))
    merged = []
    for span in ordered:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        same_class = len(span) > 2 and len(previous) > 2 and span[2] == previous[2]
        gap = float(ts[span[0]] - ts[previous[1] - 1]) if span[0] > 0 else 0.0
        if same_class and gap <= limit:
            merged[-1] = (previous[0], span[1], *previous[2:])
        else:
            merged.append(span)
    return sorted(merged, key=lambda item: item[0])


def apply_duration_prior(spans: list, class_name: str, priors: dict, ts: np.ndarray) -> list:
    """Clamp span length into the class's [p10, p90] train duration; expand spans shorter
    than p10 symmetrically about their centre toward the class median.

    Priors come from the train ground truth (medians: accident 5.0s, congestion 5.3s,
    fire 5.8s, smoke 5.8s, waterlogging 5.7s, spill 3.1s, blocking 5.0s, wrong_way 5.0s,
    stalled 7.8s, fighting 25.8s, loitering 30.0s). Expansion is symmetric to protect the
    centre, which is what the IoU geometry above says dominates.
    """
    prior = priors.get(class_name) if priors else None
    if not prior:
        return list(spans)
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.5
    result = []
    for start, end, *rest in spans:
        duration = float(ts[end - 1] - ts[start] + step)
        if duration < prior["p10"]:
            desired = prior["p50"]
            centre = (start + end - 1) / 2
            half = max(1, int(round(desired / step / 2)))
            start, end = max(0, int(np.floor(centre - half))), min(len(ts), int(np.ceil(centre + half + 1)))
        result.append((start, end, *rest))
    return result


def extract_events(frame_probs: np.ndarray, boundary: np.ndarray, emb: np.ndarray,
                   ts: np.ndarray, level: int, cfg, snap: bool = True) -> list:
    """Full pipeline: per-class anomaly curve -> smooth -> hysteresis -> snap -> bridge ->
    duration prior -> expand by cfg.segment.expand -> drop spans under min_dur -> Event list.

    Runs per class independently, then concatenates. Emits at most one span per contiguous
    detection: fragments are merged, not listed separately.
    Level 1 never calls this — it takes the single argmax label with null timestamps.
    """
    del boundary, level  # The R5 decoder uses embedding changepoints; L1 never enters here.
    cps = changepoints(emb, cfg.segment.changepoint_k) if snap else np.empty(0, dtype=np.int64)
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / cfg.features.target_fps
    priors = getattr(cfg.segment, "duration_priors", {})
    events = []
    for class_idx, class_name in enumerate(cfg.classes):
        curve = smooth(frame_probs[:, class_idx], cfg.segment.smooth_win)
        spans = hysteresis(curve, cfg.segment.hi, cfg.segment.lo)
        spans = snap_to_changepoints(spans, cps, cfg.segment.snap_max_shift_frames)
        spans = [(start, end, class_name) for start, end in spans]
        spans = bridge_gaps(spans, cfg.segment.max_gap_sec, ts)
        spans = apply_duration_prior(spans, class_name, priors, ts)
        for start, end, _ in spans:
            centre = (start + end - 1) / 2
            half = (end - start) * cfg.segment.expand / 2
            start = max(0, int(np.floor(centre - half)))
            end = min(len(ts), int(np.ceil(centre + half + 1)))
            start_sec, end_sec = float(ts[start]), float(ts[end - 1] + step)
            if end_sec - start_sec >= cfg.segment.min_dur_sec:
                events.append(Event(class_name, start_sec, end_sec, float(curve[start:end].mean())))
    return sorted(events, key=lambda event: event.score, reverse=True)


def duration_priors_from_train(clip_table) -> dict:
    """-> {class_name: {'p10':…, 'p50':…, 'p90':…}} from train ground truth.

    CAVEAT: train clips are pre-trimmed, so for 7 of 11 classes the event spans ~the whole
    clip. These priors describe how long an event of that class TENDS to be, which is still
    useful, but they are not evidence about untrimmed footage. Do not use them to justify
    predicting a whole test video as one event.
    """
    rows = clip_table[clip_table.is_anomaly].copy()
    duration = (rows.end_time_sec.astype(float) - rows.start_time_sec.astype(float)).clip(lower=0)
    rows = rows.assign(_duration=duration)
    return {class_name: {"p10": float(group._duration.quantile(.1)),
                         "p50": float(group._duration.quantile(.5)),
                         "p90": float(group._duration.quantile(.9))}
            for class_name, group in rows.groupby("class_name")}
