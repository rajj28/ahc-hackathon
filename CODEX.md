# CODEX BRIEF — AHC Video Anomaly Detection

You are implementing a system whose architecture is already decided. **Do not redesign it.**
Implement the contracts below exactly. If a contract is ambiguous, implement the literal
reading and leave a `# ASSUMPTION:` comment; do not invent new modules or stages.

Working dir: `C:\Users\samar\ahc`. Interpreter: `.venv\Scripts\python.exe` (Python 3.12,
torch 2.6.0+cu124, CUDA available, RTX 4050 6 GB). Windows / PowerShell 5.1 — no `&&`.

---

## 0. The competition in one screen

Private eval pack of **28 videos** (`E001…`) delivered on the day via `manifest.json`,
which gives each `video_id` its `level`. We submit one JSON. Score = `25·L1 + 35·L2 + 40·L3`
(+5 speed, +5 reasoning bonus = 110 max).

- **Level 1** — one label for the whole clip, timestamps **must be null**. Pooled across all
  L1 videos: half anomaly-vs-normal accuracy, half class accuracy.
- **Level 2 / 3** — every event with start+end. Scored **per video, then averaged**.
  - GT normal → predict nothing = 1, predict anything = **0**.
  - GT has events → weighted mix of *did you alert*, *matched events*, *timing quality*.
    Timing weighs more at L3.
  - An event matches only when class is right **and** temporal IoU ≥ 0.5. Greedy one-to-one:
    at most one prediction matches a GT event, the rest count against you.

**COMPETITION RULE — binding on the architecture.** From the problem statement:
*"The system must run in real time on limited GPU capability."* and *"Larger hosted models can
be used during development, for comparison, or to generate training data, but they cannot be
part of what makes the detector work at runtime."*

So: Gemini, NVIDIA NIM, gpt-5.6-luna and any other hosted model may be used offline to build
training data or to compare against — **never in the inference path**, explanations included.
The local RTX 4050 is not a limitation we are working around, it is the target the brief
describes. Every runtime component must be local.

**We hold an NVIDIA NIM API key. It is OFFLINE-ONLY.** Do not import it, reference it, or add
a network call to `predict.py`, `explain.py`, `segment.py` or anything else on the inference
path — not even for the `explanation` field. Permitted uses are exactly two:
  1. generating a static explanation phrase bank that is then **baked into the source** as
     literal text, with no runtime call;
  2. producing a comparison baseline scored with our own `scorer.py`, written to `runs/`.
Both happen in a standalone script under `tools/`, never inside `src/ahc/`.
The key lives in an environment variable. Never hardcode it, never commit it — the code
repository is a submitted deliverable that a judge will open, so a committed key is a
published key. Ensure `.gitignore` covers `.env` and `*.key`.

Hard rejection rules (a rejected file is free, but wastes wall-clock):
1. `"class_name": "normal"` is rejected — express normal as `"events": []`.
2. Timestamps on a Level-1 event are rejected — must be `null`.
3. `runtime_metadata` is **required on every video**.
4. An omitted video is **not cleared** — it keeps its previous answer, and a never-answered
   video is scored as normal.
5. Latest upload always wins. There is no best-of. Max 5 MB.

The 11 classes (exact strings):
```
traffic_accident  traffic_congestion  stalled_or_broken_down_vehicle  vehicle_blocking_traffic
fire  smoke  waterlogging_or_flood  wrong_way_driving  road_spill_or_debris
fighting_or_violence  loitering_or_suspicious_presence
```
plus `normal`, which is **never** emitted as a class_name.

---

## 0b. OBSERVED LEADERBOARD BEHAVIOUR — read before tuning any threshold

The judging dashboard reports, per difficulty: **marks, P (precision), R (recall), found
(x/y), FA (false alarms)**. Observed standings:

| participant | model | D1 (P/R/found/FA) | D2 | D3 | total |
|---|---|---|---|---|---|
| Yash Waghmare | `probe`, 24 runs | 25.0 (100/100, 20/20, 0) | 29.9 (100/22, 4/18, 0) | 37.2 (100/50, 4/8, 0) | **92.1** |
| Aryan Varale | qwen3vl4b-lora | 10.6 (50/35, 7/20, 7) | 25.1 (38/28, 5/18, 8) | 12.0 (25/25, 2/8, 6) | 51.1 |
| Neeraj Gupta | cascade-appearance-vlm | 17.5 (64/70, 14/20, 8) | 23.7 (30/17, 3/18, 7) | 8.0 (0/0, 0/8, 6) | 49.2 |
| Manikandan V. | Qwen2.5-VL-3B+lora | 13.2 (38/55, 11/20, 18) | 8.9 (2/11, 2/18, **94**) | 11.2 (1/13, 1/8, **67**) | 34.2 |
| Aniruddha More | qwen3-vl-4b-lora | 2.6 (0/0, 0/20, 3) | 15.2 (0/0, 0/18, 2) | 2.0 (0/0, 0/8, 2) | 19.8 |

**Three things this settles.**

1. **`w_alert = 0.2`, confirmed.** At L3 every video is anomalous, so alerting is free credit.
   Neeraj alerted on 4/4 with zero matches -> 8.0/40 = 0.20/video. Aniruddha alerted on 1/4
   -> 2.0/40 = 0.05/video. Two independent points, both exact.

2. **PRECISION DOMINATES RECALL.** The leader scores 85% of D2 on **22% recall** and 93% of
   D3 on **50% recall**, purely by holding precision at 100% and FA at 0. Manikandan has
   comparable recall with FA=94 and scores 8.9/35. Do NOT tune for coverage. Emit few,
   confident events. A missed event is cheap; a false one is ruinous.

3. **The eval set mirrors the public dev set exactly.** The `found` denominators are 20 / 18 /
   8. Public dev has 20 anomalous L1 videos, 18 L2 events (6+4+4+4) and 8 L3 events (1+4+2+1).
   So thresholds calibrated on public dev transfer. On the day, check whether
   `manifest.json` lists `T001`-`T034`.

**Policy that follows:** alert on every L3 video (0.2 each, no normal videos to lose), but
emit only high-confidence intervals. Bias `cfg.segment.hi` UP, not down. When in doubt between
one confident event and three speculative ones, emit the one.

## 1. Measured facts you must design around

These are measured, not assumed. Do not re-derive them.

**Train** — 3,173 clips, one row per clip: 973 `normal`, 2,200 anomalous.
Class counts: traffic_accident 565, loitering 300, traffic_congestion 268,
stalled_vehicle 223, wrong_way 164, road_spill 151, vehicle_blocking 148,
fighting 124, waterlogging 95, smoke 85, fire 77. **Heavily imbalanced — use class weights.**

**Train clips are pre-trimmed to the event.** 63% of anomalies span >90% of their own clip:

| class | clip_dur | event_dur | ev/clip | cover>0.9 |
|---|---|---|---|---|
| loitering_or_suspicious_presence | 29.9 | 30.0 | 1.00 | 100% |
| waterlogging_or_flood | 5.7 | 5.7 | 1.00 | 100% |
| fighting_or_violence | 25.8 | 25.8 | 1.00 | 98% |
| traffic_congestion | 5.5 | 5.3 | 0.99 | 92% |
| fire | 5.8 | 5.8 | 1.00 | 90% |
| traffic_accident | 5.0 | 5.0 | 1.00 | 80% |
| smoke | 5.8 | 5.8 | 1.00 | 78% |
| stalled_or_broken_down_vehicle | 12.4 | 7.8 | 0.75 | 42% |
| vehicle_blocking_traffic | 11.0 | 5.0 | 0.45 | 10% |
| road_spill_or_debris | 7.9 | 3.1 | 0.32 | 2% |
| wrong_way_driving | 16.2 | 5.0 | 0.28 | 5% |

**Consequence: the raw train timestamps carry almost no boundary information.** A naive
onset/offset regressor would fit `start=0, end=duration`. This is why §3 exists.

Normal clips: median 19.9 s. These are the background donors — and they include some very long
real footage: a 3599 s (60 min) video, a 2747 s, a 1025 s, and 6 more over 300 s.

**Long-video subset — the only REAL untrimmed supervision we have.** 130 train videos exceed
30 s, 91 exceed 60 s. Every one still carries exactly one ground-truth event, so a long video
with a short event has a genuine, non-degenerate boundary:

| threshold | videos | of which normal | anomalous |
|---|---|---|---|
| >30 s | 130 | 61 | 69 |
| >60 s | 91 | 49 | 42 |
| >120 s | 37 | 24 | 13 |

69 anomalous long videos is far too few to train on, but it is exactly the right size for a
**real validation set**. Use it to check that a head trained on synthetic splices transfers to
genuine untrimmed footage — see R4.

**Native frame rates:** 2,376 of 3,173 train videos are **15 fps**, then 25 fps (302), 1.88 fps
(300), 30 fps (149). Do not assume 25/30. Total train footage is 17.4 hours.

**Doc/data discrepancy:** the dataset PDF states train `ground_truth.csv` carries a `level`
column. **It does not.** Only `test/ground_truth.csv` has one. Code to the data, not the doc.

**Public dev set** (`test/`, 34 videos — a *dev* set, NOT the leaderboard):

| level | videos | normal | anomalous | durations |
|---|---|---|---|---|
| L1 | 24 (T001–T024) | 4 (17%) | 20 | 5.7–26 s |
| L2 | 6 (T025–T030) | 2 (33%) | 4 | all exactly 240.0 s |
| L3 | 4 (T031–T034) | **0 (0%)** | 4 | 308–629 s |

All-silent submission scores **15.8/100** (D1 4.2, D2 11.7, D3 **0.0**).
At L1, 83% of videos are anomalous → bias toward alerting. At L3 (public) silence is a
guaranteed zero. **Abstention policy must be per level, never global.**

**The L2 videos are constructed by splicing.** T025 = six 20 s accidents at 20/60/100/140/180/220
(40 s pitch). T028 = four 5 s accidents at 30/90/150/210 (60 s pitch). T027 = congestion
intervals of 5,5,60,5 s. T026 = four *different* classes in one 240 s video. All six L2
videos are exactly 240.0 s. This is the single most exploitable structural fact we have.

---

## 2. Pipeline (fixed)

```
video ──► sample @2 fps ──► FROZEN ENCODER ──► emb[T,D] cached .npz
                                                    │
                        ┌───────────────────────────┴──────────────────────┐
                        ▼ (level 1)                                        ▼ (level 2/3)
                  attention-pool                                    TEMPORAL HEAD
                        │                                    per-frame logits[T,12]
                  clip logits[12]                            + boundaryness[T,1]
                        │                                            │
                  calibrate + threshold                    hysteresis + changepoint snap
                        │                                            │
                  one event, ts=null                        intervals, class-conditional
                        │                                    duration prior, gap-bridge
                        └───────────────────┬────────────────────────┘
                                            ▼
                                  explanation (template ▸ VLM)
                                            ▼
                                   submission.json + validate
```

Everything downstream of the encoder operates on **cached embeddings**, never pixels.
That is what makes the experiment loop take seconds.

---

## 3. THE KEY IDEA — synthetic splice training (implement this carefully)

The problem: we have no boundary supervision and no negative frames inside anomalous videos.
The solution: **manufacture L2/L3-style videos the same way the organizers did, in embedding
space.** Concatenating embedding sequences is free, so we can generate tens of thousands of
synthetic long videos per epoch at zero decode cost.

Recipe for one synthetic video:
1. Draw a target duration from `{240 s}` (L2 style) or `U(300, 650) s` (L3 style).
2. Tile **normal** clip embeddings end-to-end to fill it (loop/shuffle donors).
3. Choose `k ~ {1..6}` events and a class per event (uniform over classes, not frequency —
   this rebalances the tail classes for free). For L2 style, optionally use a regular pitch;
   for L3 style, place them randomly with a min gap.
4. For each event, splice in a whole trimmed anomaly clip's embedding sequence at that offset.
   If the clip is longer than the slot, crop; if shorter, use it whole and shrink the slot.
5. Emit frame labels: anomaly class inside spliced spans, `normal` elsewhere.
6. Emit a `boundary` target: 1 at the exact splice frames, smoothed with a ±1-frame Gaussian.

This gives us, all at once: real boundary labels, in-video negatives, class rebalancing,
and a distribution matched to how the eval L2 set is actually built.

**Also generate held-out "pure normal" synthetic videos with k=0** — these are what calibrate
the false-positive rate that decides the normal-video score at L2/L3.

Two guards, both required:
- **Donor split.** Partition train clips into `fit` / `val` donor pools *before* splicing. A
  clip may never appear in both. Otherwise validation is meaningless.
- **Splice realism.** Real splices come from different source footage, so a cosine
  discontinuity at the seam is genuine signal — but do not let the head win by detecting
  *only* seams. Include a fraction (~25%) of synthetic videos where the "event" is a segment
  of the *same* donor scene (no seam), so the head must also use appearance.

---

## 4. Module contracts

Package root `src/ahc/`. Stubs exist with signatures; fill the bodies.

### `config.py`
Loads `configs/default.yaml` into a dotted-access object. All thresholds, weights, paths,
and encoder names live in YAML. **No magic numbers in code.**

### `features.py`
```python
ENCODERS = {"siglip2": ..., "clip_b16": ..., "dinov2": ...}   # registry

def load_encoder(name: str) -> tuple[Callable[[Tensor], Tensor], np.ndarray, np.ndarray, int]:
    """-> (encode_fn, mean, std, input_size). fp16, .cuda().eval(), no_grad."""

def sample_frames(path: str, size: int, mean, std, target_fps: float
                  ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sequential cap.grab()/retrieve() — never seek. -> (frames[N,3,S,S] f16, ts[N] f32,
    duration_sec, native_fps). Videos range 1.88–30 fps; step = max(1, round(fps/target_fps))."""

def embed_video(path, encoder, cfg) -> dict   # {"emb": f16[N,D], "ts": f32[N], "dur":…, "fps":…}
def build_cache(split: Literal["train","test"], encoder_name: str, cfg) -> None
```
Cache layout — **this is a contract, other modules glob it**:
```
cache/{encoder}/train/{class_name}/{video_id}.npz
cache/{encoder}/test/{video_id}.npz
cache/{encoder}/eval/{video_id}.npz      # the private pack on the day
```
Each `.npz` holds `emb` (float16 `[N,D]`), `ts` (float32 `[N]`), `dur`, `fps`.
Skip a video whose `.npz` already exists (resumable — this is important, extraction is
30–45 min and must survive interruption). Log progress + ETA every 25 videos.

### `ingest.py` — dataset acquisition (LOW PRIORITY, likely never runs)

The local copy is **verified complete**: 3,173/3,173 train + 34/34 test, none missing, none
zero-byte, none undecodable, no orphans, 13.9 GB. Four further Drive mirrors are listed in
`configs/default.yaml` — they are redundant copies, published so the cohort would not all hit
one folder.

Implement this only when explicitly asked. It matters in exactly one realistic case: the
organizers publish a corrected `ground_truth.csv` mid-event. `probe --mirror N` answers that
for a few KB by diffing CSV rows, without pulling video.

Two traps, both already hit once:
- `gdown --folder` silently caps at **50 files per folder**; `train/normal/videos/` holds 973.
  Never trust it for video. rclone, or a manual browser download, or nothing.
- Drive's multi-part browser download produces independent archives, not a split archive.
  Extract each separately into the same destination. On PowerShell 5.1 use
  `Expand-Archive -Force` — `ZipFile.ExtractToDirectory(src, dst, True)` does not exist there.

`merge` is additive and never overwrites a good local file. After any real merge, re-run
`verify()` and invalidate the affected `.npz` cache entries or the features go stale.

### `dataset.py`
```python
def load_clip_table(cfg) -> pd.DataFrame
    """All train clips: video_id, class_name, is_anomaly, start, end, cache_path, donor_split."""

def make_donor_split(df, val_frac=0.2, seed=0) -> pd.DataFrame   # adds donor_split col

class ClipDataset(Dataset)      # -> (emb[T,D], class_idx)  for the L1 clip head
class SpliceDataset(IterableDataset)
    """Implements §3. Yields (emb[T,D], frame_labels[T], boundary[T], meta).
    Deterministic under a seed so val is fixed."""
```

### `heads.py`
```python
class ClipHead(nn.Module):
    """Attention-pool over [T,D] -> logits[12]. ~0.5M params."""

class TemporalHead(nn.Module):
    """[T,D] -> (frame_logits[T,12], boundary[T]).
    Conv1d stem (k=5, d=256) -> 2-layer TransformerEncoder (4 heads, dropout 0.1)
    -> two linear heads. Target ~3-6M params. Must accept variable T (no fixed pos-emb;
    use sinusoidal or none). Input is L2-normalized emb concat with its 1-frame delta."""
```
Losses: class-weighted cross-entropy (weights = `1/sqrt(count)`, normalized) + BCE on
boundary, weighted `cfg.train.boundary_weight`.

### `segment.py` — the IoU engine, highest leverage module
```python
def smooth(scores: np.ndarray, win: int) -> np.ndarray            # centered moving average
def changepoints(emb: np.ndarray, k: int) -> np.ndarray
    """Cosine distance between adjacent frames -> local maxima -> candidate seam indices."""
def hysteresis(p: np.ndarray, hi: float, lo: float) -> list[tuple[int,int]]
    """Schmitt trigger: open a span when p>hi, extend while p>lo."""
def snap_to_changepoints(spans, cps, max_shift: int) -> list
    """Move each span edge to the nearest changepoint within max_shift frames. This is where
    the IoU>=0.5 gate is won — splice seams are sharp."""
def bridge_gaps(spans, max_gap_sec, ts) -> list
def apply_duration_prior(spans, class_name, cfg, ts) -> list
    """Clamp to the class's [p10,p90] train duration; expand short spans toward the class
    median. NEVER merge fragments of different classes."""
def extract_events(frame_probs, boundary, emb, ts, level, cfg) -> list[Event]
```
Geometry to respect (do not re-derive): a centered prediction of duration `k·D` has
`IoU = min(k, 1/k)`, so feasible `k ∈ [0.5, 2]`. Equal duration with center offset `δ` gives
`IoU = (D-δ)/(D+δ)`, so `δ ≤ D/3`. Slight over-prediction is safer than under; default
`cfg.segment.expand = 1.15` but it is **swept, not fixed**.

### `scorer.py` — build this FIRST, before any model
Local replica of the official metric. The L2/L3 mix weights are **not published**, so:
```python
def iou(a, b) -> float
def match_events(pred, gt, thr=0.5) -> list[tuple[int,int]]
    """Greedy by descending IoU, one-to-one, class must match exactly."""
def score_video_l23(pred, gt, level, w) -> float
    """w = (w_alert, w_match, w_time).
       normal GT: 1.0 if len(pred)==0 else 0.0
       else: w_alert*1[len(pred)>0] + w_match*F1(matched) + w_time*mean_IoU(matched)
       F1 not recall — unmatched predictions must cost, per 'the rest count against you'."""
def score_submission(pred_json, gt_df, w_l2, w_l3) -> dict
    """-> {'L1':…, 'L2':…, 'L3':…, 'D1':…, 'D2':…, 'D3':…, 'total':…}"""
def sensitivity(pred_json, gt_df) -> pd.DataFrame
    """Re-score across a GRID of plausible weightings, e.g.
       w_l2 ∈ {(.2,.5,.3), (.3,.4,.3), (.34,.33,.33)}, w_l3 with more timing weight
       {(.2,.4,.4), (.2,.3,.5), (.3,.3,.4)}.
    Report min/median/max total. A change only counts as an improvement if it wins across
    the WHOLE grid. This is a hard rule — it stops us optimizing into an assumption."""
```
Defaults: `w_l2 = (0.2, 0.5, 0.3)`, `w_l3 = (0.2, 0.4, 0.4)` ("timing weighs more at L3").

### `calibrate.py`
```python
def fit_temperature(logits, labels) -> float
def fit_isotonic(probs, labels)
def sweep_thresholds(cfg, cache, gt) -> dict
    """Grid over hi ∈ {.2,.3,.4,.5,.6,.7,.8}, lo = hi - {.05,.1,.2}, expand ∈ {1.0,1.15,1.3},
    min_dur, max_gap — PER LEVEL. Select on synthetic val + public dev, and only accept a
    setting that wins across the scorer's full sensitivity grid. Writes runs/thresholds.json."""
```
Never hardcode an abstention threshold. Report the chosen value and its margin.

### `predict.py`
```python
def predict_video(path, models, cfg) -> VideoPred
def main(manifest_path, out_path, cfg)
```
Branch on the manifest's `level`:
- **L1** → single event, `start_time_sec=None`, `end_time_sec=None`. Emit `events: []` only
  when `p(normal)` clears the L1 normal threshold. Prior is 83% anomalous — lean toward alerting.
- **L2/L3** → intervals from `segment.extract_events`.

Time each video with `time.perf_counter()` around decode+infer+postprocess, **excluding model
load**. Populate `runtime_metadata` honestly: `frames_processed`, `chunks_processed`,
`end_to_end_internal_time_ms`. If a `model_runtimes` entry gives total/count/average, the
average must equal total÷count within 2% or the file is rejected.

### `validate.py` — run before every upload
Schema pre-flight. Must catch all seven documented rejection causes plus: unknown class
string, `end <= start`, duplicate `video_id`, `video_id` not in manifest, missing
`runtime_metadata`, timestamps present at L1, `class_name == "normal"`, file > 5 MB,
`explanation` outside 20–500 chars. Exit non-zero with a per-video table.

### `explain.py`
Template first, VLM second. Template: `f"{class_phrase[cls]} observed from {start:.0f}s to
{end:.0f}s."` padded to ≥20 chars. Only if time remains, a 4-bit Qwen2.5-VL-3B pass over the
peak frame of each accepted event. **Never on the critical path — an explanation failure must
not block the submission.** Omitting `explanation` never costs points.

---

## 5. Build order (each stage must beat the previous on `scorer.py`)

| # | Stage | Gate |
|---|---|---|
| R0 | `scorer.py` + `validate.py` + all-empty submission | Reproduces 15.8/100 on public dev |
| R1 | `features.py`, cache test + train (siglip2) | ✅ DONE — 3,173 + 34 `.npz`, 0 missing, 0 corrupt, 768-dim, 129,210 frames, 0.18 GB |
| R2 | Encoder bake-off on 300-clip subset | pick winner by linear-probe accuracy |
| R3 | `ClipHead` → L1 only | D1 > 17 |
| R4 | `SpliceDataset` + `TemporalHead` | D2 > 12 |
| R5 | `segment.py` hysteresis + changepoint snap | D2 + D3 jump; this is the big one |
| R6 | `calibrate.py` per-level sweep | wins across full sensitivity grid |
| R7 | Duration priors + gap bridging | marginal IoU gains |
| R8 | `explain.py` templates | +reasoning bonus |
| R9 | RTF trim (raise fps only where it pays) | speed bonus without accuracy loss |

**Submit R0 early.** It is free (rejections don't consume runs), it validates the JSON path
end to end, and it banks a non-zero floor before anything can go wrong.

---

## 6. Rules of engagement

- Every claimed improvement is measured on `scorer.py` with `sensitivity()`, or it does not exist.
- No per-video special-casing. The leaderboard set is private and different.
- Resumable everything — a crash at 17:30 must not cost more than a minute.
- Config-driven. If Codex wants to try a value, it goes in YAML.
- Prefer boring, working code. There are seven hours.
