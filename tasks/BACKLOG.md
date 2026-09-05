# Task cards for Codex

Hand these over one at a time, in order. Each card is self-contained: read `CODEX.md` first
for the contracts, then implement only the card. Every card has a gate — do not move on until
the gate passes.

---

## R0 — Scorer, validator, floor submission  ⏱ 45 min  🔒 blocks everything

Implement `config.py`, `types.py`, `scorer.py`, `validate.py`.
Write `runs/sub_R0.json`: every dev video present, `events: []`, `runtime_metadata` filled.

**Gate:** `scorer.score_submission(sub_R0, public_gt)` returns **total ≈ 15.8** with
D1 ≈ 4.2, D2 ≈ 11.7, D3 = 0.0. If it does not, the scorer is wrong — fix it before anything
else, because every later decision is measured with it.
Also: `validate.py` must pass this file clean, and must flag a deliberately broken copy
(class_name "normal", a timestamp at L1, a missing runtime_metadata) with the right field names.

---

## R1 — Feature cache  ⏱ 45 min, mostly unattended

Implement `features.py`. Cache `test` then `train` under `siglip2`.

**Gate:** 34 test + 3,173 train `.npz` files exist; re-running the command extracts nothing
(resumability); a random `.npz` has `emb.dtype == float16`, `len(ts) == len(emb)`, and
`ts[-1] <= dur`. Run train extraction in the background and start R2/R3 while it finishes.

---

## R2 — Encoder bake-off  ⏱ 30 min  (skippable if the clock is tight)

`features.bakeoff` over ~300 clips (30/class) for siglip2, clip_b16, dinov2. Linear probe,
balanced accuracy, plus extraction wall-time.

**Gate:** a table. Pick on accuracy, break ties on speed. If siglip2 wins or ties, keep it and
do not re-extract — re-extraction costs 45 min and the margin is rarely worth it.

---

## R3 — Level 1 head  ⏱ 60 min

`dataset.ClipDataset`, `heads.ClipHead`, `train.train_clip`. Then predict L1 only and submit.

**Gate:** D1 > 17 on public dev (all-anomaly guessing gives ~15.4). Report the confusion
matrix; fire/smoke and congestion/blocking/stalled are the expensive confusions.

---

## R4 — Splice generator + temporal head  ⏱ 90 min  ⭐ the core of the system

`dataset.SpliceDataset`, `dataset.build_synth_val`, `heads.TemporalHead`,
`train.train_temporal`. See `CODEX.md` §3 — implement the recipe exactly, including the
donor split, `same_scene_prob`, and `pure_normal_frac`.

Also build the **real long-video validation set**: the 130 train videos over 30 s (69
anomalous, 61 normal), held out of splicing entirely. These are the only genuine untrimmed
footage we have, and they are the honest test of whether a head trained on synthetic splices
transfers to real video.

**Gate — two parts, both required.**

*(a) Public dev:* D2 > 12 with naive thresholding only (no changepoint snapping yet).

*(b) Real long-video holdout — report these two COMPONENT numbers, not a total:*

  1. **precision ≥ 70%** over all emitted events (matched ÷ emitted)
  2. **≤ 10%** false-alarm rate across the 61 normal videos
  3. recall ≥ 25% — a floor, not a target; do not trade precision for it

Weighted this way because of the observed leaderboard (CODEX.md §0b): the leader holds 100%
precision with 22% recall at D2 and scores 85% of the available marks, while an entry with
comparable recall and FA=94 scores 25%. Recall is the cheap axis; precision is the expensive
one. An earlier draft of this card set a 40% recall bar — that was wrong and is superseded.

Why components and not a total: on this holdout a total score is gameable. Scored L2-style at
`w = (0.2, 0.5, 0.3)`, all-silent gets **0.469**, and a model that alerts on every anomalous
video while localizing *nothing whatsoever* gets **0.575** — it beats silence purely on the
`w_alert` term. So any total-score gate below ~0.70 certifies a detector that cannot localize,
which is exactly the capability R5 is about to be built on top of. The two component numbers
cannot be faked by alerting: pure alerting scores 0% on (1) and fails outright. Together they
imply a total around 0.75, but read the components — those are what segmentation inherits.

If synthetic val looks strong and the real holdout fails, the splice distribution is wrong.
Fix the generator — most likely `same_scene_prob` is too low and the head has learned to
detect cuts rather than anomalies — rather than pressing on to R5. Building segmentation on a
head that only finds seams wastes the highest-leverage hour of the day.

Before either gate, sanity-check the generator by rendering 3 synthetic videos' label tracks
to a plot and eyeballing that spans land where the metadata says.

---

## R5 — Segmentation  ⏱ 60 min  ⭐ biggest single score jump

`segment.py` in full: smooth → hysteresis → changepoint snap → gap bridge → duration prior →
expand.

**Gate:** D2 and D3 both improve over R4, and mean IoU on *matched* events rises. Ablate the
snap step specifically and record its delta — it is the piece that exploits how the eval L2
videos are constructed, and the number belongs in the write-up.

---

## R6 — Calibration and per-level thresholds  ⏱ 45 min

`calibrate.py`. Temperature scaling, reliability curve, then the per-level sweep.

**Gate:** chosen thresholds raise the **minimum** total across `scorer.sensitivity()`'s whole
grid, not just the default weighting. Record the margin. If a setting wins under one
weighting and loses under another, reject it.

---

## R7 — Duration priors and gap bridging tuning  ⏱ 30 min

Sweep `expand`, `min_dur_sec`, `max_gap_sec` per class where there is enough val data.

**Gate:** improvement across the sensitivity grid, else revert. Expect small gains; stop early
if flat.

---

## R8 — Explanations  ⏱ 20 min

`explain.py` template mode on every event. VLM mode only if the clock is comfortable.

**Gate:** all explanations 20–500 chars; `validate.py` clean; total score unchanged (this
must not perturb detection).

---

## R9 — Latency trim  ⏱ 20 min

Measure RTF. Consider raising sampling fps only where it demonstrably buys IoU; lower it where
it does not. Confirm timings exclude model load.

**Gate:** RTF reported honestly and score not reduced.

---

## Final deliverables (not a benchmark run, editable freely)

- **Code repository** — a URL a host can open.
- **Architecture write-up** — `ARCHITECTURE.md` → HTML/PDF, ≤25 MB. A diagram beats prose.
- **Notes** — limitations, how to run, what more time would buy.

## Standing rules

- Measure on `scorer.py` with `sensitivity()` or the improvement does not exist.
- No per-video special-casing; the leaderboard set is private and different.
- Submit early and often — rejections are free, and the latest upload always wins, so never
  upload something unvalidated late in the day.
