"""Pre-flight against the LIVE submission page's field rules."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from ahc.scorer import load_gt, score_submission, sensitivity

PATH = r"runs\sub_R5_day_rehearsal.json"
CLASSES = {"traffic_accident","traffic_congestion","stalled_or_broken_down_vehicle",
           "vehicle_blocking_traffic","fire","smoke","waterlogging_or_flood",
           "wrong_way_driving","road_spill_or_debris","fighting_or_violence",
           "loitering_or_suspicious_presence"}

s = json.load(open(PATH))
gt = load_gt(r"test\ground_truth.csv")
levels = gt.groupby("video_id")["level"].first().to_dict()
manifest = list(pd.read_csv(r"test\videos.csv").video_id.astype(str))
durs = {}
for v in manifest:
    with np.load(f"cache/siglip2/test/{v}.npz") as z:
        durs[v] = float(z["dur"])

err, warn = [], []
seen, nev = [], 0
for p in s.get("predictions", []):
    vid = p.get("video_id"); seen.append(vid)
    if vid not in manifest: err.append(f"{vid}: not in manifest")
    for k in ("video_id", "events", "runtime_metadata"):
        if k not in p: err.append(f"{vid}: missing {k}")
    rm = p.get("runtime_metadata", {})
    for k in ("frames_processed", "chunks_processed", "end_to_end_internal_time_ms", "model_runtimes"):
        if k not in rm: err.append(f"{vid}: runtime_metadata missing {k}")
    for m in rm.get("model_runtimes", []):
        if {"total_time_ms","call_count","average_time_ms"} <= m.keys():
            t, c, a = m["total_time_ms"], m["call_count"], m["average_time_ms"]
            if c <= 0 or a == 0 or abs(a - t / c) / abs(a) > 0.02:
                err.append(f"{vid}: model_runtimes average {a} != {t}/{c}")
        if "call_times_ms" in m and len(m["call_times_ms"]) != m.get("call_count"):
            err.append(f"{vid}: call_times_ms length != call_count")
    lv = levels.get(vid)
    for i, e in enumerate(p.get("events", [])):
        nev += 1
        cn = e.get("class_name")
        if cn == "normal": err.append(f"{vid}[{i}]: class_name 'normal' forbidden")
        elif cn not in CLASSES: err.append(f"{vid}[{i}]: unknown class {cn}")
        st, en = e.get("start_time_sec"), e.get("end_time_sec")
        if lv == 1:
            if st is not None or en is not None: err.append(f"{vid}[{i}]: L1 timestamps must be null")
        else:
            if st is None or en is None: err.append(f"{vid}[{i}]: L{lv} timestamps required")
            else:
                if st < 0: err.append(f"{vid}[{i}]: start < 0")
                if en <= st: err.append(f"{vid}[{i}]: end <= start")
                if en >= durs[vid]: err.append(f"{vid}[{i}]: end {en:.2f} not inside duration {durs[vid]:.2f}")
        ex = e.get("explanation")
        if ex is not None and not (20 <= len(ex) <= 500): err.append(f"{vid}[{i}]: explanation length {len(ex)}")

dupes = [v for v in set(seen) if seen.count(v) > 1]
if dupes: err.append(f"duplicate video_id: {dupes}")
missing = [v for v in manifest if v not in seen]
if missing: err.append(f"manifest ids not answered: {missing}")
if "predictions" not in s: err.append("missing top-level predictions")
for k in ("submission_id", "model_name", "run_metadata"):
    if k not in s: warn.append(f"optional top-level '{k}' absent")

size = len(json.dumps(s).encode())
print(f"file: {PATH}")
print(f"videos answered : {len(seen)}/34     events: {nev}     size: {size/1024:.1f} KB (limit 5120)")
print(f"score           : {score_submission(s, gt)['total']:.2f}   grid min {sensitivity(s,gt).total.min():.2f}")
print(f"\nERRORS ({len(err)}):")
for e in err[:25]: print("   X", e)
print(f"\nWARNINGS ({len(warn)}):")
for w in warn: print("   !", w)
print("\nVERDICT:", "SAFE TO UPLOAD" if not err else "WILL LIKELY BE REJECTED")

