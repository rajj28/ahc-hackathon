"""Is the 130-video real holdout a VALID proxy for the eval set?

The eval L2/L3 events are short relative to their video: L2 events are 5-60s inside 240s
videos (2-25% coverage), L3 events 2.6-125s inside 308-629s videos (0.4-20%). If the long
train holdout instead has events covering most of their video, then a model tuned to emit
tight intervals can never match them, and the R4 gate is measuring the wrong thing.
"""
import glob, os, numpy as np, pandas as pd

MIN_DUR = 30.0
rows = []
for cd in sorted(glob.glob(r"train\*")):
    if not os.path.isdir(cd):
        continue
    cls = os.path.basename(cd)
    g = pd.read_csv(os.path.join(cd, "ground_truth.csv"))
    for _, r in g.iterrows():
        npz = os.path.join("cache", "siglip2", "train", cls, str(r.video_id) + ".npz")
        if not os.path.exists(npz):
            continue
        with np.load(npz) as z:
            dur = float(z["dur"])
        if dur <= MIN_DUR:
            continue
        anom = str(r.is_anomaly).lower() == "true"
        ev = (float(r.end_time_sec) - float(r.start_time_sec)) if anom else np.nan
        rows.append(dict(cls=cls, vid=r.video_id, dur=dur, ev=ev,
                         cov=(ev / dur) if anom else np.nan, anom=anom))

d = pd.DataFrame(rows)
a = d[d.anom]
print("HOLDOUT: %d videos  (%d anomalous, %d normal)" % (len(d), len(a), (~d.anom).sum()))
print("\n--- event coverage (event_dur / video_dur) on the %d anomalous ---" % len(a))
print("  min=%.3f  p25=%.3f  median=%.3f  p75=%.3f  max=%.3f"
      % (a.cov.min(), a.cov.quantile(.25), a.cov.median(), a.cov.quantile(.75), a.cov.max()))
for t in (0.5, 0.7, 0.9, 0.99):
    print("  coverage > %.2f : %3d / %d  (%.0f%%)" % (t, (a.cov > t).sum(), len(a), 100 * (a.cov > t).mean()))
print("\n--- by class ---")
print(a.groupby("cls").agg(n=("cov", "size"), med_dur=("dur", "median"),
                           med_ev=("ev", "median"), med_cov=("cov", "median")).round(2))

print("\n--- EVAL SET coverage, for comparison ---")
te = pd.read_csv(r"test\ground_truth.csv")
import cv2
durs = {}
for v in pd.read_csv(r"test\videos.csv").itertuples(index=False):
    c = cv2.VideoCapture(os.path.join("test", v.filename.replace("/", os.sep)))
    fps, n = c.get(cv2.CAP_PROP_FPS), c.get(cv2.CAP_PROP_FRAME_COUNT); c.release()
    durs[v.video_id] = n / fps if fps else np.nan
ev = te[(te.level.isin([2, 3])) & (te.is_anomaly.astype(str).str.lower() == "true")].copy()
ev["dur"] = ev.video_id.map(durs)
ev["cov"] = (ev.end_time_sec - ev.start_time_sec) / ev.dur
print("  n=%d  min=%.3f  median=%.3f  max=%.3f" % (len(ev), ev.cov.min(), ev.cov.median(), ev.cov.max()))
print("  coverage > 0.5 : %d / %d" % ((ev.cov > 0.5).sum(), len(ev)))
print("\n  per level:")
for lv in (2, 3):
    s = ev[ev.level == lv].cov
    print("    L%d n=%2d median=%.3f max=%.3f" % (lv, len(s), s.median(), s.max()))
