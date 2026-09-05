"""Fill the OFFICIAL starter template with our predictions.

The template uses "model_runtimes": [] and the same per-video shape we were already sending,
which rules our structure out as the cause of the save error. Filling their object in place
guarantees identical key sets and ordering.

Timestamps are clamped against the MANIFEST duration_sec (rounded, e.g. T034 = 376.5 while
the real file is 376.533) because that is the number the backend validates against.
"""
import json, sys
sys.path.insert(0, "src")
from ahc.scorer import load_gt, score_submission, sensitivity

TEMPLATE = r"C:\Users\samar\Downloads\submission-template.json"
MANIFEST = r"C:\Users\samar\Downloads\manifest.json"
OURS = r"runs\sub_A.json"

tpl = json.load(open(TEMPLATE))
man = json.load(open(MANIFEST))["videos"]
ours = {p["video_id"]: p for p in json.load(open(OURS))["predictions"]}
dur = {m["video_id"]: float(m["duration_sec"]) for m in man}
lvl = {m["video_id"]: int(m["level"]) for m in man}
gt = load_gt(r"test\ground_truth.csv")

# untouched copy, for the diagnostic upload
json.dump(tpl, open(r"runs\sub_TEMPLATE_RAW.json", "w"), indent=2)

filled = json.loads(json.dumps(tpl))          # preserve their key order exactly
filled["submission_id"] = "ahc-run-01"
filled["model_name"] = "siglip2-cliphead-window-decoder"
filled["run_metadata"] = {"total_wall_time_ms": 88360, "max_parallel_videos": 1,
                          "hardware": "1x RTX 4050 Laptop 6GB"}

n_clamp = n_drop = 0
for p in filled["predictions"]:
    vid = p["video_id"]
    src = ours.get(vid)
    if src is None:
        continue
    events = []
    for e in src.get("events", []):
        ev = {"class_name": e["class_name"]}
        if lvl[vid] == 1:
            ev["start_time_sec"] = None
            ev["end_time_sec"] = None
        else:
            st = round(float(e["start_time_sec"]), 2)
            en = round(float(e["end_time_sec"]), 2)
            lim = round(dur[vid] - 0.1, 2)
            if en > lim:
                en = lim; n_clamp += 1
            if en <= st:
                n_drop += 1; continue
            ev["start_time_sec"] = st
            ev["end_time_sec"] = en
        if e.get("explanation"):
            ev["explanation"] = e["explanation"]
        events.append(ev)
    p["events"] = events
    rm = src.get("runtime_metadata", {})
    p["runtime_metadata"] = {
        "frames_processed": int(rm.get("frames_processed", 0)),
        "chunks_processed": max(1, int(rm.get("chunks_processed", 1))),
        "end_to_end_internal_time_ms": int(round(float(rm.get("end_to_end_internal_time_ms", 0)))),
        "model_runtimes": [],                 # exactly as the official template does
    }

json.dump(filled, open(r"runs\sub_TEMPLATE_FILLED.json", "w"), indent=2)

nev = sum(len(p["events"]) for p in filled["predictions"])
sc = score_submission(filled, gt)
print("template videos      : %d" % len(tpl["predictions"]))
print("filled events        : %d   clamped=%d dropped=%d" % (nev, n_clamp, n_drop))
print("size                 : %.1f KB" % (len(json.dumps(filled).encode()) / 1024))
print("score                : %.2f  (D1 %.2f  D2 %.2f  D3 %.2f)  grid min %.2f"
      % (sc["total"], sc["D1"], sc["D2"], sc["D3"], sensitivity(filled, gt).total.min()))

bad = [(p["video_id"], e["end_time_sec"], dur[p["video_id"]])
       for p in filled["predictions"] for e in p["events"]
       if e.get("end_time_sec") is not None and e["end_time_sec"] >= dur[p["video_id"]]]
print("events outside manifest duration: %d %s" % (len(bad), bad[:4]))

