"""Independent re-score of a submission + where the marks actually come from."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from ahc.scorer import load_gt, score_submission, iou, match_events, sensitivity

sub = json.load(open(r"runs\sub_R4_fallback.json"))
gt = load_gt(r"test\ground_truth.csv")
print("re-scored:", {k: round(v, 2) for k, v in score_submission(sub, gt).items()})

pred = {p["video_id"]: p for p in sub["predictions"]}
print("\n%-6s %-3s %-6s %-6s %-8s  %s" % ("vid", "lv", "n_gt", "n_pred", "matched", "best IoU per GT event"))
tot_found = {2: 0, 3: 0}; tot_gt = {2: 0, 3: 0}; tot_emit = {2: 0, 3: 0}
near = []
for vid, g in gt.groupby("video_id", sort=True):
    lv = int(g.level.iloc[0])
    if lv == 1: continue
    anom = bool(g.is_anomaly.iloc[0])
    gts = g.to_dict("records") if anom else []
    ps = pred.get(vid, {}).get("events", [])
    m = match_events(ps, gts) if gts else []
    tot_gt[lv] += len(gts); tot_emit[lv] += len(ps); tot_found[lv] += len(m)
    ious = []
    for ge in gts:
        best = 0.0; bestc = "-"
        for pe in ps:
            if pe.get("start_time_sec") is None: continue
            v = iou((pe["start_time_sec"], pe["end_time_sec"]),
                    (ge["start_time_sec"], ge["end_time_sec"]))
            if v > best: best, bestc = v, pe["class_name"]
        ious.append("%.2f%s" % (best, "" if bestc == ge["class_name"] else "!" ))
        if 0.2 < best < 0.5: near.append((vid, ge["class_name"], bestc, round(best, 2)))
    print("%-6s L%-2d %-6d %-6d %-8d  %s" % (vid, lv, len(gts), len(ps), len(m), " ".join(ious)))

print("\n%-4s %-8s %-8s %-8s %-9s %-8s" % ("lvl", "gt", "emitted", "found", "precision", "recall"))
for lv in (2, 3):
    p = tot_found[lv] / tot_emit[lv] if tot_emit[lv] else 0.0
    r = tot_found[lv] / tot_gt[lv] if tot_gt[lv] else 0.0
    print("L%-3d %-8d %-8d %-8d %-9.1f%% %-7.1f%%" % (lv, tot_gt[lv], tot_emit[lv], tot_found[lv], 100*p, 100*r))

print("\nNEAR MISSES (0.2 < IoU < 0.5)  '!' = wrong class:")
for n in near: print("   ", n)

print("\nSENSITIVITY across weighting grid:")
s = sensitivity(sub, gt, None)
print(s[["total"]].describe().loc[["min", "50%", "max"]].round(2).to_string())
