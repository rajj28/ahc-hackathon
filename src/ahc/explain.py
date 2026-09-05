"""Event explanations - the +5 reasoning bonus.

Strictly off the critical path. `explanation` is optional, 20-500 chars, and omitting it never
costs points, so an explanation failure must NEVER block or delay a submission. Templates
first; the VLM only if the clock allows.
"""

CLASS_PHRASE = {
    "traffic_accident": "A collision between vehicles",
    "traffic_congestion": "Dense stop-and-go traffic",
    "stalled_or_broken_down_vehicle": "A vehicle stopped in the roadway",
    "vehicle_blocking_traffic": "A vehicle obstructing the active carriageway",
    "fire": "An open fire",
    "smoke": "Thick smoke",
    "waterlogging_or_flood": "Standing water covering the roadway",
    "wrong_way_driving": "A vehicle travelling against the flow of traffic",
    "road_spill_or_debris": "Debris or spilled material on the road surface",
    "fighting_or_violence": "A physical altercation between people",
    "loitering_or_suspicious_presence": "A person lingering in the area for a prolonged period",
}


def template(event, level: int, dur: float) -> str:
    """Deterministic, always valid, always 20-500 chars.

    Level 1:   "<phrase> is visible throughout the clip."
    Level 2/3: "<phrase> observed from Xs to Ys."
    Pad short strings rather than emitting anything under 20 chars.
    """
    phrase = CLASS_PHRASE[event["class_name"] if isinstance(event, dict) else event.class_name]
    if level == 1:
        text = f"{phrase} is visible throughout the clip."
    else:
        start = event["start_time_sec"] if isinstance(event, dict) else event.start_time_sec
        end = event["end_time_sec"] if isinstance(event, dict) else event.end_time_sec
        text = f"{phrase} observed from {float(start):.1f}s to {float(end):.1f}s."
    return text if len(text) >= 20 else text + " This was detected locally."


def vlm_explain(frames, event, cfg) -> str | None:
    """Optional 4-bit Qwen2.5-VL-3B pass over the peak-scoring frame of an accepted event.

    Constraints: 6 GB card, and every second here is charged against the RTF latency bonus, so
    cap total VLM calls and run them only on accepted events. Wrap in try/except and return
    None on any failure - the caller falls back to template() silently.
    Truncate to 500 chars; reject and fall back if the model returns under 20.
    """
    # Hosted or heavyweight VLMs must never be an inference dependency.  Template mode is
    # intentionally the only runtime implementation; callers silently fall back to it.
    return None


def attach(preds, cfg):
    """Fill `explanation` on every event. Uses cfg.explain.mode; template mode is the default
    and the safe default.
    """
    for prediction in preds:
        for event in prediction.get("events", []):
            level = 1 if event.get("start_time_sec") is None else 2
            duration = 0.0 if level == 1 else float(event["end_time_sec"] - event["start_time_sec"])
            explanation = None
            if cfg.explain.mode == "vlm":
                explanation = vlm_explain(None, event, cfg)
            event["explanation"] = explanation or template(event, level, duration)
    return preds
