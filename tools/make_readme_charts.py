"""Render the README charts from docs/results.json (numbers measured in ARCHITECTURE.md).

    python tools/make_readme_charts.py      # needs matplotlib
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs" / "results.json").read_text(encoding="utf-8"))
OUT = ROOT / "docs" / "images"


def _clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def score_chart() -> None:
    levels = ["D1", "D2", "D3"]
    labels = ["Level 1\nclip label", "Level 2\nevent intervals", "Level 3\nlong-video events"]
    got = [R["dev_score"][k] for k in levels]
    cap = [R["max_marks"][k] for k in levels]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150,
                               gridspec_kw={"width_ratios": [1.3, 1]})
    a.bar(labels, cap, color="#d8dee9", label="max marks")
    a.bar(labels, got, color="#1f6feb", label="scored")
    for i, (g, c) in enumerate(zip(got, cap)):
        a.text(i, g + 0.6, f"{g:.2f} / {c}", ha="center", fontsize=9)
    a.set_title(f"Public dev pack: {R['dev_score']['total']:.2f} / 100 (34 videos)", fontsize=10)
    a.legend(frameon=False, fontsize=8), _clean(a)

    abl = R["ablation_grid_min"]
    names = list(abl)
    vals = [abl[n] for n in names]
    colors = ["#2da44e", "#8c959f", "#8c959f"]
    b.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        b.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=9)
    b.set_xlim(44, 49.5), b.invert_yaxis()
    b.set_title("Ablations, scorer-grid minimum (axis starts at 44)", fontsize=10), _clean(b)
    fig.tight_layout()
    fig.savefig(OUT / "dev-score.png")
    plt.close(fig)


def runtime_chart() -> None:
    rt = R["runtime"]
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=150)
    ax.barh(["video duration", "processing time"], [rt["video_s"], rt["wall_s"]],
            color=["#d8dee9", "#2da44e"])
    ax.text(rt["video_s"] + 30, 0, f"{rt['video_s']:.0f} s", va="center", fontsize=9)
    ax.text(rt["wall_s"] + 30, 1, f"{rt['wall_s']:.2f} s  (RTF {rt['rtf']})", va="center", fontsize=9)
    ax.set_xlim(0, rt["video_s"] * 1.2), ax.set_xlabel("seconds")
    ax.set_title("Full MP4 → validated JSON on one RTX 4050 laptop GPU (6 GB)", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "runtime.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    score_chart()
    runtime_chart()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
