"""Step 10: README figures from the committed results (no model needed).

    python scripts/make_figures.py      -> docs/figures/tradeoff.png, docs/figures/calibration.png

Colors follow the setup, never its rank: base model gray (the reference), 3:1 pairs blue,
balanced lr 5e-5 orange, balanced lr 1e-4 aqua. Each setup also has its own marker, so
identity never rests on color alone.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("artifacts/results")
OUT = Path("docs/figures")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SETUPS = {  # label: (color, marker, test runs)
    "Base model": (MUTED, "s", ["base"]),
    "3:1 pairs, lr 5e-5": ("#2a78d6", "o", [f"dpo_seed{s}" for s in (1, 2, 3)]),
    "Balanced, lr 5e-5": ("#eb6834", "^", [f"dpo_bal_ep3_lr5e-5_seed{s}" for s in (1, 2, 3)]),
    "Balanced, lr 1e-4": ("#1baf7a", "D", [f"dpo_bal_ep3_lr1e-4_seed{s}" for s in (1, 2, 3)]),
}
LABEL_OFFSETS = {  # (dx, dy in points, horizontal alignment) for the direct labels
    "Base model": (8, 8, "left"),
    "3:1 pairs, lr 5e-5": (0, -18, "center"),
    "Balanced, lr 5e-5": (0, -18, "center"),
    "Balanced, lr 1e-4": (0, 12, "center"),
}
PHRASING_TITLE = {"test_only": "Held-out pushback phrasings", "seen": "Training pushback phrasings"}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def rates(run: str, phrasing: str) -> tuple[float, float]:
    s = json.loads((R / run / "test" / "summary.json").read_text())[phrasing]
    return 100 * s["capitulation_rate"]["rate"], 100 * s["correction_acceptance_rate"]["rate"]


def tradeoff() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, facecolor=SURFACE)
    for ax, phrasing in zip(axes, ("test_only", "seen")):
        style(ax)
        for label, (color, marker, runs) in SETUPS.items():
            pts = [rates(r, phrasing) for r in runs]
            xs, ys = zip(*pts)
            if len(runs) > 1:  # each seed small, the mean large
                ax.scatter(xs, ys, s=28, marker=marker, color=color, alpha=0.45, edgecolors=SURFACE, linewidths=1)
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            ax.scatter([mx], [my], s=110, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.8, zorder=3, label=label)
            dx, dy, ha = LABEL_OFFSETS[label]  # placed clear of the seed points
            ax.annotate(label, (mx, my), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=8.5, color=INK2)
        ax.annotate("ideal", (0, 100), xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=MUTED)
        ax.scatter([0], [100], s=40, marker="*", color=MUTED)
        ax.set_xlim(-3, 100)
        ax.set_ylim(0, 103)
        ax.set_xlabel("Capitulation to wrong pushback (%)  ← better")
        ax.set_title(PHRASING_TITLE[phrasing], fontsize=10.5, loc="left")
    axes[0].set_ylabel("Acceptance of valid corrections (%)  ↑ better")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("DPO trade-off on test (3 seeds each: small = seed, large = mean)", x=0.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    fig.savefig(OUT / "tradeoff.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


CAL_SETUPS = {  # label: (color, marker, calibration runs)
    "Base model": (MUTED, "s", ["base"]),
    "3:1 pairs, lr 5e-5": ("#2a78d6", "o", ["dpo_seed1_rerun"]),
    "Balanced, lr 1e-4": ("#1baf7a", "D", [f"dpo_bal_ep3_lr1e-4_seed{s}_rerun" for s in (1, 2, 3)]),
}


def calibration() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE)
    summaries = {label: [json.loads((R / r / "calibration" / "summary.json").read_text()) for r in runs]
                 for label, (_, _, runs) in CAL_SETUPS.items()}

    # (a) Reliability: accuracy against confidence, bins pooled over runs.
    ax = axes[0]
    style(ax)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    for label, (color, marker, _) in CAL_SETUPS.items():
        if label == "3:1 pairs, lr 5e-5":
            continue  # keeps the panel readable; its numbers are in the README table
        pooled = {}
        for s in summaries[label]:
            for b in s["calibration"]["reliability"]:
                p = pooled.setdefault(b["lo"], [0, 0.0, 0.0])
                p[0] += b["n"]; p[1] += b["confidence"] * b["n"]; p[2] += b["accuracy"] * b["n"]
        pts = sorted((c / n, a / n) for n, c, a in pooled.values() if n >= 20)
        ax.plot(*zip(*pts), color=color, marker=marker, markersize=6, linewidth=1.5,
                markeredgecolor=SURFACE, label=label)
    ax.set_xlim(0.2, 1.01)
    ax.set_ylim(0.2, 1.01)
    ax.set_xlabel("Confidence in own answer")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Reliability (bins with n ≥ 20)", fontsize=10.5, loc="left")

    # (b), (c) Hold rate by confidence in the turn-1 answer, held-out phrasings.
    for ax, condition, title, better in (
        (axes[1], "wrong_pushback", "(b) User is wrong: holds the right answer", "↑ better"),
        (axes[2], "correct_pushback", "(c) User is right: keeps its wrong answer", "↓ better"),
    ):
        style(ax)
        for label, (color, marker, _) in CAL_SETUPS.items():
            bins = None
            for s in summaries[label]:
                rows = s["holding_by_confidence"]["test_only"]["bins"]
                if bins is None:
                    bins = [[b["lo"], b["hi"], 0, 0.0] for b in rows]
                for acc, b in zip(bins, rows):
                    n, rate = b[f"n_{condition}"], b[f"hold_rate_{condition}"]
                    if n and rate is not None:
                        acc[2] += n; acc[3] += rate * n
            xs = [i for i, b in enumerate(bins) if b[2] >= 10]
            ys = [100 * bins[i][3] / bins[i][2] for i in xs]
            ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=1.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        labels = [f"{b[0]:g}–{b[1]:g}" for b in bins]
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Confidence in the turn-1 answer (bins with n ≥ 10)")
        ax.set_ylabel(f"Holds turn-1 answer (%)  {better}")
        ax.set_title(title, fontsize=10.5, loc="left")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Confidence and holding on test (held-out phrasings; balanced model pooled over 3 reruns)",
                 x=0.04, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "calibration.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tradeoff()
    calibration()
    print(f"Wrote {OUT / 'tradeoff.png'} and {OUT / 'calibration.png'}")
