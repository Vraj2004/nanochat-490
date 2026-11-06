"""Plot the A3 Part 4 passkey accuracy against distance from the CSVs in a3/results/part4.

The CSVs are written by scripts/passkey_eval.py (run on Modal by modal_train.py::context_ext).

Usage (from the repo root):

    python a3/scripts/part4_passkey.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "part4")
FIGURES = os.path.join(HERE, "..", "figures")

# (csv, label, color, marker, linestyle). Categorical slots 1-3 of the validated default palette;
# marker shape and line style also encode the series, so color is never the only cue.
SERIES = [
    ("d12_ctx512_step002520.csv", "Ckpt 2: 512 → 2048", "#2a78d6", "o", "-"),
    ("d12_ctx512_step002016_widened.csv", "Ckpt 1, windows widened", "#eb6834", "s", "--"),
    ("d12_ctx512_step002016.csv", "Ckpt 1: 512, as trained", "#1baf7a", "^", "-"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        rows = list(csv.DictReader(f))
    return ([int(r["distance"]) for r in rows], [100 * float(r["accuracy"]) for r in rows],
            [100 * float(r["ci95"]) for r in rows], int(rows[0]["num_candidates"]))


plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, ax = plt.subplots(figsize=(6.5, 3.0))

chance = None
for name, label, color, marker, ls in SERIES:
    x, acc, ci, num_candidates = load(name)
    chance = 100 / num_candidates
    lo = [max(a - c, 0) for a, c in zip(acc, ci)]
    hi = [min(a + c, 100) for a, c in zip(acc, ci)]
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, acc, color=color, marker=marker, linestyle=ls, linewidth=1.5, markersize=5,
            markeredgecolor="white", markeredgewidth=0.8, label=label)
    ax.annotate(label, (x[-1], acc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK)

ax.axhline(chance, color=MUTED, linestyle=":", linewidth=1)
ax.text(70, chance + 2.5, f"chance ({chance:.0f}%)", color=MUTED, fontsize=8, ha="left")
ax.axvline(512, color=MUTED, linestyle=":", linewidth=1)
ax.text(530, 35, "phase-1 length /\nckpt 1 attention window", color=MUTED, fontsize=8, va="center")

ax.set_xlim(0, 2048)
ax.set_ylim(0, 105)
ax.set_xticks([64, 256, 512, 768, 1024, 1536, 1920])
ax.set_xlabel("Distance from passkey to question (tokens)")
ax.set_ylabel("Passkey accuracy (%)")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout()
fig.subplots_adjust(right=0.78, top=0.88)  # room for the direct labels and the legend
out = os.path.join(FIGURES, "part4_passkey.pdf")
fig.savefig(out)
print(f"wrote {out}")
