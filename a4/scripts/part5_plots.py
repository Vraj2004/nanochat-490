"""Plot the A4 Part 5 figures from the CSVs in a4/results/part5 (written by part5_compare.py).

    part5_curves.pdf    pass@1, pass@8, training correctness and reverse KL: distillation runs vs RL #1 and #2
    part5_mistakes.pdf  answer categories late in training (main run vs RL) and at steps 120-240 (all runs)

Usage (from the repo root):

    uv run --no-project --python 3.12 --with matplotlib python a4/scripts/part5_plots.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "part5")
FIGURES = os.path.join(HERE, "..", "figures")
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

# (run, legend label, color, linestyle); RL baselines are muted, the main distillation run is the strongest line
RUNS = [
    ("RL #1 correctness", "RL #1 correctness only", MUTED, "--"),
    ("RL #2 finish + partial", "RL #2 shaped reward", "#b5b3ad", "--"),
    ("OPD (OMI-2 teacher)", "OPD, OMI-2 teacher (main)", "#2a78d6", "-"),
    ("OPD, weaker teacher (MetaMathQA)", "OPD, MetaMathQA teacher", "#8b5bd6", "-"),
    ("OPD + reward (lambda = 0.5)", "OPD + reward", "#1baf7a", "-"),
    ("SDPO self-teacher", "SDPO self-teacher", "#eb6834", "-"),
]

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    out = os.path.join(FIGURES, name)
    fig.savefig(out)
    print(f"wrote {out}")


# 1) Curves
curves = defaultdict(lambda: defaultdict(list))
for r in read("curves.csv"):
    curves[r["run"]][r["kind"]].append(r)
fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.6))
for run, label, color, ls in RUNS:
    pk, tr = curves[run]["passk"], curves[run]["train"]
    kw = dict(color=color, linestyle=ls, linewidth=1.5, label=label)
    axes[0].plot([int(r["step"]) for r in pk], [float(r["value1"]) for r in pk], marker="o", markersize=3, **kw)
    axes[1].plot([int(r["step"]) for r in pk], [float(r["value2"]) for r in pk], marker="o", markersize=3, **kw)
    axes[2].plot([int(r["step"]) for r in tr], [float(r["value1"]) for r in tr], **kw)
    kl = [r for r in tr if r["value2"]]
    if kl:
        axes[3].plot([int(r["step"]) for r in kl], [float(r["value2"]) for r in kl], **kw)
for ax, title, xlabel in ((axes[0], "GSM8K pass@1 (%)", "step"), (axes[1], "GSM8K pass@8 (%)", "step"),
                          (axes[2], "Correct training samples (%)", "step (bins of 20)"),
                          (axes[3], "Reverse KL to teacher", "step (bins of 20)")):
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 467)
    ax.axvline(240, color=GRID, linewidth=1, zorder=0)
    style(ax)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout()
fig.subplots_adjust(top=0.72)
save(fig, "part5_curves.pdf")

# 2) Answer categories (same groups and colors as part4_mistakes.pdf)
GROUPS = [
    ("Correct", ["correct"], "#1baf7a"),
    ("Computed the wrong quantity", ["setup"], "#2a78d6"),
    ("Answered an intermediate quantity", ["intermediate"], "#8b5bd6"),
    ("No answer: ended turn mid-solution", ["ended_early"], "#eb6834"),
    ("Other (copied a question number, unsupported\nfinal number, ran out of tokens, format)",
     ["copied", "unsupported", "ran_out", "wrong_format"], "#b5b3ad"),
]
cats = read("categories.csv")
rows = [("late (>= 350)", r) for r in cats if r["stage"] == "late (>= 350)"] + \
       [("steps 120-240", r) for r in cats if r["stage"] == "steps 120-240"]
label_of = {run: label for run, label, *_ in RUNS}
labels = [f"{label_of[r['run']]}  [{stage}]" for stage, r in rows][::-1]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
left = [0.0] * len(rows)
for name, cols, color in GROUPS:
    share = [100 * sum(float(r[c]) for c in cols) for _, r in rows][::-1]
    bars = ax.barh(labels, share, left=left, color=color, edgecolor="white", linewidth=1, label=name)
    for bar, s, l in zip(bars, share, left):
        if s >= 6:
            ax.text(l + s / 2, bar.get_y() + bar.get_height() / 2, f"{s:.0f}", ha="center", va="center", fontsize=7, color="white")
    left = [l + s for l, s in zip(left, share)]
ax.set_xlim(0, 100)
ax.set_xlabel("Share of sampled training answers (%)")
ax.tick_params(axis="y", labelsize=7.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=3, frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "part5_mistakes.pdf")
