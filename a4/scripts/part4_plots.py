"""Plot the A4 Part 4 figures from the CSVs in a4/results/part4 (written by part4_compare.py).

    part4_curves.pdf    pass@1 and pass@8 during RL, and training correctness, for the four runs
    part4_behavior.pdf  the behaviors the shaped rewards target: answers with no "#### N", answers that
                        finish on their own calculator result, matched reference intermediates
    part4_mistakes.pdf  share of each answer category late in training, per run

Usage (from the repo root):

    uv run --no-project --python 3.12 --with matplotlib python a4/scripts/part4_plots.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "part4")
FIGURES = os.path.join(HERE, "..", "figures")
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

# (run, legend label, color, marker, linestyle); the original run is the muted dashed baseline
RUNS = [
    ("#1 original", "#1 correctness only", MUTED, "o", "--"),
    ("#3 finish", "#3 + finish bonus", "#eb6834", "s", "-"),
    ("#4 partial", "#4 + partial credit", "#8b5bd6", "^", "-"),
    ("#2 finish + partial", "#2 + both", "#2a78d6", "D", "-"),
]

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def by_run(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["run"]].append(r)
    return out


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    out = os.path.join(FIGURES, name)
    fig.savefig(out)
    print(f"wrote {out}")


def plot_runs(ax, data, x, y, scale=1.0, markers=True):
    for run, label, color, marker, ls in RUNS:
        rows = data[run]
        ax.plot([float(r[x]) for r in rows], [scale * float(r[y]) for r in rows], color=color, linestyle=ls,
                marker=marker if markers else None, markersize=4, markeredgecolor="white", markeredgewidth=0.8,
                linewidth=1.5, label=label)


# 1) Eval and training curves
passk, behavior = by_run(read("passk_by_run.csv")), by_run(read("behavior_by_step.csv"))
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
plot_runs(axes[0], passk, "step", "pass1", 100)
plot_runs(axes[1], passk, "step", "pass8", 100)
plot_runs(axes[2], behavior, "step_bin", "correct", 100, markers=False)
for ax, title, xlabel in ((axes[0], "GSM8K test pass@1 (%)", "RL step"), (axes[1], "GSM8K test pass@8 (%)", "RL step"),
                          (axes[2], "Correct training samples (%)", "RL step (bins of 25)")):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 467)
    style(ax)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout()
fig.subplots_adjust(top=0.78)
save(fig, "part4_curves.pdf")

# 2) The targeted behaviors over training
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
plot_runs(axes[0], behavior, "step_bin", "no_answer", 100, markers=False)
plot_runs(axes[1], behavior, "step_bin", "finish", 100, markers=False)
plot_runs(axes[2], behavior, "step_bin", "matched", markers=False)
for ax, title in ((axes[0], "No \"#### N\" answer (%)"), (axes[1], "Ends on its own calculator result (%)"),
                  (axes[2], "Matched reference intermediates")):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("RL step (bins of 25)")
    ax.set_xlim(0, 467)
    style(ax)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout()
fig.subplots_adjust(top=0.78)
save(fig, "part4_behavior.pdf")

# 3) Answer categories late in training (same groups and colors as part3_categories.pdf)
GROUPS = [
    ("Correct", ["correct"], "#1baf7a"),
    ("Computed the wrong quantity", ["setup"], "#2a78d6"),
    ("Answered an intermediate quantity", ["intermediate"], "#8b5bd6"),
    ("No answer: ended turn mid-solution", ["ended_early"], "#eb6834"),
    ("Other (copied a question number, unsupported\nfinal number, ran out of tokens, format)",
     ["copied", "unsupported", "ran_out", "wrong_format"], "#b5b3ad"),
]
late = {r["run"]: r for r in read("categories_by_run.csv") if r["stage"] == "late"}
fig, ax = plt.subplots(figsize=(7.5, 2.4))
labels = [label for _, label, *_ in RUNS][::-1]
left = [0.0] * len(RUNS)
for name, cols, color in GROUPS:
    share = [100 * sum(float(late[run][c]) for c in cols) for run, *_ in RUNS][::-1]
    bars = ax.barh(labels, share, left=left, color=color, edgecolor="white", linewidth=1, label=name)
    for bar, s, l in zip(bars, share, left):
        if s >= 6:
            ax.text(l + s / 2, bar.get_y() + bar.get_height() / 2, f"{s:.0f}", ha="center", va="center",
                    fontsize=7.5, color="white")
    left = [l + s for l, s in zip(left, share)]
ax.set_xlim(0, 100)
ax.set_xlabel(f"Share of sampled answers late in training (steps >= 350, %)")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7.5)
fig.tight_layout()
save(fig, "part4_mistakes.pdf")
