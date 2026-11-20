"""Plot the A4 Part 3 figures from the CSVs in a4/results/part3 (written by part3_rollouts.py).

    part3_curves.pdf      reward, pass@1 and pass@8 during RL: our run vs Karpathy's original run
    part3_categories.pdf  share of each answer category over training
    part3_difficulty.pdf  per-problem solve rate by solution length and by topic, early vs late

Usage (from the repo root):

    uv run --no-project --python 3.12 --with matplotlib python a4/scripts/part3_plots.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "part3")
FIGURES = os.path.join(HERE, "..", "figures")

OURS, KARPATHY = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"
EARLY, LATE = 100, 350  # "early" = steps < 100, "late" = steps >= 350

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    out = os.path.join(FIGURES, name)
    fig.savefig(out)
    print(f"wrote {out}")


def ema(values, beta=0.9):
    out, s = [], 0.0
    for i, v in enumerate(values):
        s = beta * s + (1 - beta) * v
        out.append(s / (1 - beta ** (i + 1)))  # debiased like the W&B smoothing
    return out


# 1) Training curves vs the original run
reward = read("rl_reward.csv")
passk = read("rl_passk.csv")
kp = read("karpathy_rl_digitized.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
steps, r = [int(x["step"]) for x in reward], [float(x["reward"]) for x in reward]
axes[0].plot(steps, r, color=OURS, alpha=0.2, linewidth=0.8)
axes[0].plot(steps, ema(r), color=OURS, linewidth=1.5, label="Ours (d20_final_v2)")
kx = [(int(x["step"]), float(x["reward_smoothed"])) for x in kp if x["reward_smoothed"]]
axes[0].plot(*zip(*kx), color=KARPATHY, linestyle="--", linewidth=1.5, label="Karpathy (Oct 2025)")
axes[0].set_title("Training reward", fontsize=9)
for ax, col, title in ((axes[1], "pass1", "pass@1"), (axes[2], "pass8", "pass@8")):
    ax.plot([int(x["step"]) for x in passk], [100 * float(x[col]) for x in passk], color=OURS, marker="o",
            markersize=4, markeredgecolor="white", markeredgewidth=0.8, linewidth=1.5)
    pts = [(int(x["step"]), 100 * float(x[col])) for x in kp if x[col]]
    ax.plot(*zip(*pts), color=KARPATHY, marker="s", markersize=4, markeredgecolor="white",
            markeredgewidth=0.8, linestyle="--", linewidth=1.5)
    ax.set_title(f"GSM8K test {title} (%)", fontsize=9)
for ax in axes:
    ax.set_xlabel("RL step")
    ax.set_xlim(0, 467)
    style(ax)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.subplots_adjust(top=0.80)
save(fig, "part3_curves.pdf")

# 2) Answer categories over training (stacked share per 25-step bin)
GROUPS = [  # (label, columns, color); "other" is gray so it reads as the remainder
    ("Correct", ["correct"], "#1baf7a"),
    ("Computed the wrong quantity", ["setup"], OURS),
    ("Answered an intermediate quantity", ["intermediate"], "#8b5bd6"),
    ("No answer: ended turn mid-solution", ["ended_early"], KARPATHY),
    ("Other (copied a question number, unsupported\nfinal number, ran out of tokens, format)",
     ["copied", "unsupported", "ran_out", "wrong_format"], "#b5b3ad"),
]
cats = read("answer_categories.csv")
x = [int(c["step_bin"]) for c in cats]
totals = [sum(int(v) for k, v in c.items() if k != "step_bin") for c in cats]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
bottom = [0.0] * len(cats)
for label, cols, color in GROUPS:
    share = [100 * sum(int(c[k]) for k in cols) / t for c, t in zip(cats, totals)]
    ax.bar(x, share, width=23, bottom=bottom, color=color, edgecolor="white", linewidth=1, align="edge", label=label)
    bottom = [b + s for b, s in zip(bottom, share)]
ax.set_xlim(0, 475)
ax.set_ylim(0, 100)
ax.set_xlabel("RL step (bins of 25 steps, 16 problems x 16 samples per step)")
ax.set_ylabel("Share of sampled answers (%)")
style(ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7.5)
fig.tight_layout()
save(fig, "part3_categories.pdf")

# 3) Solve rate by difficulty and by topic, early vs late in training
problems = read("problems.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), gridspec_kw={"width_ratios": [7, 5]})
for ax, key, order, xlabel in (
    (axes[0], lambda p: min(int(p["gold_steps"]), 6), list(range(7)), "Calculator steps in the reference solution"),
    (axes[1], lambda p: p["topic"], ["counting & other", "money", "time & rates", "fractions & multiples", "percent"], ""),
):
    groups = {"early": defaultdict(list), "late": defaultdict(list)}
    for p in problems:
        step = int(p["step"])
        stage = "early" if step < EARLY else ("late" if step >= LATE else None)
        if stage:
            groups[stage][key(p)].append(int(p["num_correct"]) / 16)
    for i, (stage, color, label) in enumerate((("early", "#9cc3ee", f"Steps < {EARLY}"), ("late", OURS, f"Steps >= {LATE}"))):
        means = [100 * sum(groups[stage][g]) / len(groups[stage][g]) for g in order]
        ax.bar([j + (i - 0.5) * 0.4 for j in range(len(order))], means, width=0.38, color=color, label=label)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}+" if g == 6 else str(g) for g in order] if xlabel else
                       [g.replace(" & ", " &\n") for g in order], fontsize=7.5 if not xlabel else 9)
    ax.set_xlabel(xlabel)
    style(ax)
axes[0].set_ylabel("Mean solve rate (% of 16 samples)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.subplots_adjust(top=0.86)
save(fig, "part3_difficulty.pdf")
