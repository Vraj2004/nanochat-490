"""Plot the A4 Part 5 figures from the CSVs in a4/results/part5 (written by part5_compare.py).

    part5_curves.pdf    pass@1, pass@8, training correctness and reverse KL for the distillation and RL runs
    part5_style.pdf     how answers change: calculator use, "#### N" line, verbose opener, answer length
    part5_mistakes.pdf  answer categories at steps 120-240 for every run

Runs missing from the CSVs are skipped. Usage (from the repo root):

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

# (run, legend label, color, linestyle)
RUNS = [
    ("RL #1 correctness", "RL #1 correctness only", MUTED, "--"),
    ("RL #2 finish + partial", "RL #2 shaped reward", "#b5b3ad", "--"),
    ("OPD, OMI-2 teacher (main)", "OPD, OMI-2 teacher (main)", "#2a78d6", "-"),
    ("OPD + reward (lambda = 0.5)", "OPD + reward", "#1baf7a", "-"),
    ("OPD, MetaMathQA teacher", "OPD, MetaMathQA teacher", "#8b5bd6", "-"),
    ("SDPO self-teacher", "SDPO self-teacher", "#eb6834", "-"),
    ("OPD, OMI-2 teacher [no stop token]", "OPD, no stop token", "#2a78d6", ":"),
    ("SDPO self-teacher [no stop token]", "SDPO, no stop token", "#eb6834", ":"),
]

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def style(ax, title, xlabel):
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_xlim(0, 467)
    ax.axvline(240, color=GRID, linewidth=1, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    out = os.path.join(FIGURES, name)
    fig.savefig(out)
    print(f"wrote {out}")


def plot_panels(rows_by_run, panels, name, runs):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5, 2.6))
    for run, label, color, ls in runs:
        rows = rows_by_run.get(run)
        if not rows:
            continue
        for ax, (col, *_ ) in zip(axes, panels):
            pts = [(int(r["step"] if "step" in r else r["step_bin"]), float(r[col])) for r in rows if r[col] != ""]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=ls, linewidth=1.4, label=label)
    for ax, (_, title, xlabel) in zip(axes, panels):
        style(ax, title, xlabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(top=0.74)
    save(fig, name)


# 1) Curves: evals every 60 steps and training correctness / reverse KL per 20-step bin
curves = defaultdict(lambda: defaultdict(list))
for r in read("curves.csv"):
    curves[r["kind"]][r["run"]].append(r)
fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.6))
for run, label, color, ls in RUNS:
    kw = dict(color=color, linestyle=ls, linewidth=1.4, label=label)
    pk, tr = curves["passk"].get(run, []), curves["train"].get(run, [])
    if not pk:
        continue
    axes[0].plot([int(r["step"]) for r in pk], [float(r["value1"]) for r in pk], marker="o", markersize=2.5, **kw)
    axes[1].plot([int(r["step"]) for r in pk], [float(r["value2"]) for r in pk], marker="o", markersize=2.5, **kw)
    axes[2].plot([int(r["step"]) for r in tr], [float(r["value1"]) for r in tr], **kw)
    kl = [r for r in tr if r["value2"]]
    if kl:
        axes[3].plot([int(r["step"]) for r in kl], [float(r["value2"]) for r in kl], **kw)
for ax, title, xlabel in ((axes[0], "GSM8K pass@1 (%)", "step"), (axes[1], "GSM8K pass@8 (%)", "step"),
                          (axes[2], "Correct training samples (%)", "step (bins of 20)"),
                          (axes[3], "Reverse KL (nats/token)", "step (bins of 20)")):
    style(ax, title, xlabel)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout()
fig.subplots_adjust(top=0.74)
save(fig, "part5_curves.pdf")

# 2) Answer style over training
by_run = defaultdict(list)
for r in read("style.csv"):
    by_run[r["run"]].append(r)
style_runs = [r for r in RUNS if r[0] in ("RL #1 correctness", "OPD, OMI-2 teacher (main)", "OPD + reward (lambda = 0.5)", "OPD, MetaMathQA teacher",
                                          "SDPO self-teacher", "OPD, OMI-2 teacher [no stop token]", "SDPO self-teacher [no stop token]")]
plot_panels(by_run, [("calculator_pct", "Calculator use (%)", "step (bins of 20)"),
                     ("answer_line_pct", "Has a #### N line (%)", "step (bins of 20)"),
                     ("verbose_opener_pct", "Verbose opener (%)", "step (bins of 20)"),
                     ("median_chars", "Median length (chars)", "step (bins of 20)")],
            "part5_style.pdf", style_runs)

# 3) Answer categories at steps 120-240 (same groups and colors as part4_mistakes.pdf)
GROUPS = [
    ("Correct", ["correct"], "#1baf7a"),
    ("Computed the wrong quantity", ["setup"], "#2a78d6"),
    ("Answered an intermediate quantity", ["intermediate"], "#8b5bd6"),
    ("No answer: ended turn mid-solution", ["ended_early"], "#eb6834"),
    ("Other (copied a question number, unsupported\nfinal number, ran out of tokens, format)",
     ["copied", "unsupported", "ran_out", "wrong_format"], "#b5b3ad"),
]
cats = {r["run"]: r for r in read("categories.csv") if r["stage"] == "steps 120-240"}
present = [(run, label) for run, label, *_ in RUNS if run in cats]
fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(present) + 1.3))
labels = [label for _, label in present][::-1]
left = [0.0] * len(present)
for name, cols, color in GROUPS:
    share = [100 * sum(float(cats[run][c]) for c in cols) for run, _ in present][::-1]
    bars = ax.barh(labels, share, left=left, color=color, edgecolor="white", linewidth=1, label=name)
    for bar, s, l in zip(bars, share, left):
        if s >= 6:
            ax.text(l + s / 2, bar.get_y() + bar.get_height() / 2, f"{s:.0f}", ha="center", va="center", fontsize=7, color="white")
    left = [l + s for l, s in zip(left, share)]
ax.set_xlim(0, 100)
ax.set_xlabel("Share of sampled training answers, steps 120-240 (%)")
ax.tick_params(axis="y", labelsize=7.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "part5_mistakes.pdf")
