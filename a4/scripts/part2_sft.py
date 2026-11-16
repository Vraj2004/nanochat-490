"""Summarize the A4 Part 2 SFT runs and plot their chat_eval results.

Reads the chat_eval logs (exact correct/total counts per task) and the chat_sft training logs in
a4/results/part2/logs (copied from the Modal volume), and writes:

    a4/results/part2/chat_eval.csv   accuracy, correct/total and 95% CI per model and task
    a4/results/part2/sft_runs.csv    rows, steps, training time, final val bpb per SFT run
    a4/figures/part2_chat_eval.pdf   accuracy per task for the base model and the three SFT models

Usage (from the repo root):

    uv run --no-project --python 3.12 --with matplotlib python a4/scripts/part2_sft.py
"""
import csv
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "part2")
LOGS = os.path.join(RESULTS, "logs")
FIGURES = os.path.join(HERE, "..", "figures")
TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]
# (label, eval log, SFT training log or None, color); categorical slots in a fixed order
MODELS = [
    ("Base (pretrained)", "d20_base_chateval.log", None, "#b5b3ad"),
    ("SFT (nanochat mix)", "d20_sft_chateval.log", "d20_sft.log", "#2a78d6"),
    ("SFT + MetaMathQA", "d20_sft_metamathqa_chateval.log", "d20_sft_metamathqa.log", "#eb6834"),
    ("SFT + OpenMathInstruct-2", "d20_sft_openmathinstruct_chateval.log", "d20_sft_openmathinstruct.log", "#1baf7a"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def read(name):
    with open(os.path.join(LOGS, name), errors="replace") as f:
        return f.read().replace("\r", "\n")


def eval_counts(text):
    """chat_eval prints 'Final: correct/total (acc%)' then '<task> accuracy: ...' for each task."""
    counts = re.findall(r"Final: (\d+)/(\d+) \([\d.]+%\)\n(?:.*\n)*?([\w-]+) accuracy:", text)
    chatcore = float(re.findall(r"ChatCORE metric: (-?[\d.]+)", text)[-1])
    return {task: (int(c), int(n)) for c, n, task in counts}, chatcore


rows, runs = [], []
results = {}
for label, eval_log, sft_log, _ in MODELS:
    counts, chatcore = eval_counts(read(eval_log))
    results[label] = counts
    for task in TASKS:
        c, n = counts[task]
        p = c / n
        rows.append([label, task, c, n, round(100 * p, 2), round(100 * 1.96 * math.sqrt(p * (1 - p) / n), 2)])
    rows.append([label, "ChatCORE", "", "", round(chatcore, 4), ""])
    if sft_log:
        text = read(sft_log)
        runs.append([label, re.search(r"Training mixture: ([\d,]+) rows", text).group(1).replace(",", ""),
                     int(re.findall(r"Step (\d+) \| Validation bpb", text)[-1]),
                     float(re.search(r"Total training time: ([\d.]+)m", text).group(1)),
                     float(re.findall(r"Validation bpb: ([\d.]+)", text)[-1])])

with open(os.path.join(RESULTS, "chat_eval.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["model", "task", "correct", "total", "accuracy_pct", "ci95_pct"]] + rows)
with open(os.path.join(RESULTS, "sft_runs.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["model", "train_rows", "steps", "train_minutes", "final_val_bpb"]] + runs)
for r in rows:
    print(r)
for r in runs:
    print(r)

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, ax = plt.subplots(figsize=(7.5, 2.8))
width = 0.2
for i, (label, _, _, color) in enumerate(MODELS):
    xs = [j + (i - 1.5) * width for j in range(len(TASKS))]
    acc = [100 * results[label][t][0] / results[label][t][1] for t in TASKS]
    ci = [100 * 1.96 * math.sqrt((a / 100) * (1 - a / 100) / results[label][t][1]) for a, t in zip(acc, TASKS)]
    ax.bar(xs, acc, width=width * 0.92, color=color, yerr=ci, error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 0},
           label=label)
ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)
ax.text(len(TASKS) - 0.55, 26.5, "chance on 4-way\nmultiple choice", color=MUTED, fontsize=7.5, ha="right", va="bottom")
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels(TASKS)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 70)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout()
fig.subplots_adjust(top=0.86)
out = os.path.join(FIGURES, "part2_chat_eval.pdf")
fig.savefig(out)
print(f"wrote {out}")
