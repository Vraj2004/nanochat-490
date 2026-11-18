"""Compare the A4 Part 5 on-policy distillation runs with each other and with the RL runs of Parts 3-4.

Every run sees the same GSM8K train problems at the same steps, so rollouts can be compared problem by
problem. The main run goes the full 467 steps; ablations stop at step 240 on the same LR schedule and are
compared with the main run and the RL runs at step 240. The first distillation runs did not train the stop
token (see chat_rl.py --train-end-token); they are kept as an ablation. Runs or evals that are missing are
skipped. Writes to a4/results/part5:

    sft_repro.csv         chat_eval of the re-run SFT models (student and teachers) against Part 2
    summary.csv           every run: GSM8K test (greedy), pass@k at steps 240 and 420, training rollouts
    paired_vs_rl1.csv     per-problem difference in training solve rate vs RL #1, with a 95% CI
    categories.csv        answer categories (part3_rollouts.py) per run, steps 120-240 and >= 350
    style.csv             calculator use, "#### N" line, verbose opener, answer length per run and 20-step bin
    forgetting.csv        all chat_eval tasks: starting SFT model, main run, teacher
    curves.csv            pass@1 / pass@8 per eval, training correctness and reverse KL per 20-step bin

Needs the nanochat tokenizer and GSM8K, like part4_compare.py. Run from the repo root:

    python a4/scripts/part5_compare.py
"""
import csv
import glob
import gzip
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part3_rollouts import CATEGORIES, RESULTS as PART3, parse_rl_log
from part4_compare import RESULTS as PART4, analyze, stage_mean
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8KShaped

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "a4", "results", "part5")
LOGS = os.path.join(RESULTS, "logs")
PART2 = os.path.join(ROOT, "a4", "results", "part2")
MID, LATE, BIN = 240, 350, 20
TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]

# name, group, rollout glob, training log, {checkpoint step: chat_eval log}
RUNS = [
    ("RL #1 correctness", "RL (Parts 3-4)", os.path.join(PART3, "rollouts", "rank*.jsonl.gz"), os.path.join(PART3, "d20_rl.log"),
     {466: os.path.join(PART4, "logs", "d20_rl_chateval.log")}),
    ("RL #2 finish + partial", "RL (Parts 3-4)", os.path.join(PART4, "rollouts", "d20_rl_finish_partial", "rank*.jsonl.gz"),
     os.path.join(PART4, "logs", "d20_rl_finish_partial.log"), {466: os.path.join(PART4, "logs", "d20_rl_finish_partial_chateval.log")}),
]
for name, group, tag in [
    ("OPD, OMI-2 teacher (main)", "distillation", "p5_opd_v2"),
    ("OPD + reward (lambda = 0.5)", "distillation", "p5_opd_hybrid_v2"),
    ("OPD, MetaMathQA teacher", "distillation", "p5_opd_metamath"),
    ("SDPO self-teacher", "distillation", "p5_sdpo_v2"),
    ("OPD, OMI-2 teacher", "no stop token", "p5_opd"),
    ("OPD + reward (lambda = 0.5)", "no stop token", "p5_opd_hybrid"),
    ("SDPO self-teacher", "no stop token", "p5_sdpo"),
]:
    RUNS.append((f"{name} [{group}]" if group != "distillation" else name, group,
                 os.path.join(RESULTS, "rollouts", tag, "rank*.jsonl.gz"), os.path.join(LOGS, f"{tag}.log"),
                 {s: os.path.join(LOGS, f"{tag}_s{s}_eval.log") for s in (MID, 466)}))
MAIN = "OPD, OMI-2 teacher (main)"
VERBOSE_OPENERS = ("Let's break", "To ")


def write_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)


def eval_results(path):
    """ {task: (correct, total)} plus ChatCORE from a chat_eval log, or None if the log is missing. """
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read().replace("\r", "\n")
    out = {task: (int(c), int(t)) for c, t, task in re.findall(r"Final: (\d+)/(\d+) \([\d.]+%\)\n(?:.*\n)*?([\w-]+) accuracy:", text)}
    if m := re.findall(r"ChatCORE metric: (-?[\d.]+)", text):
        out["ChatCORE"] = float(m[-1])
    return out


def gsm8k(path):
    r = eval_results(path)
    return round(100 * r["GSM8K"][0] / r["GSM8K"][1], 2) if r and "GSM8K" in r else ""


def reverse_kl(path):
    text = open(path, errors="replace").read()
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"Step (\d+)/\d+ \| Average reward.*Reverse KL: (-?[\d.]+)", text)]


def passk_at(passk, step):
    row = next((r for r in passk if r[0] == step), None)
    return (round(100 * row[1], 2), round(100 * row[8], 2)) if row else ("", "")


def paired(a, b, lo, hi):
    """ Mean per-problem difference in solve rate (a - b) over problems seen at steps [lo, hi), with 95% CI. """
    diffs = [(a[i]["cats"]["correct"] - b[i]["cats"]["correct"]) / 16 for i in b if lo <= b[i]["step"] < hi and i in a]
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return len(diffs), round(100 * statistics.mean(diffs), 2), round(100 * 1.96 * se, 2)


def style_rows(name, pattern):
    """ Answer style per 20-step bin, straight from the completions. """
    bins = {}
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                bins.setdefault(r["step"] // BIN * BIN, []).extend(r["completions"])
    rows = []
    for b, comps in sorted(bins.items()):
        n = len(comps)
        rows.append([name, b, round(100 * sum("<|python_start|>" in c for c in comps) / n, 1),
                     round(100 * sum("####" in c for c in comps) / n, 1),
                     round(100 * sum(c.startswith(VERBOSE_OPENERS) for c in comps) / n, 1),
                     statistics.median(len(c) for c in comps)])
    return rows


def main():
    tokenizer, task = get_tokenizer(), GSM8KShaped(subset="main", split="train")
    keys = [k for k, _ in CATEGORIES]
    runs = [r for r in RUNS if glob.glob(r[2]) and os.path.exists(r[3])]
    print("runs found:", ", ".join(r[0] for r in runs))

    # SFT reproduction: our re-run checkpoints against Part 2
    part2 = {(r["model"], r["task"]): float(r["accuracy_pct"]) for r in csv.DictReader(open(os.path.join(PART2, "chat_eval.csv")))}
    write_csv("sft_repro.csv", ["model", "role", "gsm8k_part2", "gsm8k_rerun"], [
        ["SFT, nanochat mixture", "student", part2[("SFT (nanochat mix)", "GSM8K")], gsm8k(os.path.join(LOGS, "p5_sft_mix_eval.log"))],
        ["SFT + OpenMathInstruct-2", "teacher (main)", part2[("SFT + OpenMathInstruct-2", "GSM8K")], gsm8k(os.path.join(LOGS, "p5_sft_omi_eval.log"))],
        ["SFT + MetaMathQA", "teacher (ablation)", part2[("SFT + MetaMathQA", "GSM8K")], gsm8k(os.path.join(LOGS, "p5_sft_metamath_eval.log"))],
    ])

    data, passk, rkl = {}, {}, {}
    for name, _, pattern, log, _ in runs:
        print(f"analyzing {name} ...")
        data[name], passk[name], rkl[name] = analyze(pattern, task, tokenizer), parse_rl_log(log)[1], reverse_kl(log)

    # One row per run
    rows = []
    for name, group, _, _, evals in runs:
        last = max(p["step"] for p in data[name].values())
        late_lo, late_hi = (LATE, 10**9) if last >= LATE else (120, MID + 1)
        rows.append([name, group, last + 1, gsm8k(evals.get(MID, "")), gsm8k(evals.get(466, "")),
                     *passk_at(passk[name], MID), *passk_at(passk[name], 420),
                     round(100 * stage_mean(data[name], "correct", 120, MID + 1), 1),
                     round(100 * stage_mean(data[name], "no_answer", late_lo, late_hi), 1),
                     round(statistics.mean(v for s, v in rkl[name] if 120 <= s <= MID), 3) if rkl[name] else ""])
    write_csv("summary.csv", ["run", "group", "steps", "gsm8k_test_step240", "gsm8k_test_step466", "pass1_step240", "pass8_step240",
                              "pass1_step420", "pass8_step420", "train_correct_120_240", "no_answer_late", "reverse_kl_120_240"], rows)

    # Paired comparison with RL #1 on the same problems
    base, rows = data["RL #1 correctness"], []
    for name in [r[0] for r in runs[1:]]:
        rows.append([name, "steps 0-240", *paired(data[name], base, 0, MID + 1)])
        if max(p["step"] for p in data[name].values()) >= LATE:
            rows.append([name, f"steps >= {LATE}", *paired(data[name], base, LATE, 10**9)])
    write_csv("paired_vs_rl1.csv", ["run", "problems", "n", "diff_pct_points", "ci95_pct_points"], rows)

    # Answer categories
    rows = []
    for name in [r[0] for r in runs]:
        rows.append([name, "steps 120-240"] + [round(stage_mean(data[name], k, 120, MID + 1), 4) for k in keys])
        if max(p["step"] for p in data[name].values()) >= LATE:
            rows.append([name, f"steps >= {LATE}"] + [round(stage_mean(data[name], k, LATE, 10**9), 4) for k in keys])
    write_csv("categories.csv", ["run", "stage"] + keys, rows)

    write_csv("style.csv", ["run", "step_bin", "calculator_pct", "answer_line_pct", "verbose_opener_pct", "median_chars"],
              [row for name, _, pattern, _, _ in runs for row in style_rows(name, pattern)])

    # Forgetting: every chat_eval task for the starting model, the main run and the teacher
    rows = []
    main_evals = next((r[4] for r in runs if r[0] == MAIN), {})
    for label, path in [("Student (SFT, nanochat mixture)", os.path.join(LOGS, "p5_sft_mix_eval.log")),
                        ("OPD main, step 466", os.path.join(LOGS, "p5_opd_v2_s466_full_eval.log"))]:
        r = eval_results(path)
        if r and all(t in r for t in TASKS):
            rows.append([label] + [round(100 * r[t][0] / r[t][1], 2) for t in TASKS] + [r.get("ChatCORE", "")])
    rows.append(["Teacher (SFT + OpenMathInstruct-2, Part 2)"] + [part2[("SFT + OpenMathInstruct-2", t)] for t in TASKS] +
                [part2[("SFT + OpenMathInstruct-2", "ChatCORE")]])
    write_csv("forgetting.csv", ["model"] + TASKS + ["ChatCORE"], rows)

    # Curves for the plots
    rows = []
    for name in [r[0] for r in runs]:
        for r in passk[name]:
            rows.append([name, "passk", r[0], round(100 * r[1], 2), round(100 * r[8], 2)])
        last = max(p["step"] for p in data[name].values())
        for b in range(0, last + 1, BIN):
            vals = [v for s, v in rkl[name] if b <= s < b + BIN]
            rows.append([name, "train", b, round(100 * stage_mean(data[name], "correct", b, b + BIN), 2),
                         round(statistics.mean(vals), 4) if vals else ""])
    write_csv("curves.csv", ["run", "kind", "step", "value1", "value2"], rows)

    for name in ("sft_repro.csv", "summary.csv", "paired_vs_rl1.csv", "forgetting.csv"):
        print(f"\n== {name}\n" + open(os.path.join(RESULTS, name)).read())


if __name__ == "__main__":
    main()
