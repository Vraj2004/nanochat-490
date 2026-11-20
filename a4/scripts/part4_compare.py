"""Compare the A4 Part 4 RL runs (shaped rewards) with the Part 3 run (correctness only).

All four runs see the same GSM8K train problems at the same steps (same data order), so their
rollouts can be compared problem by problem. For each run this writes to a4/results/part4:

    results_table.csv      final GSM8K test accuracy, pass@k at step 420, rollout behavior late in training
    paired_vs_original.csv per-problem difference in solve rate vs run #1, with a 95% CI
    categories_by_run.csv  share of each answer category (part3_rollouts.py categories), early vs late
    behavior_by_step.csv   correct / no answer / finish / matched intermediates / calculator calls per 25 steps
    passk_by_run.csv       pass@1 and pass@8 evals during training

Needs the nanochat tokenizer and GSM8K like part3_rollouts.py. Run from the repo root with the
project environment:

    python a4/scripts/part4_compare.py
"""
import collections
import csv
import glob
import gzip
import json
import math
import os
import re
import statistics

from part3_rollouts import CATEGORIES, RESULTS as PART3, categorize, numbers, parse_rl_log, same_number
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8KShaped, extract_answer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "a4", "results", "part4")
EARLY, LATE = 100, 350  # "early" = steps < 100, "late" = steps >= 350
STEP_BIN = 25

# (run, environment, rollout glob, RL log, eval log)
RUNS = [
    ("#1 original", "gsm8k", os.path.join(PART3, "rollouts", "rank*.jsonl.gz"),
     os.path.join(PART3, "d20_rl.log"), "d20_rl_chateval.log"),
    *[(name, env, os.path.join(RESULTS, "rollouts", tag, "rank*.jsonl.gz"),
       os.path.join(RESULTS, "logs", f"{tag}.log"), f"{tag}_chateval.log")
      for name, env, tag in (("#3 finish", "gsm8k_finish", "d20_rl_finish"),
                             ("#4 partial", "gsm8k_partial", "d20_rl_partial"),
                             ("#2 finish + partial", "gsm8k_finish_partial", "d20_rl_finish_partial"))],
]


def write_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_accuracy(log_name):
    with open(os.path.join(RESULTS, "logs", log_name), errors="replace") as f:
        return float(re.findall(r"GSM8K accuracy: ([\d.]+)%", f.read().replace("\r", "\n"))[-1])


def analyze(pattern, task, tokenizer):
    """Per-problem rows: step, idx, num correct, category counts, finish, matched, calculator calls."""
    rows = []
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt") as f:
            rows.extend(json.loads(line) for line in f)
    out = {}
    for r in rows:
        conv = task[r["idx"]]
        messages = conv["messages"]
        question, parts = messages[0]["content"], messages[-1]["content"]
        gold = extract_answer(parts[-1]["text"])
        intermediates = [p["text"].strip() for p in parts if p["type"] == "python_output"]
        intermediates = [x for x in intermediates if not same_number(x, gold)]
        cats, finish, matched, calls = collections.Counter(), 0, 0, 0
        for completion in r["completions"]:
            num_tokens = len(tokenizer.enc.encode(completion, allowed_special="all"))
            cats[categorize(completion, num_tokens, gold, intermediates, numbers(question))] += 1
            _, f, m = task.reward_terms(conv, completion)
            finish, matched, calls = finish + f, matched + m, calls + completion.count("<|output_start|>")
        assert cats["correct"] == sum(r.get("correct", r["rewards"])), "categories must agree with the logged correctness"
        out[r["idx"]] = {"step": r["step"], "cats": cats, "finish": finish, "matched": matched, "calls": calls}
    return out


def stage_mean(problems, key, lo, hi):
    sub = [p for p in problems.values() if lo <= p["step"] < hi]
    if key in ("finish", "matched", "calls"):
        return sum(p[key] for p in sub) / (16 * len(sub))
    if key == "no_answer":
        return sum(p["cats"]["ran_out"] + p["cats"]["ended_early"] for p in sub) / (16 * len(sub))
    if key == "any_correct":
        return sum(p["cats"]["correct"] > 0 for p in sub) / len(sub)
    return sum(p["cats"][key] for p in sub) / (16 * len(sub))


def main():
    tokenizer = get_tokenizer()
    task = GSM8KShaped(subset="main", split="train")
    keys = [k for k, _ in CATEGORIES]
    data, passk = {}, []
    for name, env, pattern, rl_log, _ in RUNS:
        print(f"analyzing {name} ...")
        data[name] = analyze(pattern, task, tokenizer)
        passk += [[name, row[0], row[1], row[8]] for row in parse_rl_log(rl_log)[1]]
    write_csv("passk_by_run.csv", ["run", "step", "pass1", "pass8"], passk)

    # Headline table
    table = []
    for name, env, _, _, eval_log in RUNS:
        p = data[name]
        at420 = next(row for row in passk if row[0] == name and row[1] == 420)
        table.append([name, env, test_accuracy(eval_log), 100 * at420[2], 100 * at420[3]] +
                     [round(100 * stage_mean(p, k, LATE, 10**9), 1) for k in ("correct", "any_correct", "no_answer", "finish")] +
                     [round(stage_mean(p, k, LATE, 10**9), 2) for k in ("matched", "calls")])
    header = ["run", "env", "gsm8k_test", "pass1_step420", "pass8_step420", "train_correct_late",
              "train_any_correct_late", "no_answer_late", "finish_late", "matched_late", "calc_calls_late"]
    write_csv("results_table.csv", header, table)

    # Paired comparison with run #1 on the same problems (difference in per-problem solve rate)
    paired, base = [], data[RUNS[0][0]]
    for name, *_ in RUNS[1:]:
        for label, lo, hi in (("all steps", 0, 10**9), (f"steps >= {LATE}", LATE, 10**9)):
            diffs = [(data[name][i]["cats"]["correct"] - base[i]["cats"]["correct"]) / 16
                     for i in base if lo <= base[i]["step"] < hi]
            mean, se = statistics.mean(diffs), statistics.stdev(diffs) / math.sqrt(len(diffs))
            paired.append([name, label, len(diffs), round(100 * mean, 2), round(100 * 1.96 * se, 2)])
    write_csv("paired_vs_original.csv", ["run", "problems", "n", "diff_pct_points", "ci95_pct_points"], paired)

    # Answer categories early vs late, and behavior over training
    cat_rows, beh_rows = [], []
    for name, *_ in RUNS:
        for stage, lo, hi in (("early", 0, EARLY), ("late", LATE, 10**9)):
            cat_rows.append([name, stage] + [round(stage_mean(data[name], k, lo, hi), 4) for k in keys])
        for b in range(0, 467, STEP_BIN):
            beh_rows.append([name, b] + [round(stage_mean(data[name], k, b, b + STEP_BIN), 4)
                                         for k in ("correct", "no_answer", "finish", "matched", "calls")])
    write_csv("categories_by_run.csv", ["run", "stage"] + keys, cat_rows)
    write_csv("behavior_by_step.csv", ["run", "step_bin", "correct", "no_answer", "finish", "matched", "calc_calls"], beh_rows)

    print("\n" + " | ".join(header))
    for row in table:
        print(" | ".join(str(x) for x in row))
    print("\npaired difference in solve rate vs #1 original (percentage points, 95% CI):")
    for row in paired:
        print(f"  {row[0]:22s} {row[1]:14s} n={row[2]:5d}  {row[3]:+.2f} +- {row[4]:.2f}")
    print("\nanswer categories late in training (%):")
    for row in cat_rows:
        if row[1] == "late":
            print(f"  {row[0]:22s} " + "  ".join(f"{k} {100*v:.1f}" for k, v in zip(keys, row[2:])))


if __name__ == "__main__":
    main()
