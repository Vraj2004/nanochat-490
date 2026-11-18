"""Compare the A4 Part 5 on-policy distillation runs with each other and with the RL runs of Parts 3-4.

Every run sees the same GSM8K train problems at the same steps, so rollouts can be compared problem by
problem. The main run goes the full 467 steps; the ablations stop at step 240 on the same LR schedule, so
they are compared with the main run and the RL runs at step 240. Writes to a4/results/part5:

    sft_repro.csv         GSM8K of the re-run SFT models (student and teachers) against Part 2
    results_table.csv     main run vs RL #1 and #2: GSM8K test, pass@k at step 420, rollouts late in training
    ablations_step240.csv every run at step 240: GSM8K test (distillation runs), pass@k, training rollouts
    paired_vs_rl1.csv     per-problem difference in training solve rate vs RL #1, with a 95% CI
    categories.csv        answer categories (part3_rollouts.py) per run and stage
    forgetting.csv        all chat_eval tasks: starting SFT model, main run, teacher
    curves.csv            pass@1, pass@8, training correctness and reverse KL per eval step / 20-step bin

Needs the nanochat tokenizer and GSM8K, like part4_compare.py. Run from the repo root:

    python a4/scripts/part5_compare.py
"""
import csv
import glob
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
ABLATION_STEP, LATE, BIN = 240, 350, 20
TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]

# RL baselines (Parts 3-4): name -> (rollout glob, training log)
RL = {
    "RL #1 correctness": (os.path.join(PART3, "rollouts", "rank*.jsonl.gz"), os.path.join(PART3, "d20_rl.log")),
    "RL #2 finish + partial": (os.path.join(PART4, "rollouts", "d20_rl_finish_partial", "rank*.jsonl.gz"),
                               os.path.join(PART4, "logs", "d20_rl_finish_partial.log")),
    "RL #3 finish": (os.path.join(PART4, "rollouts", "d20_rl_finish", "rank*.jsonl.gz"), os.path.join(PART4, "logs", "d20_rl_finish.log")),
    "RL #4 partial": (os.path.join(PART4, "rollouts", "d20_rl_partial", "rank*.jsonl.gz"), os.path.join(PART4, "logs", "d20_rl_partial.log")),
}
RL_TEST = {"RL #1 correctness": 13.65, "RL #2 finish + partial": 15.01}  # Part 4 results_table.csv
# Distillation runs: name -> run tag (rollouts in rollouts/<tag>/, training log logs/<tag>.log)
DISTILL = {
    "OPD (OMI-2 teacher)": "p5_opd",
    "OPD, weaker teacher (MetaMathQA)": "p5_opd_metamath",
    "OPD + reward (lambda = 0.5)": "p5_opd_hybrid",
    "SDPO self-teacher": "p5_sdpo",
}
MAIN = "OPD (OMI-2 teacher)"


def write_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)


def read_log(name):
    with open(os.path.join(LOGS, name), errors="replace") as f:
        return f.read().replace("\r", "\n")


def eval_results(name):
    """ {task: (correct, total)} plus ChatCORE from a chat_eval log ('Final: c/t' then '<task> accuracy'). """
    text = read_log(name)
    out = {task: (int(c), int(t)) for c, t, task in re.findall(r"Final: (\d+)/(\d+) \([\d.]+%\)\n(?:.*\n)*?([\w-]+) accuracy:", text)}
    if m := re.findall(r"ChatCORE metric: (-?[\d.]+)", text):
        out["ChatCORE"] = float(m[-1])
    return out


def gsm8k(name):
    c, t = eval_results(name)["GSM8K"]
    return round(100 * c / t, 2)


def reverse_kl(path):
    with open(path, errors="replace") as f:
        return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"Step (\d+)/\d+ \| Average reward.*Reverse KL: (-?[\d.]+)", f.read())]


def passk_at(passk, step):
    row = next(r for r in passk if r[0] == step)
    return round(100 * row[1], 2), round(100 * row[8], 2)


def paired(a, b, lo, hi):
    """ Mean per-problem difference in solve rate (a - b) over problems seen at steps [lo, hi), with 95% CI. """
    diffs = [(a[i]["cats"]["correct"] - b[i]["cats"]["correct"]) / 16 for i in b if lo <= b[i]["step"] < hi and i in a]
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return len(diffs), round(100 * statistics.mean(diffs), 2), round(100 * 1.96 * se, 2)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tokenizer, task = get_tokenizer(), GSM8KShaped(subset="main", split="train")
    keys = [k for k, _ in CATEGORIES]

    # SFT reproduction: our re-run checkpoints against Part 2
    part2 = {(r["model"], r["task"]): float(r["accuracy_pct"]) for r in csv.DictReader(open(os.path.join(PART2, "chat_eval.csv")))}
    write_csv("sft_repro.csv", ["model", "role", "gsm8k_part2", "gsm8k_rerun"], [
        ["SFT, nanochat mixture", "student", part2[("SFT (nanochat mix)", "GSM8K")], gsm8k("p5_sft_mix_eval.log")],
        ["SFT + OpenMathInstruct-2", "teacher (main)", part2[("SFT + OpenMathInstruct-2", "GSM8K")], gsm8k("p5_sft_omi_eval.log")],
        ["SFT + MetaMathQA", "teacher (ablation)", part2[("SFT + MetaMathQA", "GSM8K")], gsm8k("p5_sft_metamath_eval.log")],
    ])

    data, passk, rkl = {}, {}, {}
    for name, (pattern, log) in RL.items():
        print(f"analyzing {name} ...")
        data[name], passk[name], rkl[name] = analyze(pattern, task, tokenizer), parse_rl_log(log)[1], []
    for name, tag in DISTILL.items():
        print(f"analyzing {name} ...")
        log = os.path.join(LOGS, f"{tag}.log")
        data[name] = analyze(os.path.join(RESULTS, "rollouts", tag, "rank*.jsonl.gz"), task, tokenizer)
        passk[name], rkl[name] = parse_rl_log(log)[1], reverse_kl(log)

    # Main run against the RL runs over the full epoch
    rows = []
    for name, test in [(n, RL_TEST[n]) for n in RL_TEST] + [(MAIN, gsm8k("p5_opd_s466_eval.log"))]:
        rows.append([name, test, *passk_at(passk[name], 420)] +
                    [round(100 * stage_mean(data[name], k, LATE, 10**9), 1) for k in ("correct", "any_correct", "no_answer")])
    write_csv("results_table.csv", ["run", "gsm8k_test", "pass1_step420", "pass8_step420",
                                    "train_correct_late", "train_any_correct_late", "no_answer_late"], rows)

    # Every run at step 240 (the ablations stop there)
    test240 = {MAIN: gsm8k("p5_opd_s240_eval.log"), **{n: gsm8k(f"{t}_s240_eval.log") for n, t in DISTILL.items() if n != MAIN}}
    rows = []
    for name in list(RL) + list(DISTILL):
        lo = ABLATION_STEP - 120
        rows.append([name, test240.get(name, ""), *passk_at(passk[name], ABLATION_STEP)] +
                    [round(100 * stage_mean(data[name], k, lo, ABLATION_STEP + 1), 1) for k in ("correct", "any_correct", "no_answer")] +
                    [round(statistics.mean(v for s, v in rkl[name] if lo <= s <= ABLATION_STEP), 3) if rkl[name] else ""])
    write_csv("ablations_step240.csv", ["run", "gsm8k_test_step240", "pass1_step240", "pass8_step240", "train_correct_120_240",
                                        "train_any_correct_120_240", "no_answer_120_240", "reverse_kl_120_240"], rows)

    # Paired comparison with RL #1 on the same problems
    base, rows = data["RL #1 correctness"], []
    for name in list(RL)[1:] + list(DISTILL):
        rows.append([name, "steps 0-240", *paired(data[name], base, 0, ABLATION_STEP + 1)])
    rows.append([MAIN, "all steps", *paired(data[MAIN], base, 0, 10**9)])
    rows.append([MAIN, f"steps >= {LATE}", *paired(data[MAIN], base, LATE, 10**9)])
    write_csv("paired_vs_rl1.csv", ["run", "problems", "n", "diff_pct_points", "ci95_pct_points"], rows)

    # Answer categories: late in training (full runs) and steps 120-240 (all runs)
    rows = []
    for name in ["RL #1 correctness", "RL #2 finish + partial", MAIN]:
        rows.append([name, "late (>= 350)"] + [round(stage_mean(data[name], k, LATE, 10**9), 4) for k in keys])
    for name in ["RL #1 correctness", "RL #2 finish + partial"] + list(DISTILL):
        rows.append([name, "steps 120-240"] + [round(stage_mean(data[name], k, 120, ABLATION_STEP + 1), 4) for k in keys])
    write_csv("categories.csv", ["run", "stage"] + keys, rows)

    # Forgetting: every chat_eval task for the starting model, the main run and the teacher
    rows = []
    for label, log in [("Student (SFT, nanochat mixture)", "p5_sft_mix_eval.log"), ("OPD, step 466", "p5_opd_s466_full_eval.log")]:
        r = eval_results(log)
        rows.append([label] + [round(100 * r[t][0] / r[t][1], 2) for t in TASKS] + [r.get("ChatCORE", "")])
    rows.append(["Teacher (SFT + OpenMathInstruct-2, Part 2)"] + [part2[("SFT + OpenMathInstruct-2", t)] for t in TASKS] +
                [part2[("SFT + OpenMathInstruct-2", "ChatCORE")]])
    write_csv("forgetting.csv", ["model"] + TASKS + ["ChatCORE"], rows)

    # Curves for the plots
    rows = []
    for name in list(RL) + list(DISTILL):
        for r in passk[name]:
            rows.append([name, "passk", r[0], round(100 * r[1], 2), round(100 * r[8], 2)])
        last = max(p["step"] for p in data[name].values())
        for b in range(0, last + 1, BIN):
            vals = [v for s, v in rkl[name] if b <= s < b + BIN]
            rows.append([name, "train", b, round(100 * stage_mean(data[name], "correct", b, b + BIN), 2),
                         round(statistics.mean(vals), 4) if vals else ""])
    write_csv("curves.csv", ["run", "kind", "step", "value1", "value2"], rows)

    for name in ("results_table.csv", "ablations_step240.csv", "paired_vs_rl1.csv", "sft_repro.csv", "forgetting.csv"):
        print(f"\n== {name}\n" + open(os.path.join(RESULTS, name)).read())


if __name__ == "__main__":
    main()
