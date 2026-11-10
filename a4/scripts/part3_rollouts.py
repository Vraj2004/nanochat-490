"""Analyze the A4 Part 3 RL rollouts: categorize every sampled answer and every GSM8K problem.

Reads the rollout log written by scripts/chat_rl.py (one JSON line per training problem with its
16 completions and rewards, gzipped in a4/results/part3/rollouts) and the RL training log, and writes
CSVs plus a text summary to a4/results/part3 for a4/scripts/part3_plots.py.

Rollout idx is an index into nanochat's shuffled GSM8K train split, so the problems are loaded with
tasks/gsm8k.py (downloads GSM8K once to $NANOCHAT_BASE_DIR) and token counts use the nanochat
tokenizer ($NANOCHAT_BASE_DIR/tokenizer). Run from the repo root with the project environment:

    python a4/scripts/part3_rollouts.py
"""
import collections
import csv
import glob
import gzip
import json
import os
import re
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K, extract_answer

RESULTS = os.path.join(ROOT, "a4", "results", "part3")
MAX_NEW_TOKENS = 256  # chat_rl default
STEP_BIN = 25

# Answer categories, checked in this order (the first that applies wins)
CATEGORIES = [
    ("correct", "Correct"),
    ("wrong_format", "Right number, wrong format (e.g. 16.0)"),
    ("ran_out", "No answer: hit the 256-token limit"),
    ("ended_early", "No answer: ended the turn mid-solution"),
    ("intermediate", "Wrong: answered an intermediate quantity"),
    ("copied", "Wrong: repeated a number from the question"),
    ("unsupported", "Wrong: final number never computed in the answer"),
    ("setup", "Wrong: computed the wrong quantity"),
]
# Problem topics by keyword, checked in this order
TOPICS = [
    ("percent", re.compile(r"%|percent", re.I)),
    ("fractions & multiples", re.compile(r"\b\d+/\d+\b|\bhalf\b|\bthird|\bquarter|\bfourth|\bfifth|\btwice\b|\bdouble|\btriple|\btimes (as|more)", re.I)),
    ("time & rates", re.compile(r"\b(hour|minute|second|day|week|month|year)s?\b|\bmph\b", re.I)),
    ("money", re.compile(r"\$|dollar|\bcents?\b|\bprice|\bcosts?\b|\bpays?\b|\bearns?\b|\bsells?\b|\bbuys?\b", re.I)),
    ("counting & other", re.compile(r"")),
]
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers(text):
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


def same_number(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def categorize(completion, num_tokens, gold, intermediates, question_numbers):
    pred = extract_answer(completion)
    if pred is None:
        return "ran_out" if num_tokens >= MAX_NEW_TOKENS - 6 else "ended_early"
    if pred == gold:
        return "correct"
    if same_number(pred, gold):
        return "wrong_format"
    pred = pred.rstrip(".")
    if any(same_number(pred, x) for x in intermediates):
        return "intermediate"
    if pred in question_numbers:
        return "copied"
    if not any(same_number(pred, x) for x in numbers(completion.split("####")[0])):
        return "unsupported"
    return "setup"


def parse_rl_log(path):
    rewards, passk = [], []
    with open(path, errors="replace") as f:
        for line in f:
            if m := re.match(r"Step (\d+)/\d+ \| Average reward: ([\d.]+) \| Average sequence length: ([\d.]+)", line):
                rewards.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif m := re.match(r"Step (\d+) \| (Pass@1: .*)", line):
                passk.append([int(m.group(1))] + [float(v) for v in re.findall(r"Pass@\d+: ([\d.]+)", m.group(2))])
    return rewards, passk


def write_csv(name, header, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    # Training curves from the RL log
    rewards, passk = parse_rl_log(os.path.join(RESULTS, "d20_rl.log"))
    write_csv("rl_reward.csv", ["step", "reward", "sequence_length"], rewards)
    write_csv("rl_passk.csv", ["step"] + [f"pass{k}" for k in range(1, 9)], passk)

    tokenizer = get_tokenizer()
    task = GSM8K(subset="main", split="train")
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "rollouts", "rank*.jsonl.gz"))):
        with gzip.open(path, "rt") as f:
            rows.extend(json.loads(line) for line in f)
    rows.sort(key=lambda r: (r["step"], r["idx"]))

    keys = [k for k, _ in CATEGORIES]
    by_bin = collections.defaultdict(collections.Counter)
    problems, examples = [], collections.defaultdict(list)
    for r in rows:
        messages = task[r["idx"]]["messages"]
        question, parts = messages[0]["content"], messages[-1]["content"]
        gold = extract_answer(parts[-1]["text"])
        intermediates = [p["text"].strip() for p in parts if p["type"] == "python_output"]
        intermediates = [x for x in intermediates if not same_number(x, gold)]
        topic = next(name for name, pattern in TOPICS if pattern.search(question))
        cats = collections.Counter()
        for completion in r["completions"]:
            num_tokens = len(tokenizer.enc.encode(completion, allowed_special="all"))
            cat = categorize(completion, num_tokens, gold, intermediates, numbers(question))
            cats[cat] += 1
            if r["step"] >= 300 and len(examples[cat]) < 3:
                examples[cat].append((r["idx"], gold, question, completion))
        assert sum(r["rewards"]) == cats["correct"], "categories must agree with the reward"
        by_bin[r["step"] // STEP_BIN * STEP_BIN].update(cats)
        top_error = max((k for k in keys if k != "correct"), key=lambda k: cats[k])
        problems.append([r["idx"], r["step"], cats["correct"], sum(p["type"] == "python" for p in parts),
                         topic, top_error if cats["correct"] < 16 else ""])

    write_csv("answer_categories.csv", ["step_bin"] + keys, [[b] + [by_bin[b][k] for k in keys] for b in sorted(by_bin)])
    write_csv("problems.csv", ["idx", "step", "num_correct", "gold_steps", "topic", "most_common_error"], problems)

    # Text summary
    lines = []
    total = collections.Counter()
    for c in by_bin.values():
        total.update(c)
    n = sum(total.values())
    early = sum((by_bin[b] for b in by_bin if b < 100), collections.Counter())
    late = sum((by_bin[b] for b in by_bin if b >= 350), collections.Counter())
    lines.append(f"{len(rows)} problems x 16 samples = {n} answers, steps 0-{rows[-1]['step']}\n")
    lines.append(f"{'category':52s} {'all':>7s} {'steps<100':>10s} {'steps>=350':>11s}")
    for k, label in CATEGORIES:
        lines.append(f"{label:52s} {100*total[k]/n:6.1f}% {100*early[k]/sum(early.values()):9.1f}% {100*late[k]/sum(late.values()):10.1f}%")
    solved = [p[2] for p in problems]
    lines.append(f"\nproblems never solved (0/16): {solved.count(0)} ({100*solved.count(0)/len(solved):.1f}%), "
                 f"always solved (16/16): {solved.count(16)} ({100*solved.count(16)/len(solved):.1f}%)")
    for name, col in (("gold solution steps (calculator calls)", 3), ("topic", 4)):
        lines.append(f"\nsolve rate by {name}: n, mean, steps<100, steps>=350")
        groups = collections.defaultdict(list)
        for p in problems:
            groups[min(p[col], 6) if col == 3 else p[col]].append(p)
        for g in sorted(groups, key=lambda g: (str(type(g)), g)):
            ps = groups[g]
            rate = lambda sub: statistics.mean(p[2] / 16 for p in sub) if sub else float("nan")
            lines.append(f"  {str(g) + ('+' if g == 6 else ''):24s} {len(ps):5d}  {rate(ps):.3f}  "
                         f"{rate([p for p in ps if p[1] < 100]):.3f}  {rate([p for p in ps if p[1] >= 350]):.3f}")
    never = [p for p in problems if p[2] == 0]
    lines.append("\nmost common error on never-solved problems: " +
                 ", ".join(f"{k} {v}" for k, v in collections.Counter(p[5] for p in never).most_common()))
    lines.append("\nexamples (steps >= 300):")
    for k, label in CATEGORIES:
        for idx, gold, question, completion in examples[k]:
            lines.append(f"--- {label} | problem {idx}, gold {gold}\n  Q: {question}\n  A: {completion!r}")
    summary = "\n".join(lines)
    with open(os.path.join(RESULTS, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary.split("\nexamples")[0])


if __name__ == "__main__":
    main()
