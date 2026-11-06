"""Fit the A3 Part 5 scaling law from our picochat runs and compare its d20 prediction with the real run.

Reads the training logs the Modal launcher writes to the `nanochat-data` volume, fits
validation BPB against training FLOPs as a power law (BPB = a * C^-b, a straight line in
log-log space) on the small models, and extrapolates to the d20 nanochat.

Usage (from the repo root; needs the `modal` CLI to fetch logs):

    python a3/scripts/part5_scaling.py --fetch

Without --fetch it reads logs already in a3/results/part5/logs. If the d20 log is missing
or unfinished, it still prints the prediction, using d20's FLOP budget from the log header
or --target-flops.
"""
import re
import csv
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A3_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = A3_DIR / "results" / "part5" / "logs"
FIT_RUNS = {  # depth -> run name; all use partial RoPE (50%) and nanochat's default 12 tokens/param
    8: "d8_rope50_scaling",
    10: "d10_rope50_scaling",
    12: "d12_rope50",  # from Part 2
    14: "d14_rope50_scaling",
}
TARGET_RUN = (20, "d20_final_v2")


def fetch(run):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dest = LOG_DIR / f"{run}_train.log"
    subprocess.run(["modal", "volume", "get", "--force", "nanochat-data", f"logs/{run}_train.log", str(dest)],
                   check=False, capture_output=True)


def last(pattern, text, cast=float):
    matches = re.findall(pattern, text)
    return cast(matches[-1].replace(",", "")) if matches else None


def parse_log(run):
    path = LOG_DIR / f"{run}_train.log"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    num_iterations = last(r"step \d+/(\d+)", text, int)
    return {
        "run": run,
        "flops": last(r"Total training FLOPs estimate: ([\d.e+]+)", text),
        "tokens": last(r"Total number of training tokens: ([\d,]+)", text, int),
        "scaling_params": (last(r"transformer_matrices\s*: ([\d,]+)", text, int) or 0)
                          + (last(r"lm_head\s*: ([\d,]+)", text, int) or 0),
        "val_bpb": last(r"Step \d+ \| Validation bpb: ([\d.]+)", text),
        "final_step": last(r"Step (\d+) \| Validation bpb", text, int),
        "num_iterations": num_iterations,
        "core": last(r"Step \d+ \| CORE metric: ([\d.]+)", text),
        "train_minutes": last(r"Total training time: ([\d.]+)m", text),
        "finished": "Total training time:" in text,
    }


def fit(flops, bpb):
    """Least-squares line in log-log space. Returns (a, b) for BPB = a * C^-b."""
    slope, intercept = np.polyfit(np.log(flops), np.log(bpb), 1)
    return float(np.exp(intercept)), float(-slope)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="download the training logs from the Modal volume first")
    parser.add_argument("--target-flops", type=float, default=None, help="d20 FLOP budget if its log is unavailable")
    args = parser.parse_args()

    if args.fetch:
        for run in [*FIT_RUNS.values(), TARGET_RUN[1]]:
            fetch(run)

    points = {d: parse_log(run) for d, run in FIT_RUNS.items()}
    missing = [FIT_RUNS[d] for d, p in points.items() if not p or not p["finished"]]
    if missing:
        raise SystemExit(f"Missing or unfinished fit runs: {missing}. Run with --fetch once they finish.")
    depths = sorted(points)
    C = np.array([points[d]["flops"] for d in depths])
    L = np.array([points[d]["val_bpb"] for d in depths])
    a, b = fit(C, L)

    target = parse_log(TARGET_RUN[1])
    target_flops = args.target_flops or (target and target["flops"])
    if not target_flops:
        raise SystemExit("No d20 FLOP budget: fetch its log (it is printed at the start) or pass --target-flops.")
    predicted = a * target_flops ** -b

    # Leave-one-out refits show how sensitive the prediction is to any single point
    loo = []
    for i in range(len(depths)):
        keep = [j for j in range(len(depths)) if j != i]
        a_i, b_i = fit(C[keep], L[keep])
        loo.append(a_i * target_flops ** -b_i)

    actual = target["val_bpb"] if target and target["finished"] else None
    result = {
        "law": f"BPB = {a:.4g} * C^-{b:.4f}",
        "a": a, "b": b,
        "target_run": TARGET_RUN[1], "target_flops": target_flops,
        "predicted_bpb": predicted,
        "loo_min": min(loo), "loo_max": max(loo),
        "actual_bpb": actual,
        "error_pct": None if actual is None else 100 * (actual - predicted) / predicted,
    }

    out_dir = A3_DIR / "results" / "part5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scaling_points.csv", "w", newline="") as f:
        fields = ["depth", "run", "scaling_params", "tokens", "flops", "val_bpb", "core", "train_minutes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in depths + ([TARGET_RUN[0]] if target else []):
            p = points.get(d) or target
            writer.writerow({"depth": d, **{k: p[k] for k in fields if k != "depth"}})
    (out_dir / "scaling_fit.json").write_text(json.dumps(result, indent=2))

    # Figure: fitted law with the small-model points, the d20 prediction and (when available) the actual d20
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grid = np.logspace(np.log10(C.min() / 1.5), np.log10(target_flops * 1.5), 100)
    ax.plot(grid, a * grid ** -b, color="#3a63a8", linewidth=1.5, label=f"fit: BPB = {a:.3g}·C$^{{-{b:.3f}}}$")
    ax.scatter(C, L, color="#3a63a8", zorder=3, label="picochat runs (fit)")
    for d, c, l in zip(depths, C, L):
        ax.annotate(f"d{d}", (c, l), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.errorbar([target_flops], [predicted], yerr=[[predicted - min(loo)], [max(loo) - predicted]],
                fmt="o", mfc="white", color="#a83a3a", capsize=4, zorder=4, label="d20 predicted (leave-one-out range)")
    if actual is not None:
        ax.scatter([target_flops], [actual], marker="*", s=160, color="#a83a3a", zorder=5, label="d20 actual")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.70, 0.75, 0.80, 0.85, 0.90, 0.95]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Training compute (FLOPs)")
    ax.set_ylabel("Final validation BPB")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(A3_DIR / "figures" / "part5_scaling.pdf")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
