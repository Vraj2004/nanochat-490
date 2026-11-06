"""Build the A3 Part 3 hardware table and figure from the training logs.

All runs train the same GeGLU picochat (d12, 2,520 steps, 1.0e18 FLOPs). The three single-GPU
runs are the Part 3 comparison; the 8x H200 run from Part 2 is a multi-GPU reference. Training
cost is training time times Modal's list price (`modal billing rates`, November 5, 2026), so it
leaves out container startup, compilation and evaluation.

Usage (from the repo root):

    python a3/scripts/part3_hardware.py
"""
import re
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A3_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = A3_DIR / "results" / "part3" / "logs"
# run -> (label, architecture, price per GPU-hour in USD, memory bandwidth in TB/s from nanochat/common.py)
RUNS = {
    "d12_hw_h100_part_two": ("H100", "Hopper", 3.95, 3.35),
    "d12_hw_a100_part_two": ("A100 80GB", "Ampere", 2.50, 2.0),
    "d12_hw_l40s_part_two": ("L40S", "Ada", 1.95, 0.864),
    "d12_geglu": ("8x H200 (Part 2)", "Hopper", 4.54, 4.8),
}


def last(pattern, text, cast=float):
    matches = re.findall(pattern, text)
    return cast(matches[-1].replace(",", "")) if matches else None


def parse(run):
    text = (LOG_DIR / f"{run}_train.log").read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    steps = re.findall(r"tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)", text)[5:]  # skip warmup/compile steps, as in Part 2
    return {
        "world_size": last(r"Distributed world size: (\d+)", text, int) or 1,
        "peak_tflops": last(r"Peak FLOPS \(BF16\): ([\d.e+]+)", text) / 1e12,
        "train_minutes": last(r"Total training time: ([\d.]+)m", text),
        "tok_per_sec": statistics.median(int(t.replace(",", "")) for t, _ in steps),
        "mfu": statistics.median(float(m) for _, m in steps),
        "val_bpb": last(r"Step \d+ \| Validation bpb: ([\d.]+)", text),
        "tokens": last(r"Total number of training tokens: ([\d,]+)", text, int),
    }


def main():
    rows = []
    for run, (label, arch, price, bandwidth) in RUNS.items():
        p = parse(run)
        hourly = price * p["world_size"]
        cost = hourly * p["train_minutes"] / 60
        rows.append({
            "run": run, "gpu": label, "architecture": arch, "num_gpus": p["world_size"],
            "peak_tflops_per_gpu": round(p["peak_tflops"]), "bandwidth_tbps_per_gpu": bandwidth,
            "price_per_hour": round(hourly, 2), "train_minutes": p["train_minutes"],
            "tok_per_sec": round(p["tok_per_sec"]), "tok_per_sec_per_gpu": round(p["tok_per_sec"] / p["world_size"]),
            "mfu_pct": round(p["mfu"], 1), "train_cost_usd": round(cost, 2),
            "cost_per_billion_tokens": round(hourly / (p["tok_per_sec"] * 3600) * 1e9, 2),
            "val_bpb": p["val_bpb"],
        })

    out_dir = A3_DIR / "results" / "part3"
    with open(out_dir / "hardware_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Figure: training time against training cost for the same model on each setup
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {"H100": "#3a63a8", "A100 80GB": "#3f7d3f", "L40S": "#a8862a", "8x H200 (Part 2)": "#a83a3a"}
    for r in rows:
        marker = "s" if r["num_gpus"] > 1 else "o"
        ax.scatter(r["train_minutes"], r["train_cost_usd"], s=70, marker=marker, color=colors[r["gpu"]], zorder=3)
        ax.annotate(f"{r['gpu']}\n{r['mfu_pct']:.0f}% MFU", (r["train_minutes"], r["train_cost_usd"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (minutes, log scale)")
    ax.set_ylabel("Training cost (USD)")
    ax.set_xlim(3, 400)
    ax.set_ylim(0, 6)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(A3_DIR / "figures" / "part3_hardware.pdf")

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
