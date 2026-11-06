"""Print the A3 Part 2 summary numbers and build the figures from the team's W&B runs.

Reads the runs only; nothing is written back to W&B. Needs WANDB_API_KEY in the
environment. Run from the repository root:

    uv run --no-project --python 3.12 --with wandb --with matplotlib \
        python a3/scripts/part2_wandb.py
"""

from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

PROJECT = "vraj-shah2121-university-of-toronto/nanochat"
A3_DIR = Path(__file__).resolve().parents[1]
FIGURES = A3_DIR / "figures"

# name -> (W&B run id, line color, marker, legend label). Color and marker follow the
# model in every figure, so the runs can also be told apart in grayscale.
RUNS = {
    "baseline": ("07y65b6s", "#2a78d6", "o", "Baseline"),
    "geglu": ("flkribau", "#eb6834", "s", "GeGLU"),
    "rope50": ("nnf5934f", "#1baf7a", "^", "Partial RoPE (50%)"),
}
CHANGES = ("geglu", "rope50")
# Partial RoPE rerun with a different seed. Used to estimate seed noise and left out of the figures.
SEED_RUN = ("rope50_seed1337", "6xgo0v64")
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def load_run(api, run_id):
    run = api.run(f"{PROJECT}/{run_id}")
    rows = list(run.scan_history())
    train = [r for r in rows if r.get("train/loss") is not None and r.get("step", 0) > 0]
    core = [r for r in rows if r.get("core_metric") is not None]
    return {
        "gpu": run.metadata["gpu"],
        "gpu_count": run.metadata["gpu_count"],
        "val": {r["step"]: r["val/bpb"] for r in rows if r.get("val/bpb") is not None},
        "loss": {r["step"]: r["train/loss"] for r in train},
        "core": {r["step"]: r["core_metric"] for r in core},
        "tasks": {
            k.split(".", 1)[1]: v for k, v in core[-1].items() if k.startswith("centered_results.")
        },
        # The first logged steps include warm-up and start-up time, so skip them.
        "tok_per_sec": median(r["train/tok_per_sec"] for r in train[5:]),
        "mfu": median(r["train/mfu"] for r in train[5:]),
        "train_seconds": run.summary["total_training_time"],
        "steps": run.summary["step"],
    }


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=9, color=TEXT)
    ax.set_xlabel("Training step", fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_series(ax, series, name, start=0):
    _, color, marker, label = RUNS[name]
    steps = [s for s in sorted(series) if s >= start]
    ax.plot(steps, [series[s] for s in steps], color=color, marker=marker, markersize=3.5,
            linewidth=1.3, label=label, markeredgecolor="white", markeredgewidth=0.4)


def plot_training(data):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for name in RUNS:
        plot_series(axes[0], data[name]["val"], name, start=250)
        plot_series(axes[1], data[name]["loss"], name, start=200)
    base = data["baseline"]["val"]
    axes[2].axhline(0, color=RUNS["baseline"][1], linewidth=1.3)
    for name in CHANGES:
        gap = {s: v - base[s] for s, v in data[name]["val"].items() if s >= 250}
        plot_series(axes[2], gap, name)
    style_axis(axes[0], "Validation loss", "bits per byte")
    style_axis(axes[1], "Training loss (smoothed)", "loss")
    style_axis(axes[2], "Validation gap to baseline", "bits per byte (above 0 = worse)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "part2_training.pdf")
    plt.close(fig)


def plot_tasks(data):
    base = data["baseline"]["tasks"]
    tasks = sorted(base, key=lambda t: mean(data[n]["tasks"][t] - base[t] for n in CHANGES))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True, sharex=True)
    for ax, name in zip(axes, CHANGES):
        _, color, marker, label = RUNS[name]
        gaps = [data[name]["tasks"][t] - base[t] for t in tasks]
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.axvline(mean(gaps), color=color, linewidth=1, linestyle="--")
        ax.scatter(gaps, range(len(tasks)), color=color, marker=marker, s=26, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.set_title(f"{label}: mean {mean(gaps):+.4f}", loc="left", fontsize=9, color=TEXT)
        ax.set_xlabel("Score gap to baseline (right = better)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_yticks(range(len(tasks)))
    axes[0].set_yticklabels([t.replace("_", " ") for t in tasks])
    fig.tight_layout()
    fig.savefig(FIGURES / "part2_core_tasks.pdf")
    plt.close(fig)


def print_summary(data):
    base = data["baseline"]
    rows = []
    for name, run in data.items():
        last = max(run["val"])
        last_core = max(run["core"])
        rows.append({
            "model": name,
            "hardware": f"{run['gpu_count']}x {run['gpu']}",
            "steps": run["steps"],
            "val_bpb": round(run["val"][last], 5),
            "val_bpb_gap": round(run["val"][last] - base["val"][last], 5),
            "core_step_2000": round(run["core"][2000], 4),
            "core_final": round(run["core"][last_core], 4),
            "core_final_gap": round(run["core"][last_core] - base["core"][last_core], 4),
            "median_tok_per_sec": round(run["tok_per_sec"]),
            "median_mfu_pct": round(run["mfu"], 2),
            "train_minutes": round(run["train_seconds"] / 60, 2),
        })
    for row in rows:
        print(row)
    for name in CHANGES:
        gaps = [data[name]["tasks"][t] - base["tasks"][t] for t in base["tasks"]]
        se = stdev(gaps) / len(gaps) ** 0.5
        print(f"{name}: CORE gap {mean(gaps):+.4f}, task SE {se:.4f}, t {mean(gaps) / se:.2f}, "
              f"up {sum(g > 0 for g in gaps)}, down {sum(g < 0 for g in gaps)}")
        val_gaps = {s: round(v - base["val"][s], 5) for s, v in data[name]["val"].items() if s > 0}
        print(f"{name}: val gaps by step {val_gaps}")
    seed42, seed1337 = data["rope50"], data[SEED_RUN[0]]
    print(f"seed difference: val {abs(seed1337['val'][2520] - seed42['val'][2520]):.5f}, "
          f"CORE {abs(seed1337['core'][2520] - seed42['core'][2520]):.4f}")


def main():
    api = wandb.Api(timeout=120)
    data = {name: load_run(api, spec[0]) for name, spec in RUNS.items()}
    data[SEED_RUN[0]] = load_run(api, SEED_RUN[1])
    FIGURES.mkdir(exist_ok=True)
    plot_training(data)
    plot_tasks(data)
    print_summary(data)


if __name__ == "__main__":
    main()
