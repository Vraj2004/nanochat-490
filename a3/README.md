# A3: Pre-training nanochat

`a3.tex` is the report (`a3.pdf` once built). This folder also has the scripts, results and logs behind every number in it.

The training code is in our nanochat fork, https://github.com/Vraj2004/nanochat-490, on `master`. This folder is also kept there as `a3/`, and the script paths below are relative to that repository's root.

## Building the report

```
latexmk -pdf a3.tex
```

Or upload the folder to Overleaf. Figures are in `figures/`. The two diagrams (`nanochat_oct_architecture`, `proposed_changes`) have an `.svg` source next to the `.pdf` that LaTeX includes; the plots are made by the scripts below.

## Where each result comes from

| Part | Script | Inputs | Outputs |
|---|---|---|---|
| 2 | `scripts/part2_wandb.py` | W&B runs (needs `WANDB_API_KEY`) | `figures/part2_training.pdf`, `figures/part2_core_tasks.pdf`, numbers for the Part 2 results table |
| 3 | `scripts/part3_hardware.py` | `results/part3/logs/` | `results/part3/hardware_summary.csv`, `figures/part3_hardware.pdf` |
| 4 | `scripts/part4_passkey.py` | `results/part4/` (passkey CSVs from `scripts/passkey_eval.py`, run on Modal by `modal_train.py::context_ext`) | `figures/part4_passkey.pdf` |
| 4 | `scripts/part4_doclen.py` | one ClimbMix shard and the tokenizer, copied from the Modal volume | `results/part4/doclen_stats.txt` |
| 5 | `scripts/part5_scaling.py` | `results/part5/logs/` (`--fetch` downloads them from Modal) | `results/part5/scaling_*.{csv,json}`, `figures/part5_scaling.pdf` |
| 5 | `scripts/emergent_eval.py` (repo root, run on Modal) | d12 and d20 checkpoints | `results/part5/emergent_results.csv`, `emergent_summary.json` |

Run them from the repository root, e.g. `python a3/scripts/part3_hardware.py`. The logs are force-added because `.gitignore` excludes `*.log` (meant for LaTeX build files).

`results/part5/emergent_results.csv` has both the automatic verdicts (`pico_correct`, `nano_correct`) and the verdicts after we read every completion (`pico_reviewed`, `nano_reviewed`, with the reason in `review_note`). The report uses the reviewed ones.

## Training code

The training changes live outside this folder, in the main code:

- `nanochat/gpt.py`, `scripts/base_train.py`: the ablation flags `--mlp-type=geglu`, `--rope-frac=0.5`, and `--seed` (default 42). The defaults reproduce the July 2026 architecture.
- `modal_train.py`: runs everything on Modal. `prepare` downloads data and trains the tokenizer, `launch` starts a training run that keeps going if the laptop disconnects, `context_ext` runs the Part 4 experiment (512, then 2,048, then the evals), and `emergent` runs the Part 5 eval.
- `scripts/base_train.py`: `--stop-at-step` (Part 4) saves a checkpoint and exits early while keeping the full run's learning-rate schedule.
- `scripts/passkey_eval.py`: the Part 4 passkey retrieval eval.
- `scripts/emergent_eval.py`: the Part 5 emergent-abilities eval.

Every run logs to the team's W&B project under the run names used in the report. For example, the final model and its baseline:

```
NANOCHAT_GPU=H200:8 modal run --detach modal_train.py::launch --run-name d20_final_v2 --depth 20 --extra-args "--rope-frac=0.5 --save-every=1000"
NANOCHAT_GPU=H200:8 modal run --detach modal_train.py::launch --run-name d20_baseline --depth 20 --extra-args "--save-every=1000"
```

The full list of Part 5 commands is in PR #6 of the team repo.
