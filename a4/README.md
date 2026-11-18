# A4: RL-ing nanochat

`a4.tex` is the report (`A4.pdf` once built). This folder also has the scripts, results and logs behind every number in it.

The training code is in our nanochat fork, https://github.com/Vraj2004/nanochat-490, on the `a4` branch, and the script paths below are relative to that repository's root.

## Building the report

```
latexmk -pdf a4.tex
```

Or upload the folder to Overleaf. Figures are in `figures/` and are made by the scripts below.

## Code changes for A4

| File | Change |
|---|---|
| `scripts/chat_sft.py` | `--extra-data` / `--extra-rows` add MetaMathQA or OpenMathInstruct-2 to the mixture, `--output-tag`, and `rope_frac` / `mlp_type` are now saved in the checkpoint config |
| `tasks/metamathqa.py`, `tasks/openmathinstruct.py` | the GSM8K-style rows of each dataset, answers ending in `#### N` |
| `scripts/chat_rl.py` | logs every rollout to `chatrl_rollouts/<tag>/rank*.jsonl`; `--env` selects the training reward, `--output-tag`; correctness logged separately from the shaped reward |
| `tasks/gsm8k.py` | `GSM8KShaped`: correctness plus the finish bonus and partial credit (Part 4) |
| `modal_train.py` | `chat` entrypoint that runs `chat_sft`, `chat_rl` or `chat_eval` on Modal |
| `scripts/chat_rl.py` (Part 5) | `--distill teacher\|self`, `--teacher-tag`, `--reward-weight` and `--stop-at-step` for on-policy distillation; reverse KL logged to W&B |
| `nanochat/distill.py`, `tests/test_distill.py` | teacher log-probs aligned to the student's tokens, the SDPO self-teacher prompt, and CPU tests for both |

All runs use the default settings otherwise, on 8x H100 (`modal run --detach modal_train.py::chat --script <script> --log-name <name> --args "..."`; the commands are in the `modal_train.py` docstring).

## Where each result comes from

| Part | Script | Inputs | Outputs |
|---|---|---|---|
| 2 | `scripts/part2_sft.py` | `results/part2/logs/` (chat_sft and chat_eval logs from Modal) | `results/part2/chat_eval.csv`, `results/part2/sft_runs.csv`, `figures/part2_chat_eval.pdf` |
| 3 | `scripts/part3_rollouts.py` | `results/part3/rollouts/` (gzipped rollout logs), `results/part3/d20_rl.log` | `results/part3/{answer_categories,problems,rl_reward,rl_passk}.csv`, `results/part3/summary.txt` |
| 3 | `scripts/part3_plots.py` | the CSVs above, `results/part3/karpathy_rl_digitized.csv` | `figures/part3_{curves,categories,difficulty}.pdf` |
| 4 | `scripts/part4_compare.py` | `results/part4/rollouts/`, `results/part4/logs/`, the Part 3 rollouts | `results/part4/{results_table,paired_vs_original,categories_by_run,behavior_by_step,passk_by_run}.csv` |
| 4 | `scripts/part4_plots.py` | the CSVs above | `figures/part4_{curves,behavior,mistakes}.pdf` |
| 5 | `scripts/part5_compare.py` | `results/part5/rollouts/`, `results/part5/logs/`, the Part 3 and 4 rollouts and logs | `results/part5/{sft_repro,results_table,ablations_step240,paired_vs_rl1,categories,forgetting,curves}.csv` |
| 5 | `scripts/part5_plots.py` | the CSVs above | `figures/part5_{curves,mistakes}.pdf` |

Run them from the repository root. The plotting scripts only need matplotlib:

```
uv run --no-project --python 3.12 --with matplotlib python a4/scripts/part3_plots.py
```

`part3_rollouts.py` and `part4_compare.py` import nanochat (GSM8K and the tokenizer), so run them with the project environment, e.g. `python a4/scripts/part4_compare.py`. They read the tokenizer from `$NANOCHAT_BASE_DIR/tokenizer` and download GSM8K there on first use.

`results/part3/karpathy_rl_digitized.csv` holds values read by eye off the curves in Karpathy's post (`results/part3/karpathy_rl_curves.png`), accurate to about +-0.003.

## W&B runs

Entity `vraj-shah2121-university-of-toronto` on the CoreWeave W&B instance.

| Run | Project | ID |
|---|---|---|
| SFT, nanochat mixture | `nanochat-sft` | `wxucp5xt` |
| SFT + MetaMathQA | `nanochat-sft` | `w1r6t2m8` |
| SFT + OpenMathInstruct-2 | `nanochat-sft` | `wzm5y72r` |
| RL #1, correctness only | `nanochat-rl` | `73elnw2o` |
| RL #2, finish + partial | `nanochat-rl` | `2i9jo316` |
| RL #3, finish | `nanochat-rl` | `p0ktopcl` |
| RL #4, partial | `nanochat-rl` | `qa8692ci` |
