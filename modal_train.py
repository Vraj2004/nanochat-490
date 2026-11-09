"""
Run nanochat pretraining on Modal (picochat ablations, Part Two).

One-time setup on your machine:
    pip install modal
    modal setup                                   # log in
    modal secret create wandb-secret WANDB_API_KEY=<your key>

1) Prepare data + tokenizer (CPU only, cheap, run once):
    modal run modal_train.py::prepare

2) Train the baseline picochat (d12) and evaluate it:
    modal run modal_train.py::train --run-name d12_baseline

   Later, for an ablation, pass a different name/tag and any extra flags:
    modal run modal_train.py::train --run-name d12_myablation --extra-args "--some-flag=1"

3) Context extension (Part Four): train at a short sequence length, then resume at 2048 for the
   rest of the same LR schedule, then run the passkey eval on both checkpoints:
    modal run --detach modal_train.py::context_ext --run-name d12_ctx512 --extra-args "--save-every=500"

   With --detach the job is handed off and the terminal can be closed. Rerunning with the same
   --run-name resumes from its latest checkpoint; use a new name to start over.

   Evaluate any checkpoint on its own with the passkey eval:
    modal run modal_train.py::passkey --model-tag d12_ctx512 --step 2016 --widen

4) A4: SFT the base model, then RL on GSM8K, evaluating each (logs go to /data/logs/<log-name>.log):
    modal run --detach modal_train.py::chat --script chat_sft --log-name d20_sft --args "--model-tag=d20_final_v2 --run=d20_sft"
    modal run --detach modal_train.py::chat --script chat_eval --log-name d20_sft_eval --args "-i sft -g d20_final_v2"
    modal run --detach modal_train.py::chat --script chat_rl --log-name d20_rl --args "--model-tag=d20_final_v2 --run=d20_rl"
    modal run --detach modal_train.py::chat --script chat_eval --log-name d20_rl_eval --args "-i rl -g d20_final_v2 -a GSM8K"

To see why Flash Attention 3 fails to load (uses 1 GPU briefly):
    modal run modal_train.py::check_fa3

Everything (shards, tokenizer, checkpoints, logs) lives on the Modal volume
"nanochat-data", mounted at /data, so it persists across runs.
"""

import os
import re
import sys
import shlex
import subprocess

import modal

APP_NAME = "nanochat-picochat"
VOLUME_DIR = "/data"
REPO_DIR = "/root/nanochat"

# GPU config. Defaults to 8xH100. Override with NANOCHAT_GPU=H100:1 etc. to change hardware.
GPU = os.environ.get("NANOCHAT_GPU", "H100:8")
NUM_GPUS = int(GPU.split(":")[1]) if ":" in GPU else 1

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "filelock>=3.19.0",
        # pinned to uv.lock: newer kernels releases change the get_kernel API that FA3 loading uses
        "kernels==0.11.7",
        "huggingface-hub==0.34.4",
        "numpy>=1.26.0",
        "psutil>=7.1.0",
        "pyarrow>=21.0.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "wandb>=0.21.3",
        "requests",
    )
    .env({
        "NANOCHAT_BASE_DIR": VOLUME_DIR,
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": REPO_DIR,
        "NANOCHAT_GPU": GPU,  # keep NUM_GPUS in the container in sync with the GPUs requested
    })
    # mounted at container start (not baked in), so code edits don't rebuild the image
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path=REPO_DIR,
        ignore=[".venv", ".git", "__pycache__", "*.ipynb", "dev/*.png"],
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("nanochat-data", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")


def _run(cmd: str):
    print(f"\n>>> {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True, cwd=REPO_DIR)  # bash for pipefail


@app.function(volumes={VOLUME_DIR: volume}, cpu=16, memory=32768, timeout=3 * 3600)
def prepare(num_shards: int = 60):
    """Download pretraining shards and train the tokenizer (CPU only).

    d12 at the default --target-param-data-ratio=12 needs ~1.3B tokens, i.e. ~40 shards
    (each shard is ~250MB of text). 60 gives headroom for ablations without re-downloading.
    """
    _run(f"python -m nanochat.dataset -n {num_shards} -w 16")
    if not os.path.exists(os.path.join(VOLUME_DIR, "tokenizer", "tokenizer.pkl")):
        _run("python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768")
    _run("python -m scripts.tok_eval")
    volume.commit()


def _train(run_name: str, model_tag: str, depth: int, extra_args: str):
    log_dir = os.path.join(VOLUME_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    train_cmd = (
        f"torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.base_train -- "
        f"--depth={depth} --run={shlex.quote(run_name)} --model-tag={shlex.quote(model_tag)} "
        f"--device-batch-size=32 --save-every=-1 {extra_args}"
    )
    _run(f"set -o pipefail; {train_cmd} 2>&1 | tee {log_dir}/{run_name}_train.log")
    volume.commit()


def _base_eval(log_name: str, model_tag: str, step: int = None):
    log_dir = os.path.join(VOLUME_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    step_arg = f"--step={step}" if step is not None else ""
    eval_cmd = (
        f"torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.base_eval -- "
        f"--model-tag={shlex.quote(model_tag)} {step_arg} --device-batch-size=32"
    )
    _run(f"set -o pipefail; {eval_cmd} 2>&1 | tee {log_dir}/{log_name}_eval.log")
    volume.commit()


def _last_complete_step(checkpoint_dir: str) -> int:
    """Latest step with metadata and every rank's optimizer shard saved (0 if none)."""
    files = set(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else set()
    steps = [int(m.group(1)) for f in files if (m := re.fullmatch(r"meta_(\d+)\.json", f))]
    complete = [s for s in steps if all(f"optim_{s:06d}_rank{r}.pt" in files for r in range(NUM_GPUS))]
    return max(complete, default=0)


def _passkey(model_tag: str, step: int, widen: bool, extra_args: str):
    log_dir = os.path.join(VOLUME_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    suffix = "_widened" if widen else ""
    cmd = (
        f"torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.passkey_eval -- "
        f"--model-tag={shlex.quote(model_tag)} --step={step} {'--widen-windows' if widen else ''} {extra_args}"
    )
    _run(f"set -o pipefail; {cmd} 2>&1 | tee {log_dir}/{model_tag}_step{step}{suffix}_passkey.log")
    volume.commit()


@app.function(
    volumes={VOLUME_DIR: volume},
    gpu=GPU,
    secrets=[wandb_secret],
    timeout=4 * 3600,
)
def train_remote(run_name: str, depth: int, extra_args: str, run_eval: bool):
    _train(run_name, run_name, depth, extra_args)
    if run_eval:
        _base_eval(run_name, run_name)


@app.function(
    volumes={VOLUME_DIR: volume},
    gpu=GPU,
    secrets=[wandb_secret],
    timeout=6 * 3600,
)
def context_ext_remote(run_name: str, depth: int, num_iterations: int, switch_step: int, short_len: int,
                       long_len: int, extra_args: str, run_eval: bool, passkey_args: str):
    # Both phases run in this one container: the optimizer state is sharded per GPU, so phase 2 must
    # resume on the same number of GPUs as phase 1. Both write to the same checkpoint dir (model tag).
    horizon = f"--num-iterations={num_iterations}"
    last = _last_complete_step(os.path.join(VOLUME_DIR, "base_checkpoints", run_name))
    if last:
        print(f"Found a checkpoint at step {last}, resuming from there", flush=True)
    if last < switch_step:
        # phase 1: 4x the rows at 1/4 the length keeps tokens per micro-batch (and grad accumulation) unchanged
        short_batch = 32 * 2048 // short_len
        resume = f"--resume-from-step={last}" if last else ""
        _train(f"{run_name}_p1_seq{short_len}", run_name, depth,
               f"{horizon} {resume} --stop-at-step={switch_step} --max-seq-len={short_len} --device-batch-size={short_batch} {extra_args}")
    if last < num_iterations:
        # phase 2: resume the same LR schedule, optimizer state and data position at the long sequence length
        _train(f"{run_name}_p2_seq{long_len}", run_name, depth,
               f"{horizon} --resume-from-step={max(last, switch_step)} --max-seq-len={long_len} {extra_args}")
    # Evals run on both checkpoints by step, so a resumed run that skipped a phase still evaluates it.
    # Passkey first: checkpoint 1 as trained, checkpoint 1 with windows widened to long_len, checkpoint 2
    _passkey(run_name, switch_step, False, f"--seq-len={long_len} {passkey_args}")
    _passkey(run_name, switch_step, True, f"--seq-len={long_len} {passkey_args}")
    _passkey(run_name, num_iterations, False, f"--seq-len={long_len} {passkey_args}")
    if run_eval:
        _base_eval(f"{run_name}_step{switch_step}", run_name, switch_step)
        _base_eval(f"{run_name}_step{num_iterations}", run_name, num_iterations)


@app.function(
    volumes={VOLUME_DIR: volume},
    gpu=GPU,
    secrets=[wandb_secret],
    timeout=6 * 3600,
)
def chat_remote(script: str, log_name: str, args: str):
    log_dir = os.path.join(VOLUME_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    cmd = f"torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.{script} -- {args}"
    _run(f"set -o pipefail; {cmd} 2>&1 | tee {log_dir}/{log_name}.log")
    volume.commit()


@app.function(volumes={VOLUME_DIR: volume}, gpu=GPU, timeout=2 * 3600)
def passkey_remote(model_tag: str, step: int, widen: bool, extra_args: str):
    _passkey(model_tag, step, widen, extra_args)


@app.function(volumes={VOLUME_DIR: volume}, gpu="H100:1", timeout=30 * 60)
def emergent_remote(pico_tag: str, nano_tag: str):
    out_dir = os.path.join(VOLUME_DIR, "part5_emergent")
    _run(f"python -m scripts.emergent_eval --pico-tag={shlex.quote(pico_tag)} --nano-tag={shlex.quote(nano_tag)} "
         f"--out-dir={out_dir} 2>&1 | tee {out_dir}.log")
    volume.commit()


@app.function(gpu="H100:1", timeout=10 * 60)
def check_fa3():
    """Try to load Flash Attention 3 and print the real error if it fails."""
    import platform
    import traceback
    from importlib.metadata import version

    import torch

    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | kernels {version('kernels')}")
    print(f"python {platform.python_version()} | glibc {platform.libc_ver()[1]}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | SM {torch.cuda.get_device_capability()}")
    try:
        from kernels import get_kernel

        print("FA3 LOADED:", get_kernel("varunneal/flash-attention-3").flash_attn_interface)
    except Exception:
        print("FA3 FAILED:")
        traceback.print_exc()


@app.local_entrypoint()
def train(run_name: str = "d12_baseline", depth: int = 12, extra_args: str = "", eval: bool = True):
    print(f"Launching {run_name} (depth={depth}) on {GPU} with extra args: {extra_args!r}")
    train_remote.remote(run_name, depth, extra_args, eval)


@app.local_entrypoint()
def context_ext(run_name: str = "d12_ctx512", depth: int = 12, num_iterations: int = 2520, switch_frac: float = 0.8,
                short_len: int = 512, long_len: int = 2048, extra_args: str = "", eval: bool = True,
                passkey_args: str = ""):
    # 2520 steps x 524,288 tokens = d12 at the default --target-param-data-ratio=12 (same budget as d12_baseline)
    switch_step = round(switch_frac * num_iterations)
    print(f"Launching {run_name} (depth={depth}) on {GPU}: steps 0-{switch_step} at seq len {short_len}, "
          f"{switch_step}-{num_iterations} at {long_len}, extra args: {extra_args!r}")
    call = context_ext_remote.spawn(run_name, depth, num_iterations, switch_step, short_len, long_len, extra_args, eval, passkey_args)
    # Blocking on .remote() lets an interrupted terminal (Ctrl+C, laptop sleep) cancel the call even with
    # --detach. A spawned call in a detached app keeps running after this exits.
    if "--detach" in sys.argv:
        print(f"Spawned {call.object_id} in the background, safe to close this terminal.")
        print("Follow it on the app page on modal.com, or with `modal app list` then `modal app logs <app id> -f`")
    else:
        call.get()


@app.local_entrypoint()
def passkey(model_tag: str, step: int, widen: bool = False, extra_args: str = ""):
    print(f"Passkey eval of {model_tag} step {step} (widen={widen}) on {GPU} with extra args: {extra_args!r}")
    passkey_remote.remote(model_tag, step, widen, extra_args)


@app.local_entrypoint()
def chat(script: str, log_name: str, args: str = ""):
    """A4: run scripts.chat_sft, scripts.chat_rl or scripts.chat_eval with the given args."""
    print(f"Launching scripts.{script} on {GPU} with args: {args!r}")
    call = chat_remote.spawn(script, log_name, args)
    if "--detach" in sys.argv:
        print(f"Spawned {call.object_id} in the background, safe to close this terminal.")
    else:
        call.get()


@app.local_entrypoint()
def emergent(pico_tag: str = "d12_rope50", nano_tag: str = "d20_final_v2"):
    """Part 5: compare picochat and nanochat completions on simple prompts (1 GPU, a few minutes)."""
    emergent_remote.remote(pico_tag, nano_tag)


@app.local_entrypoint()
def launch(run_name: str, depth: int = 12, extra_args: str = "", eval: bool = True):
    """Fire-and-forget: start training and return immediately. Use with `modal run --detach`."""
    call = train_remote.spawn(run_name, depth, extra_args, eval)
    print(f"Spawned {run_name} (depth={depth}) on {GPU} with extra args: {extra_args!r}; call id {call.object_id}")
