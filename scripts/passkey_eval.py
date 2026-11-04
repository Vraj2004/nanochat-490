"""
Passkey retrieval eval for base models: can the model recall a fact from d tokens back?

A random 5-digit passkey is hidden in filler text from the val shard and asked for at the end:

    <bos> [filler] The pass key is 48213. Remember it. 48213 is the pass key. [d filler tokens] The pass key is

We sweep the distance d. Every sequence has the same total length and each example reuses the
same passkey, distractors and filler at every distance, so only the needle's position moves.
The model does not generate: the true passkey is scored against random distractors by summed
log-prob and counts as correct if it ranks first. Chance accuracy is 1/num_candidates.

Ref: Mohtashami & Jaggi 2023, "Landmark Attention" (arXiv:2305.16300).

Examples:

    # Evaluate a checkpoint at 2048 tokens using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.passkey_eval --model-tag d12_ctx --step 2016

    # A model trained at a short --max-seq-len also has short sliding windows. Widen them to --seq-len
    # to test pure RoPE length extrapolation instead.
    torchrun --nproc_per_node=8 -m scripts.passkey_eval --model-tag d12_ctx --step 2016 --widen-windows
"""
import os
import math
import random
import argparse
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.dataset import parquets_iter_batched

NEEDLE = " The pass key is {key}. Remember it. {key} is the pass key."
QUERY = " The pass key is"


def load_filler(tokenizer, num_tokens):
    """A stream of val-shard tokens (no BOS) to cut filler windows from."""
    stream = []
    for texts in parquets_iter_batched("val"):
        for ids in tokenizer.encode([text + "\n\n" for text in texts]):
            stream.extend(ids)
        if len(stream) >= num_tokens:
            return stream[:num_tokens]
    raise ValueError(f"val shard has only {len(stream):,} tokens, need {num_tokens:,}")


def build_example(rng, tokenizer, filler, seq_len, num_candidates):
    """Draw one example's passkey, distractors and filler window. These are shared by all distances."""
    key = rng.randrange(10000, 100000)
    answer = tokenizer.encode(f" {key}")
    keys = [key]
    while len(keys) < num_candidates:
        k = rng.randrange(10000, 100000)
        # distractors have the same token count as the answer, so summed log-probs are comparable
        if k not in keys and len(tokenizer.encode(f" {k}")) == len(answer):
            keys.append(k)
    rng.shuffle(keys) # random answer position, so ties can't favor the right answer
    needle = tokenizer.encode(NEEDLE.format(key=key))
    query = tokenizer.encode(QUERY)
    num_filler = seq_len - 1 - len(needle) - len(query) - len(answer) # 1 for BOS
    start = rng.randrange(len(filler) - num_filler)
    return {
        "needle": needle,
        "query": query,
        "filler": filler[start:start + num_filler],
        "candidates": [tokenizer.encode(f" {k}") for k in keys],
        "answer_idx": keys.index(key),
    }


@torch.no_grad()
def predict(model, context, candidates, device):
    """Index of the candidate continuation with the highest summed log-prob given the context."""
    rows = torch.tensor([context + c for c in candidates], dtype=torch.long, device=device)
    x, y = rows[:, :-1], rows[:, 1:].clone()
    y[:, :len(context) - 1] = -1 # only score the candidate tokens
    nll = model(x, y, loss_reduction='none').view(len(candidates), -1).sum(dim=1)
    return nll.argmin().item()


def main():
    parser = argparse.ArgumentParser(description="Passkey retrieval eval for base models")
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--seq-len', type=int, default=2048, help='total tokens per example (BOS + filler + needle + query + answer)')
    parser.add_argument('--distances', type=str, default='64,128,256,384,512,768,1024,1536,1920', help='comma-separated needle-to-question distances in tokens')
    parser.add_argument('--num-examples', type=int, default=200, help='examples per distance')
    parser.add_argument('--num-candidates', type=int, default=10, help='true passkey + distractors (chance = 1/num_candidates)')
    parser.add_argument('--widen-windows', action='store_true', help="set the model's sliding windows as if it was trained at --seq-len")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    trained_len = model.config.sequence_len
    assert args.seq_len <= model.cos.size(1), f"--seq-len {args.seq_len} exceeds the rotary cache ({model.cos.size(1)})"
    if args.widen_windows:
        model.config.sequence_len = args.seq_len
        model.window_sizes = model._compute_window_sizes(model.config)
    print0(f"Model step {meta['step']} | trained at seq len {trained_len} | eval seq len {args.seq_len}")
    print0(f"Attention windows (left): {[w for w, _ in model.window_sizes]}")

    # same filler and examples on every rank, then each rank scores every world_size-th example
    filler = load_filler(tokenizer, num_tokens=max(1_000_000, 10 * args.seq_len))
    examples = [build_example(random.Random(f"{args.seed}-{i}"), tokenizer, filler, args.seq_len, args.num_candidates)
                for i in range(args.num_examples)]
    bos = tokenizer.get_bos_token_id()
    max_distance = len(examples[0]["filler"]) - 64 # keep some filler before the needle
    distances = [int(d) for d in args.distances.split(',')]
    for d in [d for d in distances if d > max_distance]:
        print0(f"Skipping distance {d}: does not fit in --seq-len {args.seq_len}")
    distances = [d for d in distances if d <= max_distance]

    correct = torch.zeros(len(distances), dtype=torch.long, device=device)
    for di, distance in enumerate(distances):
        for ex in examples[ddp_rank::ddp_world_size]:
            split = len(ex["filler"]) - distance
            context = [bos] + ex["filler"][:split] + ex["needle"] + ex["filler"][split:] + ex["query"]
            correct[di] += predict(model, context, ex["candidates"], device) == ex["answer_idx"]
    if ddp:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    # report accuracy with a 95% normal-approximation confidence interval
    n = args.num_examples
    rows = []
    print0(f"\n{'distance':>8} | {'accuracy':>8} | 95% CI (chance = {1 / args.num_candidates:.2f}, n = {n})")
    for distance, c in zip(distances, correct.tolist()):
        acc = c / n
        ci = 1.96 * math.sqrt(acc * (1 - acc) / n)
        rows.append((distance, acc, ci))
        print0(f"{distance:>8} | {acc:>8.3f} | ±{ci:.3f}")

    if ddp_rank == 0:
        suffix = "_widened" if args.widen_windows else ""
        output_path = os.path.join(get_base_dir(), "passkey_eval", f"{args.model_tag or 'base'}_step{meta['step']:06d}{suffix}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("distance,accuracy,ci95,num_examples,num_candidates,trained_seq_len,eval_seq_len,widened\n")
            for distance, acc, ci in rows:
                f.write(f"{distance},{acc:.6f},{ci:.6f},{n},{args.num_candidates},{trained_len},{args.seq_len},{args.widen_windows}\n")
        print0(f"\nResults written to: {output_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
