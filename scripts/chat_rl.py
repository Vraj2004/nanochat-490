"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

On-policy distillation (A4 Part 5): with --distill=teacher or --distill=self, a teacher scores every
sampled token and the per-token advantage becomes log p_teacher - log p_student, a one-sample estimate
of the negative reverse KL (Thinking Machines Lab, 2025). --distill=self uses SDPO's self-teacher
(Hubotter et al., 2026): a frozen copy of the starting checkpoint, shown a correct sibling sample.
--reward-weight mixes in the reward advantage: lambda * (r - mu) + (1 - lambda) * distill.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import os
import json
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from nanochat.distill import self_teacher_conversation, pick_solutions, generated_token_logprobs, align_to_targets, make_rng
from tasks.gsm8k import GSM8K, GSM8KShaped

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-tag", type=str, default=None, help="tag to save checkpoints and rollouts under (default = --model-tag)")
# Environment (training reward)
parser.add_argument("--env", type=str, default="gsm8k", choices=["gsm8k", "gsm8k_finish", "gsm8k_partial", "gsm8k_finish_partial"], help="gsm8k = correctness only; _finish / _partial add the shaped bonuses in tasks/gsm8k.py")
# On-policy distillation (Part 5)
parser.add_argument("--distill", type=str, default="none", choices=["none", "teacher", "self"], help="none = RL only; teacher = score samples with --teacher-tag; self = SDPO self-teacher (frozen start checkpoint shown a correct sibling)")
parser.add_argument("--teacher-source", type=str, default="sft", choices=["sft", "rl"], help="checkpoint directory of the teacher")
parser.add_argument("--teacher-tag", type=str, default=None, help="teacher model tag (default for --distill=self: --model-tag)")
parser.add_argument("--teacher-step", type=int, default=None, help="teacher model step (default = last)")
parser.add_argument("--reward-weight", type=float, default=None, help="lambda in lambda*(r - mu) + (1 - lambda)*distill advantage (default: 1 without distillation, 0 with it)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
parser.add_argument("--stop-at-step", type=int, default=-1, help="save and stop at this step, keeping the full run's LR schedule (-1 = run to the end)")
parser.add_argument("--train-end-token", type=int, default=-1, help="1 = train on the <|assistant_end|> that ended each sample, which the sampler drops (default: 1 with distillation, 0 without, as in the Part 3-4 RL runs)")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
if args.reward_weight is None:
    args.reward_weight = 1.0 if args.distill == "none" else 0.0
if args.train_end_token == -1:
    args.train_end_token = int(args.distill != "none")
assert args.distill != "none" or args.reward_weight == 1.0, "--reward-weight < 1 needs a teacher (--distill)"
assert args.distill != "teacher" or args.teacher_tag, "--distill=teacher needs --teacher-tag"
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # for sampling rollouts

# The teacher for on-policy distillation: never trained, only scores the student's samples
teacher = None
if args.distill != "none":
    teacher_tag = args.teacher_tag or args.model_tag
    teacher, teacher_tokenizer, _ = load_model(args.teacher_source, device, phase="eval", model_tag=teacher_tag, step=args.teacher_step)
    assert teacher_tokenizer.get_vocab_size() == tokenizer.get_vocab_size(), "teacher and student must share a tokenizer"
    teacher.requires_grad_(False)
    print0(f"Distillation: {args.distill}, teacher {args.teacher_source}/{teacher_tag}, reward weight {args.reward_weight}")

# Every rollout is logged (one JSON line per training example per rank) to analyze which problems the model gets right/wrong
output_dirname = args.output_tag or args.model_tag or f"d{model.config.n_layer}"
rollout_dir = os.path.join(get_base_dir(), "chatrl_rollouts", output_dirname)
os.makedirs(rollout_dir, exist_ok=True)
rollout_file = open(os.path.join(rollout_dir, f"rank{ddp_rank}.jsonl"), "w")

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8KShaped(subset="main", split="train", finish="finish" in args.env, partial="partial" in args.env)
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # used for padding (mask 0, not in the loss) and, with --train-end-token, as the trained stop token
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed, # must make sure to change the seed for each sampling step
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        generated_texts = []
        terms = [] # (correct, finish, matched intermediates) per sample, logged separately from the shaped reward
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            generated_texts.append(generated_text)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)
            terms.append(train_task.reward_terms(conversation, generated_text))
        correct, finish, matched = (list(x) for x in zip(*terms))

        # Engine.generate_batch drops the token that ended each sample, so by default the decision to stop is never
        # trained. With distillation that lets the student escape the (on average negative) per-token advantage by
        # ending early, so put <|assistant_end|> back, as a sampled token, for every sample that stopped by itself.
        if args.train_end_token:
            for seq, mask in zip(generated_token_sequences, masks):
                if len(seq) - prefix_length < args.max_new_tokens:
                    seq.append(assistant_end)
                    mask.append(1)

        # Teacher log-probs of every generated token (Part 5). distill_mask is 0 for samples with no teacher signal.
        max_length = max(len(seq) for seq in generated_token_sequences)
        generations = [seq[prefix_length:] for seq in generated_token_sequences]
        distill_mask = [1.0] * len(generations)
        teacher_logp = None
        if args.distill == "teacher":
            teacher_prefixes = [tokens] * len(generations)
        elif args.distill == "self":
            solutions = pick_solutions(correct, make_rng(step, example_idx))
            teacher_prefixes = [tokens if j is None else tokenizer.render_for_completion(self_teacher_conversation(conversation, generated_texts[j]))
                                for j in solutions]
            distill_mask = [0.0 if j is None else 1.0 for j in solutions]
        if teacher is not None:
            token_logprobs = generated_token_logprobs(teacher, teacher_prefixes, generations, assistant_end, args.device_batch_size, device)
            teacher_logp = align_to_targets(token_logprobs, prefix_length, max_length - 1, device)
        rollout_file.write(json.dumps({"step": step, "idx": example_idx, "rewards": rewards, "correct": correct,
                                       "finish": finish, "matched": matched, "completions": generated_texts}) + "\n")

        # Pad the sequences so that their lengths (in time) match
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        if teacher_logp is None: # no distillation: a zero teacher term that the reward weight of 1 ignores
            teacher_logp = torch.zeros_like(targets, dtype=torch.float)
        distill_mask = torch.tensor(distill_mask, dtype=torch.float, device=device)
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages, correct, teacher_logp, distill_mask

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= args.device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device) # pass@k for k=1..device_batch_size
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter) # collect all records
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    correct_list = [] # fraction of correct samples, = reward for --env=gsm8k
    sequence_lengths = []
    rkl_sum, rkl_tokens = 0.0, 0 # reverse KL estimate (student vs teacher) over distilled tokens, for logging
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, correct_all, teacher_logp_all, distill_mask_all = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            teacher_logp = teacher_logp_all[b0:b1]
            distill_mask = distill_mask_all[b0:b1].unsqueeze(-1)
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Per-token advantage: the reward advantage (the same for every token of a sample) mixed with the
            # distillation advantage log p_teacher - log p_student (Part 5). Both are constants for the gradient.
            distill_adv = (teacher_logp - logp.detach()) * distill_mask
            token_adv = args.reward_weight * advantages.unsqueeze(-1) + (1.0 - args.reward_weight) * distill_adv
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * token_adv).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            if teacher is not None:
                distilled = (targets >= 0) & (distill_mask > 0)
                rkl_sum += (logp.detach() - teacher_logp)[distilled].sum().item()
                rkl_tokens += distilled.sum().item()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # For logging
        rewards_list.append(rewards_all.mean().item())
        correct_list.append(sum(correct_all) / len(correct_all))
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_correct = sum(correct_list) / len(correct_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    rkl_stats = torch.tensor([rkl_sum, rkl_tokens], dtype=torch.float, device=device)
    if ddp: # aggregate across ranks
        dist.all_reduce(rkl_stats, op=dist.ReduceOp.SUM)
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_correct_tensor = torch.tensor(mean_correct, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_correct_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_correct = mean_correct_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    mean_rkl = (rkl_stats[0] / rkl_stats[1].clamp(min=1)).item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f} | Correct: {mean_correct:.4f} | Reverse KL: {mean_rkl:.4f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "correct": mean_correct,
        "sequence_length": mean_sequence_length,
        "reverse_kl": mean_rkl,
    })

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step (or the --stop-at-step step).
    is_last_step = step == num_steps - 1 or step == args.stop_at_step
    if master_process and ((step > 0 and step % args.save_every == 0) or is_last_step):
        base_dir = get_base_dir()
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")
    if step == args.stop_at_step:
        print0(f"Stopping at step {step} (--stop-at-step); the LR schedule was that of the full {num_steps}-step run")
        break

batch_iterator.close() # finish the rollout generator now rather than during interpreter shutdown
rollout_file.close()
wandb_run.finish() # wandb run finish
compute_cleanup()
