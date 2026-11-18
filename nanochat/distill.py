"""
Helpers for on-policy distillation in scripts/chat_rl.py (A4 Part 5).

The student samples completions as in RL, and a teacher scores every sampled token. The
per-token advantage is the negative of a one-sample estimate of the reverse KL,
log pi_teacher(x_t) - log pi_student(x_t), with no discounting (Thinking Machines Lab, 2025).

Two kinds of teacher are supported:
- an external teacher: another checkpoint with the same tokenizer, prompted exactly like the student
- a self-teacher (SDPO, Hubotter et al., 2026): a frozen copy of the student's starting checkpoint,
  shown a correct solution from the same group of samples as an earlier turn of the conversation
"""

import re
import random

import torch

TOOL_CALL_RE = re.compile(r"<\|python_start\|>(.*?)<\|python_end\|>(?:<\|output_start\|>(.*?)<\|output_end\|>)?", re.S)


def completion_to_parts(completion):
    """ Split a decoded completion into assistant message parts (text / python / python_output), so a sampled
    solution can be re-rendered with its calculator calls as real tool tokens. """
    completion = completion.replace("<|assistant_end|>", "")
    parts, pos = [], 0
    for m in TOOL_CALL_RE.finditer(completion):
        if m.start() > pos:
            parts.append({"type": "text", "text": completion[pos:m.start()]})
        parts.append({"type": "python", "text": m.group(1)})
        if m.group(2) is not None:
            parts.append({"type": "python_output", "text": m.group(2)})
        pos = m.end()
    if pos < len(completion):
        parts.append({"type": "text", "text": completion[pos:]})
    return parts


def self_teacher_conversation(conversation, solution):
    """ SDPO-style teacher context: the question, a correct solution as an earlier assistant turn (calculator calls
    kept as tool tokens, so the teacher does not penalize the tool itself), then a request to solve it again.
    The final assistant message is a placeholder that render_for_completion drops. """
    question = conversation["messages"][0]["content"]
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": completion_to_parts(solution)},
        {"role": "user", "content": "That is correct. Solve the same problem again."},
        conversation["messages"][-1],
    ]}


def pick_solutions(correct, rng):
    """ For each sample in a group, the index of the correct sample the self-teacher sees: the sample itself
    if it is correct (as in SDPO), otherwise a random correct sibling, or None if no sample in the group is
    correct (the teacher would then see the same prompt as the student, so there is nothing to distill). """
    winners = [i for i, c in enumerate(correct) if c]
    if not winners:
        return [None] * len(correct)
    return [i if c else rng.choice(winners) for i, c in enumerate(correct)]


@torch.no_grad()
def generated_token_logprobs(model, prefixes, generations, pad_id, batch_size, device):
    """
    Log-probability that `model` assigns to each generated token, conditioned on its own prefix.
    prefixes[i] and generations[i] are token lists; returns a list of 1-D float tensors, one per sample,
    with len(generations[i]) entries.
    """
    out = []
    for b0 in range(0, len(prefixes), batch_size):
        seqs = [p + g for p, g in zip(prefixes[b0:b0 + batch_size], generations[b0:b0 + batch_size])]
        max_len = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        targets = torch.full((len(seqs), max_len - 1), -1, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            targets[i, :len(s) - 1] = ids[i, 1:len(s)]
        logp = -model(ids[:, :-1], targets, loss_reduction='none').view_as(targets).float()
        for i, (p, g) in enumerate(zip(prefixes[b0:b0 + batch_size], generations[b0:b0 + batch_size])):
            # token g[k] sits at position len(p) + k and is predicted at position len(p) + k - 1
            out.append(logp[i, len(p) - 1:len(p) - 1 + len(g)])
    return out


def align_to_targets(token_logprobs, prefix_length, num_positions, device):
    """ Place per-sample generated-token log-probs at the student's target positions: the student's
    targets are ids[:, 1:], so generated token k (at index prefix_length + k) is target prefix_length + k - 1. """
    aligned = torch.zeros(len(token_logprobs), num_positions, dtype=torch.float, device=device)
    for i, lp in enumerate(token_logprobs):
        aligned[i, prefix_length - 1:prefix_length - 1 + len(lp)] = lp
    return aligned


def make_rng(step, example_idx):
    """ Deterministic RNG for choosing which correct sibling a self-teacher sees. """
    return random.Random(hash((step, example_idx)) & 0x7FFFFFFF)
