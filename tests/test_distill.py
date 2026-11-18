"""
Tests for the on-policy distillation helpers (nanochat/distill.py) used by scripts/chat_rl.py.
Runs on CPU with a tiny random GPT.

Run: python -m pytest tests/test_distill.py -v
"""
import random

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.distill import completion_to_parts, pick_solutions, generated_token_logprobs, align_to_targets


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    # init_weights zeroes the output projections; randomize everything so log-probs depend on the context
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    return model.eval()


def student_logprobs(model, sequences, prefix_length, pad_id):
    """ What chat_rl.py computes for the student: log-probs at target positions, 0 where ignored. """
    max_len = max(len(s) for s in sequences)
    ids = torch.tensor([s + [pad_id] * (max_len - len(s)) for s in sequences])
    targets = ids[:, 1:].clone()
    targets[:, :prefix_length - 1] = -1
    for i, s in enumerate(sequences):
        targets[i, len(s) - 1:] = -1
    with torch.no_grad():
        logp = -model(ids[:, :-1], targets, loss_reduction='none').view_as(targets)
    return logp, targets


def test_completion_to_parts():
    completion = ("Weng earns 12/60 = <|python_start|>12/60<|python_end|><|output_start|>0.2<|output_end|>0.2 per minute.\n"
                  "#### 10<|assistant_end|>")
    assert completion_to_parts(completion) == [
        {"type": "text", "text": "Weng earns 12/60 = "},
        {"type": "python", "text": "12/60"},
        {"type": "python_output", "text": "0.2"},
        {"type": "text", "text": "0.2 per minute.\n#### 10"},
    ]
    # a call that was cut off before the engine inserted its output
    assert completion_to_parts("x = <|python_start|>3*4<|python_end|>") == [
        {"type": "text", "text": "x = "}, {"type": "python", "text": "3*4"}]


def test_pick_solutions():
    rng = random.Random(0)
    assert pick_solutions([0, 0, 0], rng) == [None, None, None]
    picks = pick_solutions([0, 1, 0, 1], rng)
    assert picks[1] == 1 and picks[3] == 3  # correct samples see themselves
    assert picks[0] in (1, 3) and picks[2] in (1, 3)  # wrong samples see a correct sibling


def test_same_teacher_same_prompt_gives_zero_advantage():
    """ A teacher identical to the student, prompted the same way, must reproduce the student's log-probs
    at exactly the student's target positions, so the distillation advantage is zero everywhere. """
    model = tiny_model()
    prefix = [1, 5, 9, 2]
    generations = [[7, 8, 9, 10, 11], [12, 13], [14, 15, 16]]
    pad = 3
    sequences = [prefix + g for g in generations]
    logp, targets = student_logprobs(model, sequences, len(prefix), pad)
    teacher = align_to_targets(generated_token_logprobs(model, [prefix] * 3, generations, pad, 2, "cpu"),
                               len(prefix), targets.size(1), "cpu")
    valid = targets >= 0
    assert torch.allclose(teacher[valid], logp[valid], atol=1e-5)
    assert torch.all(teacher[~valid] == 0)


def test_longer_teacher_prefix_is_aligned():
    """ With a different (longer) teacher prompt, each generated token's teacher log-prob must still land on the
    student's target position for that token. Checked against a direct forward pass on the teacher's sequence. """
    model = tiny_model(1)
    student_prefix, teacher_prefix = [1, 5, 9, 2], [1, 40, 41, 42, 43, 44, 5, 9, 2]
    gen = [20, 21, 22, 23]
    aligned = align_to_targets(generated_token_logprobs(model, [teacher_prefix], [gen], 3, 8, "cpu"),
                               len(student_prefix), len(student_prefix) + len(gen) - 1, "cpu")
    seq = torch.tensor([teacher_prefix + gen])
    with torch.no_grad():
        full = torch.log_softmax(model(seq[:, :-1]).float(), dim=-1)[0]
    for k, token in enumerate(gen):
        expected = full[len(teacher_prefix) - 1 + k, token]
        assert torch.allclose(aligned[0, len(student_prefix) - 1 + k], expected, atol=1e-4)
