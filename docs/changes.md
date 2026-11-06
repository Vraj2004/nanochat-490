# CSC490 A3 — Proposed Changes

Oct 30, 2026 · @Vraj

We propose three architecture changes taken from ModernBERT's update of BERT. The July 2026 nanochat code has none of them:

1. **GeGLU feed-forward** in place of ReLU².
2. **Partial RoPE**: rotate only part of each attention head.
3. **Separate RoPE θ for local and global layers.**

We recommend ablating 1 and 2 in Part 2. Both are small edits to `gpt.py` and add no training cost. Change 3 is best tested in Part 4, because ModernBERT saw no gain from it during normal pretraining; its payoff is at context extension.

## The two papers

BERT is the starting point; ModernBERT is the playbook. Our changes come from ModernBERT, which tested each one with ablations (its Appendix D).

| Paper | What it is | What it changed | Use for us |
| --- | --- | --- | --- |
| [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018) | 12- and 24-layer bidirectional encoder | Masked-LM and next-sentence pretraining; learned absolute and segment position embeddings; GELU MLP with biases | Mostly a baseline. Its objectives need bidirectional attention and don't fit a next-token decoder. Its model-size ablation (Table 6: deeper and wider models do better) is useful background |
| [ModernBERT](https://aclanthology.org/2025.acl-long.127/) (Warner et al., ACL 2025) | BERT rebuilt with LLM-era parts; 149M and 395M params, 8,192 context, 2T tokens | No biases; RoPE; pre-norm; GeGLU; alternating global/local attention with separate RoPE θ; unpadding; Flash Attention; hardware-aware sizes | Source of our three changes |

Of ModernBERT's changes, July nanochat already has no biases, RoPE, pre-norm, alternating local/global attention (`SSSL`), Flash Attention, a softcap and a vocabulary padded to a multiple of 64. What's left is the GLU feed-forward and the finer RoPE choices.

One caveat for the writeup: ModernBERT's ablations used encoders of 150M parameters or less, trained on 8–20B tokens. Effects may differ in a causal decoder.

## Proposed changes

This is the Part 1 table. Section numbers point to the ModernBERT paper.

| # | Change | Source | Motivation | Technical detail in nanochat | Expected impact |
| --- | --- | --- | --- | --- | --- |
| 1 | **GeGLU feed-forward** instead of ReLU² | ModernBERT §2.1.1, building on Shazeer (2020), *GLU Variants Improve Transformer*. BERT used a plain GELU MLP | Gated MLPs gave consistent gains across recent work. ModernBERT adopted GeGLU and found it about equal to SwiGLU (App. D.1) | In `MLP`: `c_fc` maps d → 2h, split into gate and up, output = `c_proj`(GELU(gate) ⊙ up). Set h = 8d/3 so parameters match ReLU²'s 4d hidden: 1,024 at d6 (384 wide), about 3,456 at d20. Keep `c_proj` zero-init | Small loss improvement at equal parameters, about 1.5× the MLP activation memory. ReLU² is a strong baseline, so a null result is possible and still worth reporting |
| 2 | **Partial RoPE**: rotate 50% of each head | ModernBERT App. D.1; GPT-NeoX (Black et al., 2022) rotates 25% | Dims that aren't rotated can match on content regardless of distance. ModernBERT found 50% and 75% slightly better than 100% but kept 100% to be safe at full scale | In `apply_rotary_emb`, rotate only the first 64 of the 128 head dims and pass the rest through. Build cos/sin for 64 dims. No new parameters | Slightly lower loss or no change at pico scale. Might also help Part 4, since unrotated dims carry no position signal to break when context grows |
| 3 | **Separate RoPE θ for local and global layers** | ModernBERT §2.1.2 and §2.2.2: local 10,000, global 160,000 | Short-window layers only see nearby tokens, so slow rotations are wasted on them. Global layers need slow rotations to tell far positions apart. Extending context then only means raising global θ | Build two cos/sin tables. `S` layers use θ = 10,000; `L` layers use 100,000 now, raised for extension. Pick the table per layer using `window_sizes` in `GPT.forward` | Little change at a fixed context: ModernBERT's ablations saw no difference across θ values (App. D.1). The payoff comes when extending 512 → 2048 in Part 4 |

Backups if the team wants a different third pick: **intra-document attention masking** (ModernBERT App. A.2). July nanochat starts each row with BOS but still lets packed documents attend to each other. Or a **deeper, narrower shape** at equal parameters (ModernBERT §2.1.3; BERT Table 6).

## Change 1 explained: GeGLU

GeGLU replaces the MLP's single "yes/no" score with two scores multiplied together: one for relevance, one for value.

&#91;embedded content: change 1 · today's MLP vs GeGLU\]

### Background: what the MLP is

- The model turns each word (token) into a list of 1280 numbers: its notes about that word. Every layer improves those notes so the last word's notes can predict the next word.
- Each of the 20 layers has two steps. **Attention** lets a word look at earlier words and copy useful information ("talk to the others"). The **MLP** then processes each word on its own ("think alone").
- MLP stands for *multi-layer perceptron*, an old name for a small plain neural network. In our model it is a mini network run on every word inside every layer.

### How today's MLP works (ReLU²)

1. **Ask 5,120 questions.** `c_fc` (1280 → 5120) holds 5,120 learned patterns. Each one scores how well this word's notes match it.
2. **Keep the yes answers.** ReLU² sets negative scores to 0 and squares positive ones, so a strong match counts far more than a weak one.
3. **Write back.** `c_proj` (5120 → 1280) adds a learned edit for each yes, scaled by its strength, to the word's notes.

Limitation: one number per question has to say both *whether* it applies and *what* to add, and it can never be negative.

### What GeGLU changes

- **Relevance** (`c_gate` then GELU): does this question apply right now? It comes out near 0 for no and large for yes. GELU is a smooth version of ReLU.
- **Value** (`c_fc`, no activation): what should it say? This one can be positive or negative.
- **Multiply them.** Relevance × value: off when irrelevant, otherwise a signed push in either direction.
- Analogy: today's MLP is a switch that can only turn things up. GeGLU is a switch plus a dial.

Toy example, one hidden unit:

| Token | Relevance (GELU) | Value | Output |
| --- | --- | --- | --- |
| "2026" in a date | 2.0 → 1.95 | +1.2 | +2.3, pushes toward year-like words |
| "2026" in a phone number | 2.0 → 1.95 | −0.8 | −1.6, pushes away |
| "the" | −2.0 → −0.05 | +1.2 | −0.05, effectively off |

### Keeping the comparison fair

- GeGLU has 3 matrices (gate, up, down) instead of 2. At the same hidden size it would be 50% bigger, and a bigger model winning proves nothing.
- ReLU² MLP: 2 × d × 4d = 8d² parameters. GeGLU: 3 × d × h. Setting them equal gives **h = 8d/3**.
- d6 picochat (d = 384): h = 1,024, so the parameter count is exactly equal (1,179,648 per MLP). FLOPs match too, because nanochat counts FLOPs from matmul weights, so MFU stays comparable.
- Gate and up stay as **separate matrices**. Muon orthogonalizes each matrix as a whole, and fusing them into one 2h × d matrix would couple their updates.

### Code sketch (behind a flag; baseline untouched)

```python
# GPTConfig
mlp_type: str = "relu2"   # "relu2" | "geglu"

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        self.geglu = config.mlp_type == "geglu"
        h = 128 * round(8 * d / 3 / 128) if self.geglu else 4 * d   # param-matched
        self.c_fc = Linear(d, h, bias=False)                          # value (or the only in-proj)
        self.c_gate = Linear(d, h, bias=False) if self.geglu else None
        self.c_proj = Linear(h, d, bias=False)

    def forward(self, x):
        if self.geglu:
            x = F.gelu(self.c_gate(x)) * self.c_fc(x)
        else:
            x = F.relu(self.c_fc(x)).square()
        return self.c_proj(x)
```

Also needed: init `c_gate` like `c_fc` in `init_weights`; add `--mlp-type` to `base_train.py` and pass it to `GPTConfig`; default `mlp_type` to `"relu2"` in `checkpoint_manager.py` so old checkpoints still load.

### What to expect

- **For:** Llama, Gemma and ModernBERT all use gated MLPs. Shazeer (2020) found GLU variants learn better at equal size. ModernBERT found GeGLU ≈ SwiGLU.
- **Against:** Karpathy chose ReLU² because it won modded-nanogpt speedruns. A small gain or a tie are both plausible, and both are reportable.
- **How we measure:** train baseline and GeGLU with everything else identical and compare validation bits-per-byte (lower is better). A second baseline seed shows how much of a difference is just noise.

## Change 2 explained: partial RoPE

Partial RoPE adds position to only half of each attention head. The other half compares words purely by meaning, whatever the distance between them.

&#91;embedded content: change 2 · full vs partial RoPE\]

### Background: why the model needs position

- Attention compares words with each other. Without position information, "dog bites man" and "man bites dog" contain the same words and look identical.
- Each word makes a **query** ("what am I looking for?") and a **key** ("what do I contain?"): 128 numbers per head each. Their match score is a dot product, which measures how well two arrows point the same way.

### How RoPE works today

- RoPE (rotary position embedding) splits each head's 128 numbers into **64 pairs**. Each pair is a little arrow on a dial.
- Each arrow is **rotated by an angle = position × speed**. Word 5 is turned 5 steps, word 9 is turned 9 steps.
- When two words are compared, their rotations cancel except for the **difference**. So the score depends on how far apart the words are (9 − 5 = 4), not where they sit in the text: "4 words back" means the same thing everywhere.
- Each pair spins at a **different speed**, like the second, minute and hour hands of a clock. Fast dials tell neighbours apart; slow dials measure long distances. In July nanochat the fastest turns about 1 radian per token and the slowest barely moves (θ = 100,000).
- RoPE adds **no parameters**. It is pure rotation.

### The problem with rotating everything

- Every one of the 64 pairs is rotated, so every match mixes **what** a word is with **where** it is.
- Some heads want pure content: "find where *Paris* was mentioned 800 words ago." At that distance the fast dials have spun many times and point in effectively random directions, adding noise to the match.

### What partial RoPE changes

- Rotate only the first **50%** of each head (32 of 64 pairs). Leave the other 32 pairs **fixed**.
- Spinning pairs still encode distance. Fixed pairs give a clean "same meaning?" channel that works at any distance.
- Analogy: today every compass on the table spins as you walk, so you can't use any of them to check true north. Partial RoPE keeps half of them still.

### Evidence

- **ModernBERT** (App. D.1) tested rotating 50%, 75% and 100% of each head. Lower percentages gave slightly better results, but the gap was small, so they kept 100% to be safe at full scale. Report this honestly.
- **GPT-NeoX-20B and GPT-J** (as cited in ModernBERT) rotate only 25%.

### Keeping the comparison fair

- Zero new parameters and the same FLOPs. Only the rotation step changes, which makes this the cleanest ablation of the three.
- Trade-off: fewer spinning pairs means coarser position information. 50% still leaves 32 pairs from fast to slow.

### Code sketch (behind a flag)

```python
# GPTConfig
rope_frac: float = 1.0   # 1.0 = today; 0.5 = partial RoPE

def apply_rotary_emb(x, cos, sin):
    d_rot = cos.shape[-1] * 2                 # dims that get rotated
    x_rot, x_pass = x[..., :d_rot], x[..., d_rot:]
    d = d_rot // 2
    x1, x2 = x_rot[..., :d], x_rot[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2, x_pass], 3)    # unrotated half passes through

# in GPT.__init__ and GPT.init_weights:
rotary_dim = int(head_dim * config.rope_frac)   # 128 * 0.5 = 64
cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, rotary_dim)
```

Also needed: `--rope-frac` in `base_train.py`, passed to `GPTConfig`; default `rope_frac` to `1.0` in `checkpoint_manager.py`. QK-norm still runs after RoPE, unchanged.

### What to expect

- At picochat scale: a small loss improvement or no change.
- Tie-in to Part 4: the fixed pairs never see angles they weren't trained on, so they shouldn't break when the context grows from 512 to 2048. That gives us a second place to test this change.

## Change 3 explained: separate RoPE θ for local and global layers

Give short-sighted layers fast dials and far-sighted layers slow dials, instead of one setting for all.

- **What θ does.** θ sets how slow the slowest RoPE dials turn (see change 2). A bigger θ lets the slow dials measure longer distances before they wrap around.
- **Two kinds of layers already exist.** July nanochat uses the pattern `SSSL`. Short (S) layers only look back a quarter of the context: 512 tokens at 2048. Long (L) layers see everything. Today every layer uses the same θ = 100,000.
- **The mismatch.** S layers never look past their window, so dials built for very long distances are wasted on them. L layers need those slow dials to tell far-apart words apart.
- **The change.** S layers use θ = 10,000 (more dials useful up close). L layers keep 100,000, and only theirs is raised when we extend the context.
- **Analogy.** Someone reading one sentence needs a ruler; someone flipping through the whole book needs a tape measure. Today everyone gets the tape measure.
- **Evidence.** ModernBERT uses exactly this: local 10,000, global 160,000. Its ablations found no difference during normal pretraining (App. D.1); the benefit shows up when stretching the context.
- **Code.** Build two cos/sin tables in `GPT.__init__` and `init_weights`, then pick the table per layer in `forward` based on `window_sizes`. About 10 lines, no new parameters, same FLOPs.
- **What to expect.** No change in Part 2; test it in Part 4. Note that the short window scales with context length (128 tokens at 512, 512 at 2048), so the S layers also see longer distances after extension.

## Fit check against July nanochat

All three are new to the July code (`92d63d4`), and each is a contained edit to `nanochat/gpt.py` behind a config flag.

| Change | In July code? | What to edit | Extra params | Cost of one Part 2 run | Ablate where |
| --- | --- | --- | --- | --- | --- |
| GeGLU MLP | No: ReLU², 4d hidden | `MLP`, `init_weights` | 0 when h = 8d/3 | Same as baseline | Part 2 |
| Partial RoPE | No: rotates all 128 dims | `apply_rotary_emb`, `_precompute_rotary_embeddings` | 0 | Same as baseline | Part 2 |
| Local/global θ | No: one θ = 100,000 for every layer | `__init__` (second table), `forward` (per-layer pick) | 0 | Same as baseline | Part 4 |

One trap: Part 4 runs at 512 context with `--window-pattern=L`, so every layer is global and change 3 does nothing there. To test it, keep the `SSSL` pattern. At 512 that makes the short layers 128 tokens wide, which matches ModernBERT's window.

## Sources

- Devlin, Chang, Lee, Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). arXiv 1810.04805, 2018.
- Warner et al. [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT)](https://aclanthology.org/2025.acl-long.127/). ACL 2025, pages 2526–2547.
- Cited through ModernBERT's bibliography, not read directly: Shazeer, *GLU Variants Improve Transformer* (2020); Black et al., *GPT-NeoX-20B* (2022).
- nanochat source at commit `92d63d4` (Jul 3, 2026): `nanochat/gpt.py`, `nanochat/dataloader.py`, `scripts/base_train.py`.
