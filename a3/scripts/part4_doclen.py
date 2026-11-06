"""Measure ClimbMix document lengths in tokens, for the A3 Part 4 choice of a 512 sequence length.

Tokenizes one training shard with the nanochat tokenizer (counting the BOS token) and reports how many
documents, and how many tokens, fit within each candidate sequence length. Output is in
a3/results/part4/doclen_stats.txt.

Usage (from the repo root), after copying a shard and the tokenizer from the Modal volume:

    modal volume get nanochat-data base_data_climbmix/shard_00000.parquet /tmp/doclen/
    modal volume get nanochat-data tokenizer/tokenizer.pkl /tmp/doclen/
    python a3/scripts/part4_doclen.py /tmp/doclen
"""
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

d = sys.argv[1]
with open(f"{d}/tokenizer.pkl", "rb") as f:
    enc = pickle.load(f)
texts = pq.read_table(f"{d}/shard_00000.parquet", columns=["text"]).column("text").to_pylist()
n = np.array([len(t) + 1 for t in enc.encode_ordinary_batch(texts, num_threads=8)])  # +1 for BOS
print(f"docs {len(n):,}  tokens {n.sum():,}  median {np.median(n):.0f}  mean {n.mean():.0f}")
for p in (25, 50, 75, 90, 95):
    print(f"p{p}: {np.percentile(n, p):.0f}")
for T in (128, 256, 512, 1024, 2048):
    print(f"T={T}: docs fit {np.mean(n <= T):.1%}  tokens in docs that fit {n[n <= T].sum() / n.sum():.1%}  "
          f"tokens within first T of their doc {np.minimum(n, T).sum() / n.sum():.1%}")
