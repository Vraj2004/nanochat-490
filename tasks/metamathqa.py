"""
MetaMathQA: GSM8K and MATH train problems rewritten by GPT-3.5 (rephrased, answer-augmented,
self-verification and backward reasoning versions), with step-by-step solutions.
https://huggingface.co/datasets/meta-math/MetaMathQA

We only use the GSM8K-derived rows (type GSM_*, 240K of the 395K). Their responses end in
"#### N" followed by "The answer is: N"; we drop the last line so they end like GSM8K.
"""

import pyarrow.compute as pc
from tasks.common import Task, load_hub_dataset, HubDataset

class MetaMathQA(Task):
    """ MetaMathQA, GSM8K-derived rows only. 240K rows, train split only. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        table = load_hub_dataset("meta-math/MetaMathQA", split="train").table
        table = table.filter(pc.starts_with(table["type"], "GSM_"))
        self.ds = HubDataset(table).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        response = row["response"].split("\nThe answer is:")[0]
        messages = [
            {"role": "user", "content": row["query"]},
            {"role": "assistant", "content": response},
        ]
        return {"messages": messages}
