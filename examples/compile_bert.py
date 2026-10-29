"""Any nn.Module on the Neural Engine: BERT-base from transformers, eager CPU vs ANE.

    python examples/compile_bert.py

The first call per input shape builds a Core AI program (seconds); later runs load it
from the disk cache (~/.cache/torch_ane).
"""
import time

import torch
import torch.nn as nn
from transformers import AutoModel

import torch_ane


class Encoder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, ids):
        return self.bert(input_ids=ids).last_hidden_state


def bench(fn, x, n=20):
    fn(x)
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t) / n * 1e3


def main():
    model = Encoder(AutoModel.from_pretrained("bert-base-uncased")).eval()
    ids = torch.randint(1000, 30000, (8, 128))
    ane = torch_ane.compile(model, example_inputs=(ids,))
    with torch.no_grad():
        ref = model(ids)
        out = ane(ids)
        print(f"relative error vs eager: {((out - ref).norm() / ref.norm()).item():.2e}")
        print(f"eager CPU: {bench(model, ids):.1f} ms   ANE: {bench(ane, ids):.1f} ms")


if __name__ == "__main__":
    main()
