"""A small LLM entirely on the Neural Engine, KV cache kept as Core AI state.

    python examples/generate.py "Explain the Neural Engine in two sentences."
    python examples/generate.py --model Qwen/Qwen2.5-1.5B-Instruct "..."

First run builds the program (a few minutes for 0.5B), then it is loaded from cache.
"""
import argparse
import time

from transformers import AutoTokenizer

import torch_ane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    lm = torch_ane.CausalLM.from_pretrained(a.model)
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                  return_dict=False)
    lm.reset()
    t = time.perf_counter()
    logits = lm.feed(ids)  # prefill
    ttft = time.perf_counter() - t
    out, shown = [], ""
    t = time.perf_counter()
    while len(out) < a.max_tokens:
        nxt = int(logits.argmax())
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        text = tok.decode(out, skip_special_tokens=True)
        if not text.endswith("�"):  # don't print half of a multi-byte character
            print(text[len(shown):], end="", flush=True)
            shown = text
        logits = lm.feed([nxt])  # one decode step
    dt = time.perf_counter() - t
    print(f"\n\n[prompt {len(ids)} tokens, first token {ttft * 1e3:.0f} ms, "
          f"{len(out)} tokens at {len(out) / max(dt, 1e-9):.1f} tok/s]")


if __name__ == "__main__":
    main()
