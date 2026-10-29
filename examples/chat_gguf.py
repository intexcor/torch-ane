"""Chat with a large Llama-style GGUF model (e.g. Devstral Small 24B Q4_K_M) on the Neural Engine.

    python examples/chat_gguf.py path/to/model.gguf path/to/hf_config_dir

hf_config_dir holds the original model's config.json and tokenizer files. The first run
builds the model in 4-layer chunks with 4-bit palettes (~35 min for 24B, ~13 GB of cache).

The model lives in a separate process: the Python wrapper of Core AI aborts the process
after 16384 allocated buffers (apple/coreai-torch#75), so the engine is restarted ahead of
time and the conversation is replayed. Commands: /reset, /exit.
"""
import argparse
import atexit
import multiprocessing as mp
import time
import warnings

import numpy as np


def worker(conn, gguf, config):
    warnings.filterwarnings("ignore")
    from torch_ane import ChunkedCausalLM
    lm = ChunkedCausalLM.from_gguf(gguf, config, log=lambda s: conn.send(("log", s)))
    conn.send(("ready", lm.max_len, lm.steps_left()))
    idle_since = time.monotonic()
    while True:
        # ANE unloads programs after ~10 s idle and the next prefill is ~3x slower: keep it warm
        # for a while (each wake-up also spends the buffer budget, hence the 3 min cap)
        if not conn.poll(6):
            if time.monotonic() - idle_since < 180 and lm.steps_left() > 100:
                lm.keep_warm()
            continue
        cmd, arg = conn.recv()
        idle_since = time.monotonic()
        logits = None
        if cmd == "prefill":  # reuse the cached common prefix
            common = 0
            for a, b in zip(lm._hist[:lm.pos], arg):
                if a != b:
                    break
                common += 1
            lm.pos = min(common, len(arg) - 1, lm.pos)
            lm._hist = lm._hist[:lm.pos]
            logits = lm.feed(arg[lm.pos:])
        elif cmd == "step":
            logits = lm.feed([arg])
        elif cmd == "reset":
            lm.reset()
        conn.send(("ok", logits, lm.steps_left()))


class Engine:
    def __init__(self, gguf, config):
        self.args, self.ctx, self.hist = (gguf, config), mp.get_context("spawn"), []
        self._start()

    def _start(self):
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=worker, args=(child, *self.args))
        self.proc.start()
        atexit.register(lambda p=self.proc: p.is_alive() and p.kill())
        while True:
            msg = self._recv()
            if msg[0] != "log":
                _, self.max_len, self.left = msg
                return
            print(msg[1], flush=True)

    def _recv(self):
        while not self.conn.poll(1):
            if not self.proc.is_alive():
                raise RuntimeError(f"engine process died (exit code {self.proc.exitcode})")
        return self.conn.recv()

    def _call(self, cmd, arg=None):
        self.conn.send((cmd, arg))
        _, logits, self.left = self._recv()
        return logits

    def _ensure(self, need):
        if self.left > need:
            return
        print("\n[restarting the ANE engine: Core AI buffer limit]", flush=True)
        self.proc.kill()
        self.proc.join()
        self._start()
        if self.hist:
            self._call("prefill", self.hist)

    def reset(self):
        self.hist = []
        self._call("reset")

    def prefill(self, ids):
        self._ensure(len(ids) // 8 + 64)
        self.hist = list(ids)
        return self._call("prefill", self.hist)

    def step(self, tok):
        self._ensure(4)
        self.hist.append(tok)
        return self._call("step", tok)


def sample(logits, temperature, top_p, rng, top_k=64):
    if temperature <= 0:
        return int(logits.argmax())
    idx = np.argpartition(-logits, top_k)[:top_k]
    p = np.exp((logits[idx] - logits[idx].max()) / temperature)
    p /= p.sum()
    order = np.argsort(-p)
    keep = order[:int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
    return int(idx[rng.choice(keep, p=p[keep] / p[keep].sum())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("config", help="directory with config.json and tokenizer of the original model")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.config)
    eng = Engine(a.gguf, a.config)
    rng, msgs = np.random.default_rng(), []
    print("ready. /reset — new conversation, /exit — quit\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/exit":
            break
        if user == "/reset":
            msgs = []
            eng.reset()
            continue
        if not user:
            continue
        msgs.append({"role": "user", "content": user})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)
        while len(ids) + 64 > eng.max_len and len(msgs) > 1:  # forget the oldest turns
            msgs = msgs[2:]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)
        t = time.perf_counter()
        logits = eng.prefill(ids)
        ttft = time.perf_counter() - t
        out, shown, t = [], "", time.perf_counter()
        print("model> ", end="", flush=True)
        while len(out) < a.max_tokens and len(ids) + len(out) < eng.max_len - 1:
            nxt = sample(logits, a.temperature, a.top_p, rng)
            if nxt == tok.eos_token_id:
                break
            out.append(nxt)
            text = tok.decode(out, skip_special_tokens=True)
            if not text.endswith("�"):
                print(text[len(shown):], end="", flush=True)
                shown = text
            logits = eng.step(nxt)
        dt = time.perf_counter() - t
        print(f"\n[{len(ids)} context tokens, first token {ttft:.1f} s, {len(out) / max(dt, 1e-9):.1f} tok/s]\n")
        msgs.append({"role": "assistant", "content": tok.decode(out, skip_special_tokens=True)})


if __name__ == "__main__":
    main()
