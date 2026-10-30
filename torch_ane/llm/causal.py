"""LLM on the Neural Engine with the KV cache inside the model, Apple's scheme (coreai-models,
iOS export): all of K in one Core AI state, all of V in another, a chunk is written with
coreai.slice_update at an offset taken from an input. States live between calls and
are passed without copying (call time barely grows with cache size), unlike a KV cache
passed as an input (~22 ms of copying per decode step for Qwen2.5-0.5B).

The earlier attempt (ANEChunkedPrefill: index_copy_ into a separate buffer per layer)
silently computed wrong results on ANE; with two shared buffers and slice_update it is correct.

As at Apple, two functions over the same states: prefill in chunks of PREFILL_Q
tokens and decode in windows of DECODE_Q (the window is aligned to a multiple of DECODE_Q,
the unfinished tail is recomputed; writing the same values to the same positions is harmless).
The transformers model is not rewritten: a custom cache layer (_StateLayer) writes K/V into
the buffers and hands the attention layer a slice over the whole context; the mask is a 4D input.
"""
import asyncio
import contextlib
import hashlib
import io
import warnings

import numpy as np
import torch
from transformers.cache_utils import Cache

from .. import passes
from ..runtime import CACHE_DIR, compress_program, model_digest, prune_cache, touch_cache
from ._kv import DECODE_Q, MASK_NEG, StateLayer, fresh_rope, register_lowering, window_mask

PREFILL_Q = 256
LADDER = (512, 1024, 2048, 4096, 8192)
# deliberately bilingual (Russian + English): calibration data, not docs
CALIB_TEXT = ("Объясни подробно, как устроен трансформер: внимание, MLP, нормализация. "
              "Explain how attention works, step by step, with a short Python example.")


class _StateStep(torch.nn.Module):
    """(emb [1, Q, hidden], pos [1, Q], mask [1, 1, Q, M], step [1], pick [1, Q])
    -> logits of one row [1, V] (the row is selected by the one-hot vector pick);
    the chunk's K/V are written into the key_cache/value_cache buffers [L, 1, kv, M, hd].

    As at Apple: embeddings come in as an input (table lookup happens outside), the graph has
    one copy of the lm_head weights; logits only for the needed position, not the whole window.
    Row selection is a multiplication by a one-hot in fp16 (ANE dislikes integer indexing)."""

    def __init__(self, lm, max_len):
        super().__init__()
        cfg = lm.config.get_text_config()
        self.lm, self.max_len = lm, max_len
        self.n_layers = cfg.num_hidden_layers
        kv = cfg.num_key_value_heads
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        shape = (self.n_layers, 1, kv, max_len, hd)
        self.register_buffer("key_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("value_cache", torch.zeros(shape), persistent=False)

    def forward(self, emb, pos, mask, step, pick):
        dt = self.lm.lm_head.weight.dtype
        cache = Cache(layers=[StateLayer(self, i, step) for i in range(self.n_layers)])
        o = self.lm.model(inputs_embeds=emb.to(dt), position_ids=pos, attention_mask=mask.to(dt),
                          past_key_values=cache, use_cache=True)
        if pick.shape[0] > 1:  # batch of streams: pick [P, Q] -> logits of P rows [P, V]
            return self.lm.lm_head(torch.matmul(pick.to(dt), o.last_hidden_state[0]))
        h = (pick.to(dt)[:, :, None] * o.last_hidden_state).sum(1)  # [1, hidden]
        return self.lm.lm_head(h)


class CausalLM:
    """A transformers causal LM entirely on ANE: prefill in chunks, decode in windows, KV cache as
    Core AI states. The first build takes minutes, after that the program comes from the cache.

        lm = CausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", batch=True)
        tokens = lm.generate(prompt_ids, max_new_tokens=64)
        outs = lm.generate_batch([ids1, ids2, ids3], max_new_tokens=64)   # up to 8 streams per step

    max_len: the largest context; compress: "lut4"/"lut4g32" (Apple palettes (LUT)) or compression
    of the finished program ("int8"...); composite: Core AI composite ops ("sdpa", "rope");
    calib_ids: tokens of real text for fp16 scale calibration; batch: build
    the functions for generate_batch; mlp_chunk: MLP in chunks sized for ANE's internal memory."""

    @classmethod
    def from_pretrained(cls, model_id, **kw):
        """Model and (for calibration) tokenizer from transformers by name or path."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        if "calib_ids" not in kw:
            try:
                tok = AutoTokenizer.from_pretrained(model_id)
                kw["calib_ids"] = tok(CALIB_TEXT)["input_ids"][:128]
            except Exception:  # noqa: BLE001 — no tokenizer: fixed tokens
                pass
        return cls(hf, **kw)

    def __init__(self, hf_lm, max_len=2048, device="ane", prefill_q=PREFILL_Q, compress=None,
                 ladder=LADDER, composite=(), calib_ids=None, batch=False, mlp_chunk=True):
        import copy

        from ..runtime import _device
        device = _device(device)
        self.lm = fresh_rope(copy.deepcopy(hf_lm).float()).eval()
        self._embed = self.lm.get_input_embeddings()
        # small MLP/V activations don't drown in fp16 subnormals on ANE (before swapping Linear for convs);
        # each projection pair gets its own scale, from calibration (a shared 2^8 overflowed fp16 on
        # massive activations: Qwen2.5 on ANE got stuck in loops, as did Devstral)
        vocab = self.lm.get_input_embeddings().num_embeddings
        cal = torch.tensor([list(calib_ids) if calib_ids is not None
                            else [1 % vocab] + [i % vocab for i in range(1000, 1095)]])
        k_for, _ = passes.calibrate_rescale(self.lm, lambda: self.lm(input_ids=cal).logits)
        passes.rescale_for_fp16(self.lm, k_for=k_for)
        # MLP in chunks of the hidden dimension sized for ANE's internal memory (passes.chunk_mlp)
        self.mlp_chunk = mlp_chunk and passes.chunk_mlp(self.lm)
        if compress and compress.startswith("lut"):  # "lut4" / "lut4g32": palettes (LUT) per Apple's recipe
            bits, _, g = compress[3:].partition("g")
            passes.swap_linear(self.lm.model)  # projections become 1x1 convs (lm_head stays Linear)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lm = passes.palettize_lut(self.lm, int(bits), int(g or 8))
        # prefill_q: prefill chunks are larger than the decode window: each chunk rereads all
        # the weights, small chunks are memory-bound rather than compute-bound
        self.max_len, self.device, self.prefill_q, self.compress = max_len, device, prefill_q, compress
        self.batch = batch  # "batch" functions: a decode window of tokens from different streams (generate_batch)
        # context ladder (as at Apple): attention on ANE costs in proportion to
        # cache length, a short conversation runs on functions with a small cache
        self.ladder = tuple(b for b in ladder if b < max_len and b >= prefill_q) + (max_len,)
        passes.swap_rmsnorm(self.lm)
        # Core AI composite SDPA/RoPE (composite=("sdpa", "rope")), as in Apple's macOS export
        self.composite = tuple(composite)
        self._extra_specs = passes.use_composite_ops(self.lm, self.composite) if self.composite else []
        self.loop = asyncio.new_event_loop()
        import coreai.runtime as rt
        self._rt = rt
        self._load(self._build())
        self.reset()

    def _entries(self):
        kinds = ("prefill", "decode") + (("batch",) if self.batch else ())
        return [(kind, M) for M in self.ladder for kind in kinds]

    def _build(self):
        """All ladder functions in one Core AI program: the weights are shared."""
        from coreai_torch import ExternalizeSpec, TorchConverter
        from coreai_torch.composite_ops import RMSNormImpl

        tag = hashlib.sha256((f"{self.prefill_q}|{self.ladder}|{self.compress}|{self.composite}|{passes.PASSES_VERSION}"
                              + ("|batch" if self.batch else "") + ("|mlpchunk" if self.mlp_chunk else "")).encode()).hexdigest()[:8]
        path = CACHE_DIR / f"{model_digest(self.lm)}_state_{tag}_v5.aimodel"
        if path.exists():
            touch_cache(path)
            return path
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        hid = self.lm.config.get_text_config().hidden_size
        with warnings.catch_warnings(), torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            conv = TorchConverter()
            register_lowering(conv)
            mods = {}
            for kind, M in self._entries():
                q = self.prefill_q if kind == "prefill" else DECODE_Q
                m = mods.setdefault(M, _StateStep(self.lm, M).eval())  # modules share self.lm, and so the weights
                # positions are int16: ANE has no int32, and an int32 -> float cast in RoPE falls off ANE
                # (Apple's position_ids are uint16 for the same reason)
                args = (torch.zeros(1, q, hid, dtype=torch.float16), torch.arange(q, dtype=torch.int16)[None],
                        torch.from_numpy(window_mask(0, q, q, M)), torch.zeros(1, dtype=torch.int32),
                        torch.zeros(q if kind == "batch" else 1, q, dtype=torch.float16))
                conv.add_pytorch_module(
                    m, export_fn=passes.export_for_ane(m, args), entrypoint_name=f"{kind}_{M}",
                    externalize_modules=[ExternalizeSpec(target_class=RMSNormImpl, composite_op_name="rms_norm",
                                                         composite_attrs=["axes", "eps"])] + self._extra_specs)
            prog = conv.to_coreai()
            if self.compress:
                # lut*: projections are already compressed with palettes (LUT), int8 covers the rest (lm_head)
                prog = compress_program(prog, "int8" if self.compress.startswith("lut") else self.compress)
            # hardware constraints (as at Apple): states are IOSurface, memory that
            # ANE reads directly
            from coreai.authoring.types import AllocationType, HardwareConstraints
            hc = HardwareConstraints(AllocationType.IOSurface, alignments=(1,) * 6, interleave=(1,) * 5)
            for kind, M in self._entries():
                prog.set_hardware_constraints(f"{kind}_{M}", {"b_key_cache": hc, "b_value_cache": hc})
            prog.save_asset(path)
        prune_cache(keep=(path,))
        return path

    def _load(self, path):
        rt = self._rt
        model = self.loop.run_until_complete(rt.AIModel.load(
            path, rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, self.device)())))
        self.fn = {(kind, M): model.load_function(f"{kind}_{M}") for kind, M in self._entries()}

    def reset(self, M=None):
        """Fresh state buffers for ladder step M (the smallest by default)."""
        self.M = M or self.ladder[0]
        d = self.fn[("decode", self.M)].desc
        self.state = {n: self._rt.NDArray.from_descriptor(d.state_descriptor(n)) for n in d.state_names}
        self.pos = 0

    def _run(self, kind, tokens, start):
        rt = self._rt
        q = self.prefill_q if kind == "prefill" else DECODE_Q
        n = len(tokens)
        ids = torch.zeros(1, q, dtype=torch.long)
        ids[0, :n] = torch.tensor(tokens)
        with torch.no_grad():  # embedding table lookup on CPU (like Apple's gather_embeddings)
            emb = self._embed(ids).half().numpy()
        pick = np.zeros((1, q), np.float16)
        pick[0, n - 1] = 1
        fn = self.fn[(kind, self.M)]
        d = fn.desc
        vals = (emb, np.arange(start, start + q, dtype=np.int16)[None], window_mask(start, n, q, self.M),
                np.array([start], np.int32), pick)
        out = self.loop.run_until_complete(fn({k: rt.NDArray(v) for k, v in zip(d.input_names, vals)},
                                              state=self.state))
        return np.array(out[d.output_names[0]].numpy())[0].astype(np.float32)

    def feed(self, tokens):
        """Append tokens to the cache (after those already processed) -> logits of the last one."""
        toks = list(tokens)
        self._hist = getattr(self, "_hist", [])[:self.pos] + toks
        end = len(self._hist)
        if end > min(self.max_len, 32767):  # positions are int16
            raise ValueError(f"context longer than {self.max_len}")
        if end > self.M:  # doesn't fit the step: next step, history is recomputed from scratch
            self.reset(next(b for b in self.ladder if b >= end))
        Q, logits = self.prefill_q, None
        while end - self.pos > DECODE_Q:  # prefill chunks, the last one partial (padding under the mask)
            start = min(self.pos, self.M - Q)  # chunk fully inside the cache: shifted back at the edge
            stop = min(start + Q, end)
            logits = self._run("prefill", self._hist[start:stop], start)
            self.pos = stop
        while self.pos < end:  # tail in decode windows aligned to DECODE_Q
            start = (self.pos // DECODE_Q) * DECODE_Q
            stop = min(start + DECODE_Q, end)
            logits = self._run("decode", self._hist[start:stop], start)
            self.pos = stop
        return logits

    def generate(self, prompt, max_new_tokens=32, eos=None):
        """Greedy generation from scratch: prompt is a list of ids -> new ids (with eos, if hit)."""
        self.reset()
        self._hist = []
        out = []
        logits = self.feed(prompt)
        for _ in range(max_new_tokens):
            tok = int(logits.argmax())
            out.append(tok)
            if tok == eos or len(out) == max_new_tokens:
                break
            logits = self.feed([tok])
        return out

    def _call(self, kind, tokens, pos, mask, step, pick):
        """One function call: tokens across window rows, each row with its own positions and mask."""
        rt = self._rt
        q = self.prefill_q if kind == "prefill" else DECODE_Q
        ids = torch.zeros(1, q, dtype=torch.long)
        ids[0, :len(tokens)] = torch.tensor(tokens)
        with torch.no_grad():
            emb = self._embed(ids).half().numpy()
        p = np.zeros((1, q), np.int16)
        p[0, :len(pos)] = pos
        fn = self.fn[(kind, self.M)]
        d = fn.desc
        vals = (emb, p, mask, np.array([step], np.int32), pick)
        out = self.loop.run_until_complete(fn({k: rt.NDArray(v) for k, v in zip(d.input_names, vals)},
                                              state=self.state))
        return np.array(out[d.output_names[0]].numpy()).astype(np.float32)

    def generate_batch(self, prompts, max_new_tokens=32):
        """Greedy generation of up to DECODE_Q streams at once (needs batch=True). A decode step is one
        ANE call: window row j is the next token of stream j with its own position; K/V of all
        streams are interleaved in the shared buffer, a row's mask opens only the columns of its
        stream. Weights are read once per step, so a step costs almost as much as for a single stream."""
        if not self.batch:
            raise ValueError("generate_batch: build the model with batch=True")
        B, Q, PQ = len(prompts), DECODE_Q, self.prefill_q
        if not 1 <= B <= Q:
            raise ValueError(f"streams must be from 1 to {Q}")
        need = sum(len(p) for p in prompts) + B * max_new_tokens + PQ
        if need > self.max_len:
            raise ValueError(f"does not fit the cache: need {need} columns, max_len={self.max_len}")
        self.reset(next(b for b in self.ladder if b >= need))
        M, cur = self.M, 0
        own = np.zeros((B, M), bool)  # which cache columns belong to the stream
        logits = []
        for s, prompt in enumerate(prompts):  # prefill streams one by one, each into its own cache segment
            done = 0
            while done < len(prompt):
                chunk = prompt[done:done + PQ]
                n = len(chunk)
                m = np.full((1, 1, PQ, M), MASK_NEG, np.float16)
                m[0, 0, :, own[s]] = 0
                for j in range(PQ):  # causal within the chunk; padding rows see themselves
                    m[0, 0, j, cur:cur + (j + 1 if j < n else 0)] = 0
                    m[0, 0, j, cur + j] = 0
                pick = np.zeros((1, PQ), np.float16)
                pick[0, n - 1] = 1
                lg = self._call("prefill", chunk, np.arange(done, done + n), m, cur, pick)[0]
                own[s, cur:cur + n] = True
                cur += n
                done += n
            logits.append(lg)
        outs = [[int(lg.argmax())] for lg in logits]
        pos = [len(p) for p in prompts]
        mask = np.full((1, 1, Q, M), MASK_NEG, np.float16)
        pick = np.eye(Q, dtype=np.float16)
        for _ in range(max_new_tokens - 1):
            mask[:] = MASK_NEG
            for j in range(Q):
                if j < B:
                    mask[0, 0, j, own[j]] = 0
                mask[0, 0, j, cur + j] = 0
            lg = self._call("batch", [o[-1] for o in outs], pos, mask, cur, pick)
            own[np.arange(B), cur + np.arange(B)] = True
            cur += B
            for j in range(B):
                outs[j].append(int(lg[j].argmax()))
                pos[j] += 1
        return outs
