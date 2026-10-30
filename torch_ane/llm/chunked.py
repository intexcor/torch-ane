"""Large LLMs on the Neural Engine in chunks, for models that can't be built whole:
palette (LUT) weight compression in torch keeps indices at one byte per weight, so a 24B model is 22 GB
even before export. Each chunk of several decoder layers is built and compressed
separately (memory is freed); in the finished file the indices are packed at 4 bits.

A chunk: hidden state -> hidden state, KV of its own layers as Core AI states
(Apple's scheme, llm/_kv.py), context ladder, wide MLP projections split
into SPLIT channels (on ANE a 5120x32768 conv is twice as slow as two/four narrow ones).
The head (final norm + lm_head) is a separate program; embeddings run on CPU.

Currently the weight source is GGUF (llama.cpp) of Llama-like models (Devstral/Ministral3).

    lm = ChunkedCausalLM.from_gguf(gguf_path, config_dir)
    tokens = lm.generate(prompt_ids, 64)
"""
import asyncio
import contextlib
import copy
import gc
import hashlib
import io
import json
import pathlib
import time
import warnings

import numpy as np
import torch

from .. import passes
from ..runtime import CACHE_DIR, compress_program, prune_cache, touch_cache
from ._kv import DECODE_Q, StateLayer, register_lowering, window_mask

SPLIT = 8192
PREFILL_QS = (32, 64, 128, 256)  # prefill chunk length ladder (Apple uses q from {8, 16, 64})
# deliberately bilingual (Russian + English + code): calibration data, not docs
CALIB_TEXT = ("Привет! Объясни, как работает рекурсия, и напиши пример на Python.\n"
              "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
              "The capital of Australia is Canberra. Мак почти не греется, когда работает Neural Engine.")


class _Chunk(torch.nn.Module):
    """(h [1, Q, H], pos [1, Q] int16, mask [1, 1, Q, M], step [1]) -> h."""

    def __init__(self, cfg, layers, rotary, max_len):
        super().__init__()
        self.layers, self.rotary, self.max_len = layers, rotary, max_len
        for i, layer in enumerate(layers):
            layer.self_attn.layer_idx = i
        shape = (len(layers), 1, cfg.num_key_value_heads, max_len, cfg.head_dim)
        self.register_buffer("key_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("value_cache", torch.zeros(shape), persistent=False)

    def forward(self, h, pos, mask, step):
        from transformers.cache_utils import Cache
        dt = torch.float32  # export in fp32, conversion to fp16 is done by the passes._to_fp16 pass
        h, mask = h.to(dt), mask.to(dt)
        cache = Cache(layers=[StateLayer(self, i, step) for i in range(len(self.layers))])
        pe = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=cache, position_embeddings=pe)
        return h


class _LastChunk(_Chunk):
    """The last chunk together with the head (final norm + lm_head): one fewer call per token
    and fewer buffers (Core AI crashes after 16384 allocated buffers per process, #75).
    (h, pos, mask, step, pick [1, Q]) -> logits [1, V] of row pick."""

    def __init__(self, cfg, layers, rotary, max_len, norm, lm_head):
        super().__init__(cfg, layers, rotary, max_len)
        self.norm, self.lm_head = norm, lm_head

    def forward(self, h, pos, mask, step, pick):
        h = super().forward(h, pos, mask, step)
        h = (pick.float()[:, :, None] * self.norm(h)).sum(1, keepdim=True)
        return self.lm_head(h)[:, 0]


class _Head(torch.nn.Module):
    """(h [1, Q, H], pick [1, Q]) -> logits [1, V] of row pick."""

    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm, self.lm_head = norm, lm_head

    def forward(self, h, pick):
        h = (pick.float()[:, :, None] * self.norm(h.float())).sum(1, keepdim=True)  # [1, 1, H]
        return self.lm_head(h)[:, 0]


def _convert(module, entries, path, compress_int8=False):
    """entries: {function name: example args}. Saves the Core AI program to path."""
    from coreai.authoring.types import AllocationType, HardwareConstraints
    from coreai_torch import ExternalizeSpec, TorchConverter
    from coreai_torch.composite_ops import RMSNormImpl

    with warnings.catch_warnings(), torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        conv = TorchConverter()
        register_lowering(conv)
        for name, args in entries.items():
            conv.add_pytorch_module(
                module, export_fn=passes.export_for_ane(module, args), entrypoint_name=name,
                externalize_modules=[ExternalizeSpec(target_class=RMSNormImpl, composite_op_name="rms_norm",
                                                     composite_attrs=["axes", "eps"])])
        prog = conv.to_coreai()
        if compress_int8:
            prog = compress_program(prog, "int8")
        if hasattr(module, "key_cache"):
            hc = HardwareConstraints(AllocationType.IOSurface, alignments=(1,) * 6, interleave=(1,) * 5)
            for name in entries:
                prog.set_hardware_constraints(name, {"b_key_cache": hc, "b_value_cache": hc})
        tmp = path.with_name(path.stem + ".partial.aimodel")  # save_asset requires .aimodel
        prog.save_asset(tmp)
        tmp.rename(path)  # the finished file appears atomically: an interrupted build won't leave a broken cache


class ChunkedCausalLM:
    def __init__(self, cfg, load_layers, rotary, norm, lm_head_weight, embed_rows, key,
                 layers_per_chunk=4, max_len=4096, ladder=(1024, 2048), prefill_q=256, split=SPLIT, log=print,
                 six_bit=None, calib_ids=None, programs=None, mlp_chunk=False):
        """load_layers(lo, hi) -> ModuleList of layers [lo, hi) in fp32; lm_head_weight() -> [V, H]
        (a function: called only when building the head); embed_rows(ids) -> [n, H];
        key: a string that uniquely identifies the weights (for the cache name);
        six_bit(i) -> names of layer i's projections ("self_attn.v_proj", "mlp.down_proj") to compress
        to 6 bits (where the source is more precise than 4 bits: Q6_K in GGUF);
        mlp_chunk: MLP in chunks of the hidden dimension of 16 MB of weights each (passes.chunk_mlp): for
        Devstral MLP 5120 -> 32768 -> 5120 with 4-bit palettes (LUT), prefill 256 goes 52.8 -> 33.3 ms,
        decode is on par. Doesn't work on the whole model:
        build takes 4.6 h instead of 35 min, some chunks fall off to GPU with timeouts, tokens are garbage
        -- so it is off."""
        self.cfg, self.max_len, self.prefill_q = cfg, max_len, prefill_q
        # request length ladder, as at Apple (extend/prompt_opt with q from a set): short
        # turns go in a chunk of 64, long ones in 256, the tail and decode in a window of 8
        self.qs = {f"prefill{q}": q for q in PREFILL_QS if q <= prefill_q}
        self.qs["decode"] = DECODE_Q
        self.ladder = tuple(b for b in ladder if b < max_len and b >= prefill_q) + (max_len,)
        self.embed_rows = embed_rows
        self.loop = asyncio.new_event_loop()
        import coreai.runtime as rt
        self._rt = rt
        L, H = cfg.num_hidden_layers, cfg.hidden_size
        tag = hashlib.sha256((f"{key}|{layers_per_chunk}|{self.ladder}|{prefill_q}|{split}|v10-qladder"
                              + ("|mlpchunk16" if mlp_chunk else "")).encode()).hexdigest()[:10]
        root = CACHE_DIR / f"chunked_{tag}"
        root.mkdir(parents=True, exist_ok=True)
        touch_cache(root)
        prune_cache(keep=(root,))  # make room for the build before it starts (Devstral ~13 GB)
        self.root = root
        # program = layers_per_chunk layers (fewer calls and buffers per token); it is built
        # in pairs of layers (part): a pair in fp32 is ~4.4 GB, ~1 GB after compression, and only then
        # are the pairs joined into a program. The last program includes the head.
        part = 2
        spans = [(lo, min(lo + layers_per_chunk, L)) for lo in range(0, L, layers_per_chunk)]
        paths = [root / (f"layers_{lo:03d}_{hi:03d}" + ("_head" if hi == L else "") + ".aimodel") for lo, hi in spans]
        # fp16 scale calibration: a short real text, hidden states chunk by chunk in fp32
        cal_ids = list(calib_ids) if calib_ids is not None else [1] + list(range(1000, 1063))
        self._cal_h = torch.from_numpy(np.asarray(embed_rows(cal_ids), np.float32))[None]
        need = [not p.exists() for p in paths]
        last_needed = max((i for i, n in enumerate(need) if n), default=-1)
        import os
        workers = max(1, min(6, (os.cpu_count() or 4) - 2))
        for ci, ((lo, hi), path) in enumerate(zip(spans, paths)):
            if path.exists():
                if ci < last_needed:  # program already built: just carry the calibration forward
                    for plo in range(lo, hi, part):
                        with torch.no_grad():
                            self._cal_h = _Chunk(cfg, load_layers(plo, min(plo + part, hi)), rotary,
                                                 self._cal_h.shape[1]).eval()(*self._cal_args())
                        gc.collect()
                continue
            t = time.perf_counter()
            done, ks = [], []
            for plo in range(lo, hi, part):
                phi = min(plo + part, hi)
                layers = load_layers(plo, phi)
                # scale against fp16 subnormals: each projection pair gets its own, from calibration
                # on real text (a shared 2^8 overflowed fp16 in layers with massive activations)
                cal = _Chunk(cfg, layers, rotary, self._cal_h.shape[1]).eval()
                k_for, self._cal_h = passes.calibrate_rescale(layers, lambda: cal(*self._cal_args()))
                passes.rescale_for_fp16(layers, k_for=k_for)
                ks += [f"{k_for(l.mlp.down_proj.inner)}/{k_for(l.self_attn.o_proj.inner)}"
                       if hasattr(l.mlp.down_proj, "inner") else "?" for l in layers]
                if mlp_chunk:
                    passes.chunk_mlp(layers, max_bytes=16 << 20)
                passes.swap_linear(layers)
                passes.split_wide(layers, split)
                m = _Chunk(cfg, layers, rotary, max_len).eval()
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # tensors that are more precise than 4 bits in the source (Q6_K in GGUF) get per-channel int8
                    # (error vs GGUF 0.235 -> ~0.13); the rest get lut4
                    # via coreai-opt's exact 1D k-means (kmeans1d) across several processes
                    hi_prec = tuple(f"layers.{i - plo}.{name}" for i in range(plo, phi)
                                    for name in (six_bit(i) if six_bit else ()))
                    if mlp_chunk:  # after chunk_mlp, down_proj is chunks mlp.down.<j> (the pattern is a substring)
                        hi_prec = tuple(n.replace("mlp.down_proj", "mlp.down.") for n in hi_prec)
                    m = passes.palettize_lut(m, 4, 8, skip=(), example_inputs=self._chunk_args(DECODE_Q, max_len),
                                          bits_by_name={p: None for p in hi_prec}, num_workers=workers)
                    m, _ = passes.quantize_int8_modules(m, hi_prec, self._chunk_args(DECODE_Q, max_len))
                done += list(m.layers)
                del layers, m, cal
                gc.collect()
            layers = torch.nn.ModuleList(done)
            if hi == L:  # the last program includes the head (lm_head per-channel int8)
                w = lm_head_weight()
                lin = torch.nn.Linear(H, w.shape[0], bias=False)
                with torch.no_grad():
                    lin.weight.copy_(w)
                del w
                wrap = torch.nn.Module()
                wrap.lm_head = lin
                passes.swap_linear(wrap, skip=())
                passes.split_wide(wrap, split)
                m = _LastChunk(cfg, layers, rotary, max_len, copy.deepcopy(norm), wrap.lm_head).eval()
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m, _ = passes.quantize_int8_modules(m, ("lm_head",), self._chunk_args(DECODE_Q, max_len, pick=True))
            else:
                m = _Chunk(cfg, layers, rotary, max_len).eval()
            passes.swap_rmsnorm(m)
            self._convert_ladder(m, path, head=hi == L)
            del layers, m
            gc.collect()
            log(f"  layers {lo}-{hi - 1}{' + head' if hi == L else ''}: built in {time.perf_counter() - t:.0f} s "
                f"(scale 2^k, k mlp/attn: {ks})")
        # programs: load only some of the programs (hybrid: first layers on ANE, the rest on GPU)
        self.chunks = [p for i, p in enumerate(paths) if programs is None or i in programs]
        self.has_head = self.chunks[-1] == paths[-1]
        self._load()
        self.reset()
        self.keep_warm(kinds=None)  # the first call of each function is expensive: do it now, not on the first turn

    def _cal_args(self):
        n = self._cal_h.shape[1]
        return (self._cal_h, torch.arange(n)[None], torch.from_numpy(window_mask(0, n, n, n)).float(),
                torch.zeros(1, dtype=torch.int32))

    def _chunk_args(self, q, M, pick=False):
        H = self.cfg.hidden_size
        args = (torch.zeros(1, q, H, dtype=torch.float16), torch.arange(q, dtype=torch.int16)[None],
                torch.from_numpy(window_mask(0, q, q, M)), torch.zeros(1, dtype=torch.int32))
        return args + (torch.zeros(1, q, dtype=torch.float16),) if pick else args

    def _convert_ladder(self, m, path, lut=None, head=False):
        """Ladder functions as one program: each cache size has its own wrapper module
        with the same (already compressed) layers."""
        entries = {}
        mods = {}
        for M in self.ladder:
            mm = copy.copy(m)  # same submodules/weights, own cache buffers
            torch.nn.Module.__init__(mm)
            mm.layers, mm.rotary, mm.max_len = m.layers, m.rotary, M
            if head:
                mm.norm, mm.lm_head = m.norm, m.lm_head
            shape = (len(m.layers), 1, self.cfg.num_key_value_heads, M, self.cfg.head_dim)
            mm.register_buffer("key_cache", torch.zeros(shape), persistent=False)
            mm.register_buffer("value_cache", torch.zeros(shape), persistent=False)
            mods[M] = mm
        from coreai.authoring.types import AllocationType, HardwareConstraints
        from coreai_torch import ExternalizeSpec, TorchConverter
        from coreai_torch.composite_ops import RMSNormImpl

        with warnings.catch_warnings(), torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            conv = TorchConverter()
            register_lowering(conv)
            for M, mm in mods.items():
                for kind, q in self.qs.items():
                    args = self._chunk_args(q, M, pick=head)
                    entries[f"{kind}_{M}"] = args
                    conv.add_pytorch_module(
                        mm, export_fn=passes.export_for_ane(mm, args), entrypoint_name=f"{kind}_{M}",
                        externalize_modules=[ExternalizeSpec(target_class=RMSNormImpl, composite_op_name="rms_norm",
                                                             composite_attrs=["axes", "eps"])])
            prog = conv.to_coreai()
            if lut:  # tables from already clustered weights (int8 weights are quantized and left untouched)
                    prog = compress_program(prog, lut)
            hc = HardwareConstraints(AllocationType.IOSurface, alignments=(1,) * 6, interleave=(1,) * 5)
            for name in entries:
                prog.set_hardware_constraints(name, {"b_key_cache": hc, "b_value_cache": hc})
            tmp = path.with_name(path.stem + ".partial.aimodel")  # save_asset requires .aimodel
            prog.save_asset(tmp)
            tmp.rename(path)

    def _load(self):
        rt = self._rt
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())
        self.fn = []
        for p in self.chunks:
            model = self.loop.run_until_complete(rt.AIModel.load(p, opts))
            self.fn.append({name: model.load_function(name) for name in model.function_names})
        # Core AI crashes (uncatchably) after 16384 allocated NDArray buffers per process
        # (apple/coreai-torch#75): count them so the engine can be restarted in advance
        self.n_alloc = 0

    def reset(self, M=None):
        self.M = M or self.ladder[0]
        self.state = []
        for fns in self.fn:
            d = fns[f"decode_{self.M}"].desc
            self.state.append({n: self._rt.NDArray.from_descriptor(d.state_descriptor(n)) for n in d.state_names})
        self.pos, self._hist = 0, []

    def _run(self, kind, tokens, start):
        rt = self._rt
        q = self.qs[kind]
        n = len(tokens)
        h = np.zeros((1, q, self.cfg.hidden_size), np.float16)
        h[0, :n] = self.embed_rows(tokens)
        hnd = rt.NDArray(h)
        others = (np.arange(start, start + q, dtype=np.int16)[None], window_mask(start, n, q, self.M),
                  np.array([start], np.int32))
        others = [rt.NDArray(v) for v in others]
        name = f"{kind}_{self.M}"
        pick = np.zeros((1, q), np.float16)
        pick[0, n - 1] = 1
        last = len(self.fn) - 1 if self.has_head else -1
        for i, (fns, st) in enumerate(zip(self.fn, self.state)):
            fn = fns[name]
            d = fn.desc
            ins = [hnd] + others + ([rt.NDArray(pick)] if i == last else [])
            out = self.loop.run_until_complete(fn(dict(zip(d.input_names, ins)), state=st))
            hnd = out[d.output_names[0]]  # hidden state goes straight in as the next chunk's input
        self.n_alloc += 5 + len(self.fn)  # inputs (h, pos, mask, step, pick) + program outputs
        if not self.has_head:  # without the head: the window's hidden state [q, H] (hybrid with GPU)
            return np.array(hnd.numpy())[0]
        return np.array(hnd.numpy())[0].astype(np.float32)

    def keep_warm(self, kinds=("decode",)):
        """Keep ANE from unloading the programs: after ~10 s idle it unloads them, and the next
        prefill costs 4.5-6 s instead of 1.75. Calls the kinds functions of every
        program on dummy data and in a separate dummy cache (the real one is untouched).
        kinds=None: all functions (warm-up after loading: the first call of a function is expensive)."""
        rt = self._rt
        M, H = self.ladder[0], self.cfg.hidden_size
        if not hasattr(self, "_warm"):
            self._warm = {}
            states = [{n: rt.NDArray.from_descriptor(f[f"decode_{M}"].desc.state_descriptor(n))
                       for n in f[f"decode_{M}"].desc.state_names} for f in self.fn]
            self._warm_states = states
            self.n_alloc += 2 * len(self.fn)
        for kind in (kinds or tuple(self.qs)):
            q = self.qs[kind]
            if kind not in self._warm:
                ins = [rt.NDArray(np.zeros((1, q, H), np.float16)), rt.NDArray(np.arange(q, dtype=np.int16)[None]),
                       rt.NDArray(window_mask(0, q, q, M)), rt.NDArray(np.array([0], np.int32)),
                       rt.NDArray(np.eye(1, q, q - 1, dtype=np.float16))]
                self._warm[kind] = ins
                self.n_alloc += 5
            ins = self._warm[kind]
            for i, (fns, st) in enumerate(zip(self.fn, self._warm_states)):
                fn = fns[f"{kind}_{M}"]
                args = ins[:4] + ([ins[4]] if self.has_head and i == len(self.fn) - 1 else [])
                self.loop.run_until_complete(fn(dict(zip(fn.desc.input_names, args)), state=st))
            self.n_alloc += len(self.fn)

    ALLOC_LIMIT = 16384  # apple/coreai-torch#75

    def steps_left(self):
        """How many more calls (tokens) the process can take before Core AI crashes."""
        return (self.ALLOC_LIMIT - 200 - self.n_alloc) // (5 + len(self.fn))

    def feed(self, tokens):
        """Append tokens to the cache -> logits of the last one (like CausalLM.feed)."""
        self._hist = self._hist[:self.pos] + list(tokens)
        end = len(self._hist)
        if end > min(self.max_len, 32767):
            raise ValueError(f"context longer than {self.max_len}")
        if end > self.M:
            hist = self._hist
            self.reset(next(b for b in self.ladder if b >= end))
            self._hist = hist
        logits = None
        while end - self.pos > DECODE_Q:
            # the smallest chunk that fits the remainder (padding at most ~2x), otherwise the largest
            left = end - self.pos
            pre = sorted((q, k) for k, q in self.qs.items() if k != "decode")
            q, kind = next(((q, k) for q, k in pre if q >= left), pre[-1])
            start = min(self.pos, self.M - q)
            stop = min(start + q, end)
            logits = self._run(kind, self._hist[start:stop], start)
            self.pos = stop
        while self.pos < end:
            start = (self.pos // DECODE_Q) * DECODE_Q
            stop = min(start + DECODE_Q, end)
            logits = self._run("decode", self._hist[start:stop], start)
            self.pos = stop
        return logits

    def generate(self, prompt, max_new_tokens=32, eos=None):
        self.reset()
        out = []
        logits = self.feed(prompt)
        for _ in range(max_new_tokens):
            tok = int(logits.argmax())
            out.append(tok)
            if tok == eos or len(out) == max_new_tokens:
                break
            logits = self.feed([tok])
        return out

    @classmethod
    def from_gguf(cls, gguf_path, config, **kw):
        """Llama-like model from GGUF (Devstral/Ministral3 etc.); config: config.json
        of the source model (folder, file or dict; for multimodal ones text_config is used)."""
        import gguf
        from transformers import AutoConfig

        from .gguf import BLOCK, unpermute
        gguf_path = pathlib.Path(gguf_path)
        if not isinstance(config, dict):
            p = pathlib.Path(config)
            config = json.loads((p / "config.json" if p.is_dir() else p).read_text())
        tc = dict(config.get("text_config", config))
        cfg = AutoConfig.for_model(tc.pop("model_type"), **tc)
        cfg._attn_implementation = "sdpa"  # otherwise standalone layers use "eager" attention
        from transformers import AutoModelForCausalLM
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(cfg)
        layer_cls = type(skeleton.model.layers[0])
        rotary = type(skeleton.model.rotary_emb)(config=cfg)
        reader = gguf.GGUFReader(str(gguf_path))
        tensors = {t.name: t for t in reader.tensors}

        def deq(name):
            t = tensors[name]
            return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).copy())

        def load_layers(lo, hi):
            layers = torch.nn.ModuleList(layer_cls(cfg, i) for i in range(lo, hi))
            sd = {}
            for i in range(lo, hi):
                for kind, dst in BLOCK.items():
                    for suffix in ("weight", "bias"):
                        name = f"blk.{i}.{kind}.{suffix}"
                        if name not in tensors:
                            continue
                        w = deq(name)
                        if kind in ("attn_q", "attn_k") and suffix == "weight":
                            heads = cfg.num_attention_heads if kind == "attn_q" else cfg.num_key_value_heads
                            w = torch.from_numpy(unpermute(w.numpy(), heads).copy())
                        sd[f"{i - lo}.{dst}.{suffix}"] = w
            res = layers.load_state_dict(sd, strict=False)
            assert not res.missing_keys, res.missing_keys
            return layers.float().eval()

        norm = type(skeleton.model.norm)(cfg.hidden_size, eps=cfg.rms_norm_eps)
        with torch.no_grad():
            norm.weight.copy_(deq("output_norm.weight"))
        head_name = "output.weight" if "output.weight" in tensors else "token_embd.weight"
        emb_t = tensors["token_embd.weight"]

        def embed_rows(ids):
            rows = np.asarray(emb_t.data)[list(ids)]  # rows of the quantized table: dequantize only those
            return gguf.quants.dequantize(rows, emb_t.tensor_type).reshape(len(ids), -1).astype(np.float16)

        key = f"{gguf_path.resolve()}:{gguf_path.stat().st_size}"
        if "calib_ids" not in kw:  # fp16 scale calibration on real text, if a tokenizer is available
            try:
                from transformers import AutoTokenizer
                cdir = pathlib.Path(config) if not isinstance(config, dict) else None
                tok = AutoTokenizer.from_pretrained(cdir if cdir and cdir.is_dir() else cdir.parent,
                                                    fix_mistral_regex=True)
                kw["calib_ids"] = tok(CALIB_TEXT)["input_ids"][:96]
            except Exception:  # noqa: BLE001 — no tokenizer: BOS + fixed tokens
                pass
        del skeleton
        def six_bit(i):
            return [dst for kind, dst in (("attn_v", "self_attn.v_proj"), ("ffn_down", "mlp.down_proj"))
                    if tensors.get(f"blk.{i}.{kind}.weight") is not None
                    and tensors[f"blk.{i}.{kind}.weight"].tensor_type.name in ("Q6_K", "Q8_0", "Q5_K")]

        kw.setdefault("six_bit", six_bit)
        return cls(cfg, load_layers, rotary, norm, lambda: deq(head_name), embed_rows, key, **kw)
