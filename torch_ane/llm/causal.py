"""LLM на Neural Engine с KV-кешем внутри модели — схема Apple (coreai-models,
iOS-экспорт): весь K одним состоянием Core AI, весь V другим, запись куска —
coreai.slice_update со смещением из входа. Состояния живут между вызовами и
передаются без копирования (время вызова
почти не растёт с размером кеша), в отличие от KV-кеша входом (~22 мс копирования
на шаг декода у Qwen2.5-0.5B).

Прежняя попытка (ANEChunkedPrefill: index_copy_ в отдельный буфер на каждый слой)
на ANE тихо считала неверно; с двумя общими буферами и slice_update — верно.

Как у Apple, две функции над одними состояниями: prefill кусками по PREFILL_Q
токенов и декод окнами по DECODE_Q (окно выровнено на кратное DECODE_Q, недописанный
хвост пересчитывается — запись тех же значений в те же позиции безвредна).
Модель transformers не переписывается: свой слой кеша (_StateLayer) пишет K/V в
буферы и отдаёт слою внимания срез на весь контекст, маска — 4D-вход.
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
CALIB_TEXT = ("Объясни подробно, как устроен трансформер: внимание, MLP, нормализация. "
              "Explain how attention works, step by step, with a short Python example.")


class _StateStep(torch.nn.Module):
    """(emb [1, Q, hidden], pos [1, Q], mask [1, 1, Q, M], step [1], pick [1, Q])
    -> логиты одной строки [1, V] (строка выбирается one-hot вектором pick);
    K/V куска пишутся в буферы key_cache/value_cache [L, 1, kv, M, hd].

    Как у Apple: эмбеддинги приходят входом (поиск по таблице — снаружи), в графе
    одна копия весов lm_head; логиты — только нужной позиции, а не всего окна.
    Выбор строки — умножение на one-hot в fp16 (целочисленный индекс ANE не любит)."""

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
        if pick.shape[0] > 1:  # батч потоков: pick [P, Q] -> логиты P строк [P, V]
            return self.lm.lm_head(torch.matmul(pick.to(dt), o.last_hidden_state[0]))
        h = (pick.to(dt)[:, :, None] * o.last_hidden_state).sum(1)  # [1, hidden]
        return self.lm.lm_head(h)


class CausalLM:
    """Causal LM из transformers целиком на ANE: prefill кусками, декод окнами, KV-кеш —
    состояния Core AI. Первая сборка — минуты, дальше программа из кеша.

        lm = CausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", batch=True)
        tokens = lm.generate(prompt_ids, max_new_tokens=64)
        outs = lm.generate_batch([ids1, ids2, ids3], max_new_tokens=64)   # до 8 потоков за шаг

    max_len — наибольший контекст; compress — "lut4"/"lut4g32" (палитры Apple) или сжатие
    готовой программы ("int8"...); composite — составные операции Core AI ("sdpa", "rope");
    calib_ids — токены настоящего текста для калибровки масштаба fp16; batch — собрать
    функции для generate_batch; mlp_chunk — MLP кусками под внутреннюю память ANE."""

    @classmethod
    def from_pretrained(cls, model_id, **kw):
        """Модель и (для калибровки) токенизатор из transformers по имени или пути."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        if "calib_ids" not in kw:
            try:
                tok = AutoTokenizer.from_pretrained(model_id)
                kw["calib_ids"] = tok(CALIB_TEXT)["input_ids"][:128]
            except Exception:  # noqa: BLE001 — без токенизатора: фиксированные токены
                pass
        return cls(hf, **kw)

    def __init__(self, hf_lm, max_len=2048, device="ane", prefill_q=PREFILL_Q, compress=None,
                 ladder=LADDER, composite=(), calib_ids=None, batch=False, mlp_chunk=True):
        import copy

        from ..runtime import _device
        device = _device(device)
        self.lm = fresh_rope(copy.deepcopy(hf_lm).float()).eval()
        self._embed = self.lm.get_input_embeddings()
        # мелкие активации MLP/V не тонут в субнормальных fp16 на ANE (до подмены Linear свёртками);
        # масштаб — свой у каждой пары проекций, по калибровке (общий 2^8 переполнял fp16 на
        # массивных активациях: Qwen2.5 на ANE зацикливался, как и Devstral)
        vocab = self.lm.get_input_embeddings().num_embeddings
        cal = torch.tensor([list(calib_ids) if calib_ids is not None
                            else [1 % vocab] + [i % vocab for i in range(1000, 1095)]])
        k_for, _ = passes.calibrate_rescale(self.lm, lambda: self.lm(input_ids=cal).logits)
        passes.rescale_for_fp16(self.lm, k_for=k_for)
        # MLP кусками скрытого измерения под внутреннюю память ANE (passes.chunk_mlp)
        self.mlp_chunk = mlp_chunk and passes.chunk_mlp(self.lm)
        if compress and compress.startswith("lut"):  # "lut4" / "lut4g32": палитры по рецепту Apple
            bits, _, g = compress[3:].partition("g")
            passes.swap_linear(self.lm.model)  # проекции — свёртки 1x1 (lm_head остаётся Linear)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lm = passes.palettize_lut(self.lm, int(bits), int(g or 8))
        # prefill_q: куски prefill крупнее окна декода — каждый кусок заново читает все
        # веса, мелкие куски упираются в память, а не в вычисления
        self.max_len, self.device, self.prefill_q, self.compress = max_len, device, prefill_q, compress
        self.batch = batch  # функции "batch": окно декода из токенов разных потоков (generate_batch)
        # лестница контекстов (как у Apple): внимание на ANE стоит пропорционально
        # длине кеша, короткий разговор идёт функциями с маленьким кешем
        self.ladder = tuple(b for b in ladder if b < max_len and b >= prefill_q) + (max_len,)
        passes.swap_rmsnorm(self.lm)
        # составные SDPA/RoPE Core AI (composite=("sdpa", "rope")) — как в экспорте Apple под macOS
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
        """Все функции лестницы — в одной программе Core AI: веса общие."""
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
                m = mods.setdefault(M, _StateStep(self.lm, M).eval())  # модули делят self.lm — и веса
                # позиции — int16: ANE не знает int32, и каст int32 -> float в RoPE уходит с ANE
                # (у Apple position_ids — uint16 по той же причине)
                args = (torch.zeros(1, q, hid, dtype=torch.float16), torch.arange(q, dtype=torch.int16)[None],
                        torch.from_numpy(window_mask(0, q, q, M)), torch.zeros(1, dtype=torch.int32),
                        torch.zeros(q if kind == "batch" else 1, q, dtype=torch.float16))
                conv.add_pytorch_module(
                    m, export_fn=passes.export_for_ane(m, args), entrypoint_name=f"{kind}_{M}",
                    externalize_modules=[ExternalizeSpec(target_class=RMSNormImpl, composite_op_name="rms_norm",
                                                         composite_attrs=["axes", "eps"])] + self._extra_specs)
            prog = conv.to_coreai()
            if self.compress:
                # lut*: проекции уже сжаты палитрами, int8 добирает остальное (lm_head)
                prog = compress_program(prog, "int8" if self.compress.startswith("lut") else self.compress)
            # hardware constraints (как у Apple): состояния — IOSurface, память, которую
            # ANE читает напрямую
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
        """Новые буферы состояний ступени M (по умолчанию — младшей)."""
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
        with torch.no_grad():  # поиск по таблице эмбеддингов — на CPU (как gather_embeddings у Apple)
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
        """Дописать токены в кеш (после уже обработанных) -> логиты последнего."""
        toks = list(tokens)
        self._hist = getattr(self, "_hist", [])[:self.pos] + toks
        end = len(self._hist)
        if end > min(self.max_len, 32767):  # позиции — int16
            raise ValueError(f"контекст длиннее {self.max_len}")
        if end > self.M:  # не влезает в ступень: следующая ступень, история пересчитывается заново
            self.reset(next(b for b in self.ladder if b >= end))
        Q, logits = self.prefill_q, None
        while end - self.pos > DECODE_Q:  # куски prefill, последний — неполный (паддинг под маской)
            start = min(self.pos, self.M - Q)  # кусок целиком в кеше: у края сдвигается назад
            stop = min(start + Q, end)
            logits = self._run("prefill", self._hist[start:stop], start)
            self.pos = stop
        while self.pos < end:  # хвост окнами декода, выровненными на DECODE_Q
            start = (self.pos // DECODE_Q) * DECODE_Q
            stop = min(start + DECODE_Q, end)
            logits = self._run("decode", self._hist[start:stop], start)
            self.pos = stop
        return logits

    def generate(self, prompt, max_new_tokens=32, eos=None):
        """Жадная генерация с нуля: prompt — список id -> новые id (с eos, если встретился)."""
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
        """Один вызов функции: токены по строкам окна, свои позиции и маска у каждой строки."""
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
        """Жадная генерация до DECODE_Q потоков сразу (нужен batch=True). Шаг декода — один
        вызов ANE: строка j окна — очередной токен потока j со своей позицией; K/V всех
        потоков лежат в общем буфере вперемешку, маска строки открывает только столбцы её
        потока. Веса читаются раз на шаг, поэтому шаг стоит почти как у одного потока."""
        if not self.batch:
            raise ValueError("generate_batch: собери модель с batch=True")
        B, Q, PQ = len(prompts), DECODE_Q, self.prefill_q
        if not 1 <= B <= Q:
            raise ValueError(f"потоков от 1 до {Q}")
        need = sum(len(p) for p in prompts) + B * max_new_tokens + PQ
        if need > self.max_len:
            raise ValueError(f"не влезает в кеш: нужно {need} столбцов, max_len={self.max_len}")
        self.reset(next(b for b in self.ladder if b >= need))
        M, cur = self.M, 0
        own = np.zeros((B, M), bool)  # какие столбцы кеша принадлежат потоку
        logits = []
        for s, prompt in enumerate(prompts):  # prefill потоков по очереди, каждый — в свой отрезок кеша
            done = 0
            while done < len(prompt):
                chunk = prompt[done:done + PQ]
                n = len(chunk)
                m = np.full((1, 1, PQ, M), MASK_NEG, np.float16)
                m[0, 0, :, own[s]] = 0
                for j in range(PQ):  # причинно внутри куска; строки паддинга видят себя
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
