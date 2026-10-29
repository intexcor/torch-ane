"""Проходы, которые делают произвольную torch-модель пригодной для Neural Engine.

ANE исполняет только fp16 (и int8). Модели из transformers на него не идут:
  * RMSNorm внутри уходит во float32 (`hidden_states.to(torch.float32)`);
  * маска внимания строится сравнениями int64/bool, RoPE (cos/sin) — во float32.
Core AI режет такой граф между ANE/GPU/CPU — медленно и (coreai-torch #115) тихо
неверно. Поэтому перед конвертацией, без знания о конкретной модели:

  1. swap_rmsnorm: модули, которые ВЕДУТ СЕБЯ как RMSNorm (проверка численная,
     не по имени), заменяются на RMSNormImpl — составную операцию Core AI:
     компилятор подставляет родное ядро, fp32-ветки в графе нет;
  2. fold_constants: всё, что не зависит ни от входа, ни от весов (маска, таблицы
     RoPE), считается заранее во fp32 и кладётся в граф константами (как freezing
     в Inductor и constant folding в XLA) — при статической форме это всё;
  3. causal_mask: свёрнутая маска SDPA, численно равная причинной, заменяется
     флагом is_causal=True;
  4. только потом модель и константы переводятся в fp16 — точность RoPE не теряется.
"""
import copy

import torch

from .layout import chunk_mlp, swap_linear, use_ane_attention


aten = torch.ops.aten
# Правила Apple (ниже) по умолчанию выключены — замеры на M4, macOS 27.0.1, Qwen2.5-0.5B
# компилятор Core AI сам
# кладёт SDPA и Linear на ANE; свёртка 1x1 — вровень (+0..7%), внимание по головам —
# в 1.1-2.6 раза медленнее слитого SDPA даже в чистой раскладке BC1S, а от 12 слоёв
# ANE ещё и тихо обнуляет выходы после последнего слоя.
# Включено одно правило — "mlp_chunk" (MLP кусками под внутреннюю память ANE, passes.chunk_mlp):
# энкодер Whisper large-v3-turbo 368 -> 280 мс, проход Qwen2.5-0.5B на 256 токенах
# 30.5 -> 26.3 мс, на 8 токенах вровень.
DEFAULT_RULES = ("mlp_chunk",)


# --- 1. RMSNorm -> составная операция Core AI -----------------------------------

class ANERMSNorm(torch.nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        from coreai_torch.composite_ops import RMSNormImpl
        self.weight = weight  # тот же Parameter, не копия
        self.impl = RMSNormImpl(eps=float(eps))

    def forward(self, x):
        return self.impl(x, self.weight)


def _looks_like_rmsnorm(mod):
    w = getattr(mod, "weight", None)
    eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", None))
    if not (isinstance(w, torch.nn.Parameter) and w.ndim == 1 and isinstance(eps, float)):
        return None
    if list(mod.children()):
        return None
    D = w.shape[0]
    x = torch.randn(3, 5, D, dtype=w.dtype) * 3
    try:
        with torch.no_grad():
            out = mod(x)
            ref = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * w
    except Exception:  # noqa: BLE001
        return None
    if isinstance(out, torch.Tensor) and out.shape == ref.shape and torch.allclose(out, ref, rtol=1e-4, atol=1e-5):
        return eps
    return None


def swap_rmsnorm(model):
    """Заменяет на месте; возвращает число замен."""
    count = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, ANERMSNorm):
            continue
        eps = _looks_like_rmsnorm(mod)
        if eps is None:
            continue
        parent = model
        *path, last = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, last, ANERMSNorm(mod.weight, eps))
        count += 1
    return count


# --- 2–3. свёртка констант и причинная маска (на развёрнутом графе) ------------

def fold_constants(gm, param_names):
    """Узлы, не зависящие ни от placeholder'ов, ни от параметров, вычисляются и
    заменяются константами (буферами gm). Возвращает число свёрнутых границ."""
    g = gm.graph
    live = set()
    for n in g.nodes:
        if n.op == "placeholder" or (n.op == "get_attr" and n.target in param_names):
            live.add(n)
        elif n.op == "call_function" and str(n.target).startswith("coreai."):
            # разжатие весов (coreai.lut_to_dense и т.п.) — не сворачивать: иначе сжатые
            # веса молча превратятся обратно в плотные константы
            live.add(n)
        elif n.op == "call_function" and (any(a in live for a in n.all_input_nodes)
                                          or n.target is aten._assert_tensor_metadata.default
                                          or getattr(n.target, "_schema", None) is not None
                                          and n.target._schema.is_mutable):
            live.add(n)
        elif n.op == "output":
            live.add(n)
    foldable = [n for n in g.nodes if n.op in ("call_function", "get_attr") and n not in live]
    # граница: свёртываемый узел, результат которого нужен «живой» части графа
    frontier = [n for n in foldable if n.op == "call_function" and any(u in live for u in n.users)]
    if not frontier:
        return 0
    env = {}
    for n in g.nodes:
        if n not in foldable:
            continue
        if n.op == "get_attr":
            obj = gm
            for part in n.target.split("."):
                obj = getattr(obj, part)
            env[n] = obj
        else:
            args = torch.fx.node.map_arg(n.args, lambda a: env[a])
            kwargs = torch.fx.node.map_arg(n.kwargs, lambda a: env[a])
            env[n] = n.target(*args, **kwargs)
    count = 0
    for i, n in enumerate(frontier):
        val = env[n]
        if not isinstance(val, torch.Tensor):
            continue
        name = f"_folded_{i}"
        gm.register_buffer(name, val.detach().clone().contiguous())
        with g.inserting_before(n):
            c = g.get_attr(name)
        c.meta["val"] = n.meta.get("val")
        n.replace_all_uses_with(c)
        count += 1
    g.eliminate_dead_code()
    gm.recompile()
    return count


def causal_mask_to_flag(gm):
    """attn_mask SDPA — константа, совпадающая с причинной, -> is_causal=True."""
    g = gm.graph
    count = 0
    for n in g.nodes:
        if n.op != "call_function" or n.target is not aten.scaled_dot_product_attention.default:
            continue
        args = list(n.args) + [None] * (7 - len(n.args))
        mask = n.kwargs.get("attn_mask", args[3])
        if not (isinstance(mask, torch.fx.Node) and mask.op == "get_attr"):
            continue
        m = getattr(gm, mask.target)
        L, S = m.shape[-2], m.shape[-1]
        causal = torch.ones(L, S, dtype=torch.bool).tril(S - L)
        if m.dtype == torch.bool:
            ok = bool((m == causal).all())
        else:  # аддитивная маска 0 / -inf
            ok = bool(((m == 0) == causal).all()) and bool((m[..., ~causal] < -1e4).all())
        if not ok:
            continue
        kw = dict(n.kwargs)
        kw.pop("attn_mask", None)
        kw["is_causal"] = True
        n.args = tuple(n.args[:3])
        n.kwargs = kw
        count += 1
    g.eliminate_dead_code()
    gm.recompile()
    return count


def _to_fp16(gm, example_args=()):
    """Параметры, буферы и свёрнутые константы — в fp16; явные касты во fp32
    внутри графа (кроме касты прямо в выход) — тоже в fp16."""
    # служебные проверки export вида «этот тензор fp32» после перевода в fp16 ложны
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target is aten._assert_tensor_metadata.default:
            gm.graph.erase_node(n)
    for name, t in list(gm.named_buffers()) + list(gm.named_parameters()):
        if t.is_floating_point():
            t.data = t.data.half()
    # вход графа остаётся в исходном типе (конвертер после вынесения модулей сам
    # переэкспортирует граф с исходными входами), каст в fp16 — первым узлом:
    # каст F32 <-> F16 на входе/выходе ANE поддерживает
    # у плейсхолдеров развёрнутого графа нет meta["val"] — типы берём из примерных
    # входов (плейсхолдеры идут в том же порядке, что и аргументы)
    for ph, v in zip([n for n in gm.graph.nodes if n.op == "placeholder"], example_args):
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            with gm.graph.inserting_after(ph):
                cast = gm.graph.call_function(aten._to_copy.default, (ph,), {"dtype": torch.float16})
            ph.replace_all_uses_with(cast)
            cast.args = (ph,)
    out_args = set(next(n for n in gm.graph.nodes if n.op == "output").all_input_nodes)
    # неявный float32 из целых входов (деление позиций и т.п. — масштаб внимания Llama 4
    # у Ministral3): явного dtype у узла нет, поэтому после него вставляется каст в fp16
    for n in list(gm.graph.nodes):
        val = n.meta.get("val") if n.op == "call_function" else None
        ins = [a.meta.get("val") for a in n.all_input_nodes] if val is not None else []
        if (isinstance(val, torch.Tensor) and val.dtype == torch.float32 and n not in out_args
                and "dtype" not in n.kwargs and ins
                and all(isinstance(v, torch.Tensor) and not v.is_floating_point() for v in ins)):
            with gm.graph.inserting_after(n):
                cast = gm.graph.call_function(aten._to_copy.default, (n,), {"dtype": torch.float16})
            n.replace_all_uses_with(cast)
            cast.args = (n,)
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.kwargs.get("dtype") is torch.float32 and n not in out_args:
            kw = dict(n.kwargs)
            kw["dtype"] = torch.float16
            n.kwargs = kw
    gm.recompile()
    return gm


# --- сборка --------------------------------------------------------------------

# версия проходов export_for_ane — входит в ключи кешей сборок (меняется — сборки пересобираются)
PASSES_VERSION = "s1"  # s1: SiLU через tanh


def silu_to_tanh(gm):
    """SiLU -> x·0.5·(1 + tanh(x/2)). Родная SiLU на ANE врёт около нуля: на выходе
    GroupNorm (|x| < 2) ~5% относительной ошибки, UNet tiny-sd из-за этого расходился с
    eager на 2.5e-2 за вызов. Запись x·sigmoid(x) компилятор узнаёт и подставляет ту же
    SiLU; через tanh — не узнаёт и считает точно (7.8e-4). Возвращает число замен."""
    aten = torch.ops.aten
    sig = {aten.sigmoid.default}
    n = 0
    for node in list(gm.graph.nodes):
        x = None
        if node.op == "call_function" and node.target == aten.silu.default:
            x = node.args[0]
        elif node.op == "call_function" and node.target == aten.mul.Tensor and len(node.args) == 2:
            a, b = node.args
            for p, q in ((a, b), (b, a)):  # x * sigmoid(x)
                if getattr(q, "op", None) == "call_function" and q.target in sig and q.args[0] is p:
                    x = p
        if x is None:
            continue
        with gm.graph.inserting_before(node):
            half = gm.graph.call_function(aten.mul.Tensor, (x, 0.5))
            t = gm.graph.call_function(aten.tanh.default, (half,))
            one = gm.graph.call_function(aten.add.Tensor, (t, 1.0))
            y = gm.graph.call_function(aten.mul.Tensor, (half, one))  # x·0.5·(1 + tanh(x/2))
        node.replace_all_uses_with(y)
        gm.graph.erase_node(node)
        n += 1
    if n:
        gm.graph.eliminate_dead_code()
        gm.recompile()
    return n


def export_for_ane(model, args, stats=None):
    """export_fn для TorchConverter.add_pytorch_module: fp32-экспорт -> свёртка
    констант -> причинная маска -> fp16 -> повторный экспорт с декомпозициями."""
    from coreai_torch import get_decomp_table

    def export_fn(m):
        # входы — fp16 (их подаёт _HalfInputs из runtime.py, внутри они приводятся к fp32,
        # после _to_fp16 этот каст исчезает): fp32-вход с кастом прямо в операцию ANE
        # считает неверно («ANE I/O op can only do F16 MemRef <-> F32 Tensor cast»)
        ep = torch.export.export(m, args).run_decompositions(get_decomp_table())
        # копия: ep.module() делит параметры с моделью, а _to_fp16 меняет их на месте —
        # конвертер потом ещё раз экспортирует исходную модель с fp32-входами
        gm = copy.deepcopy(ep.module())
        params = {n for n, _ in gm.named_parameters()}
        # изменяемые буферы (KV-кеш как состояние модели) — не константы
        params |= set(ep.graph_signature.buffers_to_mutate.values())
        folded = fold_constants(gm, params)
        causal = causal_mask_to_flag(gm)
        silu = silu_to_tanh(gm)
        _to_fp16(gm, args)
        ep2 = torch.export.export(gm, args).run_decompositions(get_decomp_table())
        if stats is not None:
            stats.update(folded=folded, causal=causal, silu=silu)
        return ep2
    return export_fn


def prepare(model, rules=None):
    """Копия модели под ANE. rules — набор правил Apple: "conv" (Linear -> свёртка 1x1,
    раскладка bc1s), "conv_n111" (то же, токены по батчу), "heads" (внимание по
    головам), "mlp_chunk" (MLP кусками скрытого измерения под внутреннюю память ANE). Возвращает (модель, статистика)."""
    m = copy.deepcopy(model).eval()
    stats = {"rms_norm": swap_rmsnorm(m)}
    rules = set(DEFAULT_RULES if rules is None else rules)
    if "heads" in rules:
        stats["heads"] = use_ane_attention(m)
    if "mlp_chunk" in rules:
        stats["mlp_chunk"] = chunk_mlp(m)
    if "conv" in rules or "conv_n111" in rules:
        stats["conv"] = swap_linear(m, "n111" if "conv_n111" in rules else "bc1s")
    return m, stats
