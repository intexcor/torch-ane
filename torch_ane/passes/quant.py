"""Сжатие весов под ANE (палитры, int8) и масштаб против субнормальных fp16."""
import torch


def palettize_lut(model, n_bits=4, group_size=8, skip=("lm_head",), example_inputs=None, per_channel_scale=False,
                  bits_by_name=None, num_workers=1):
    """Сжатие весов таблицей значений по рецепту Apple (coreai-models, пресет
    4bit_weight_palettized_group{8,32}): на уровне torch до экспорта, k-means по группам
    выходных каналов, эмбеддинги и lm_head не трогаются. В графе остаётся
    coreai.lut_to_dense — его разжимает компилятор Core AI."""
    from coreai_opt import ExportBackend
    from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
    def spec(bits):
        sp = {"n_bits": bits, "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": group_size}}
        if per_channel_scale:  # строки нормируются своим масштабом перед кластеризацией
            sp["enable_per_channel_scale"] = True
        return {"op_state_spec": {"weight": sp}}
    names = {n: None for n in skip}
    # bits_by_name: {подстрока имени модуля: биты} — например 6 бит там, где в GGUF был Q6_K
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) and bits_by_name:
            for pat, b in bits_by_name.items():
                if pat in name:  # b=None — модуль не сжимать палитрой (например, позже int8)
                    names[name] = None if b is None else spec(b)
    cfg = KMeansPalettizerConfig.from_dict({"kmeans_palettization_config": {
        "global_config": spec(n_bits),
        "module_type_configs": {"torch.nn.modules.sparse.Embedding": None},
        "module_name_configs": names}})
    p = KMeansPalettizer(model, cfg)
    ex = example_inputs if example_inputs is not None else (torch.zeros(1, 8, dtype=torch.long),)
    out = p.finalize(p.prepare(example_inputs=ex, num_workers=num_workers), backend=ExportBackend.CoreAI)
    # таблицы/индексы, посчитанные из весов с requires_grad, остаются узлами autograd —
    # такие тензоры не копируются (deepcopy при экспорте); веса нам нужны только константами
    for mod in out.modules():
        for name, b in list(mod._buffers.items()):
            if b is not None and (b.requires_grad or b.grad_fn is not None):
                mod._buffers[name] = b.detach()
    for prm in out.parameters():
        prm.requires_grad_(False)
    return out


class _ScaleOut(torch.nn.Module):
    """Выход вложенной проекции, умноженный на s (масштаб — степень двойки, точно)."""

    def __init__(self, inner, s):
        super().__init__()
        self.inner, self.s = inner, float(s)

    def forward(self, x):
        return self.inner(x) * self.s


def rescale_for_fp16(model, k=8, k_for=None):
    """ANE сбрасывает в ноль крошечные (субнормальные) значения fp16, а у части моделей
    (Devstral/Mistral) промежуточные активации MLP и V/внимания как раз такие: слой на
    ANE ошибался на 2% выхода при 0.04% у fp16 в torch. Точная поправка без смены
    математики: веса up_proj и v_proj умножаются на 2^k (MLP линеен по up, внимание —
    по V), выход down_proj и o_proj — на 2^-k (веса down/o не трогаем: их деление
    само загнало бы их в субнормальные). Для Llama-подобных слоёв."""
    count = 0
    for mod in list(model.modules()):
        for up_name, down_name, owner in (("up_proj", "down_proj", "mlp"), ("v_proj", "o_proj", "self_attn")):
            up, down = getattr(mod, up_name, None), getattr(mod, down_name, None)
            if not (isinstance(up, torch.nn.Linear) and isinstance(down, torch.nn.Linear)):
                continue
            if down.bias is not None:  # смещение на выходе масштаб бы испортил
                continue
            kk = k if k_for is None else k_for(down)
            if not kk:
                continue
            s = 2.0 ** kk
            with torch.no_grad():
                up.weight.mul_(s)
                if up.bias is not None:
                    up.bias.mul_(s)
            setattr(mod, down_name, _ScaleOut(down, 1 / s))
            count += 1
    return count


def _kmeans1d_groups(x, k, iters=20):
    """Одномерный k-means для каждой строки x [G, N] -> (центры [G, k], индексы [G, N]).
    Старт с квантилей; присваивание — searchsorted по серединам между отсортированными
    центрами (O(N log k)), пересчёт — scatter_add."""
    G, N = x.shape
    qs = torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=x.device, dtype=x.dtype)
    sample = x if N <= 16384 else x[:, torch.randint(0, N, (16384,), device=x.device)]
    c = torch.quantile(sample, qs, dim=1).t().contiguous()  # [G, k]
    for _ in range(iters):
        c, _ = c.sort(dim=1)
        mid = (c[:, 1:] + c[:, :-1]) / 2
        idx = torch.searchsorted(mid, x)  # [G, N] в [0, k)
        s = torch.zeros(G, k, device=x.device, dtype=x.dtype).scatter_add_(1, idx, x)
        n = torch.zeros(G, k, device=x.device, dtype=x.dtype).scatter_add_(1, idx, torch.ones_like(x))
        c = torch.where(n > 0, s / n.clamp(min=1), c)  # пустой кластер — остаётся на месте
    c, _ = c.sort(dim=1)
    idx = torch.searchsorted((c[:, 1:] + c[:, :-1]) / 2, x)
    return c, idx


def prequantize_lut(model, n_bits=4, group_size=8, skip=(), device=None, batch_values=1 << 26, iters=20):
    """Быстрая замена медленной части сжатия палитрой: k-means для всех групп сразу на
    GPU (MPS), веса заменяются своими центрами — в каждой группе из group_size выходных
    каналов остаётся <= 2^n_bits значений. Потом обычный palettize_lut (публичное API
    coreai-opt) на таких весах сходится сразу: на Devstral k-means coreai-opt шёл
    ~40-90 с на слой одним ядром. skip — подстроки имён модулей, которые не трогать."""
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    k, count = 2 ** n_bits, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Conv2d) or any(s in name for s in skip):
            continue
        w = mod.weight.detach()
        O = w.shape[0]
        if O % group_size:
            continue
        flat = w.reshape(O // group_size, -1)  # группа = group_size выходных каналов целиком
        out = torch.empty_like(flat)
        step = max(1, batch_values // flat.shape[1])
        for g0 in range(0, flat.shape[0], step):
            part = flat[g0:g0 + step].to(dev, torch.float32)
            c, idx = _kmeans1d_groups(part, k, iters)
            out[g0:g0 + step] = c.gather(1, idx).to("cpu", w.dtype)
        with torch.no_grad():
            mod.weight.copy_(out.reshape(w.shape))
        count += 1
    return count


def quantize_int8_modules(model, patterns, example_inputs):
    """int8 по каналам (только веса, пресет w8 coreai-opt) для модулей, в имени которых
    есть одна из подстрок patterns; остальное не трогается. После этого их веса — int8
    константы с деквантованием, и сжатие таблицей готовой программы их не задевает.
    patterns=None — все nn.Linear и nn.Conv2d (с осью 0 явно: у подклассов Linear,
    как в Whisper, Quantizer её сам не выводит). Conv1d не квантуется: int8 в ней роняет
    компилятор ANE, и вся функция молча уходит на GPU."""
    from coreai_opt import ExportBackend
    from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
    from coreai_opt.quantization.config import ExecutionMode
    if patterns is None:
        from coreai_opt.quantization.spec import PerChannelGranularity, QuantizationScheme, QuantizationSpec
        w = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC,
                             granularity=PerChannelGranularity(axis=0))
        cfg = ModuleQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec={"weight": w})
        names = [n for n, m in model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    else:
        cfg = ModuleQuantizerConfig.presets.w8()
        names = [n for n, m in model.named_modules()
                 if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and any(p in n for p in patterns)]
    if not names:
        return model, 0
    q = Quantizer(model, QuantizerConfig(global_config=None, execution_mode=ExecutionMode.EAGER,
                                         module_name_configs={n: cfg for n in names}))
    q.prepare(example_inputs)
    out = q.finalize(backend=ExportBackend.CoreAI)
    for mod in out.modules():  # как в palettize_lut: веса — константы, без autograd
        for k, b in list(mod._buffers.items()):
            if b is not None and (b.requires_grad or b.grad_fn is not None):
                mod._buffers[k] = b.detach()
    for prm in out.parameters():
        prm.requires_grad_(False)
    return out, len(names)


def calibrate_rescale(model, run, inter_max=2048.0, out_max=8192.0, kmax=8):
    """Масштаб 2^k для rescale_for_fp16 — свой у каждой пары проекций, по калибровке:
    run() прогоняет модель во fp32 на настоящем тексте. Без этого общий 2^8 переполнял
    fp16 в слоях с «массивными активациями» (у Devstral слой 2: silu(gate)*up до 282,
    x256 = 72198 > 65504) — ANE обрезает до максимума, и выход куска превращался в мусор.
    k = наибольшее <= kmax, при котором вход down_proj/o_proj (и выход v_proj) x2^k <= inter_max
    и выход down_proj/o_proj x2^k <= out_max. Возвращает k_for(down_module) -> k."""
    import math
    stats, hooks = {}, []

    def upd(key, t):
        stats[key] = max(stats.get(key, 0.0), float(t.detach().abs().max()))

    for mod in list(model.modules()):
        for up_name, down_name in (("up_proj", "down_proj"), ("v_proj", "o_proj")):
            up, down = getattr(mod, up_name, None), getattr(mod, down_name, None)
            if isinstance(up, torch.nn.Linear) and isinstance(down, torch.nn.Linear):
                hooks.append(down.register_forward_pre_hook(lambda m, a: upd((id(m), "in"), a[0])))
                hooks.append(down.register_forward_hook(lambda m, a, o: upd((id(m), "out"), o)))
                if up_name == "v_proj":
                    hooks.append(up.register_forward_hook(lambda m, a, o, d=down: upd((id(d), "in"), o)))
    try:
        with torch.no_grad():
            out = run()
    finally:
        for h in hooks:
            h.remove()

    def k_for(down):
        a, b = stats.get((id(down), "in"), 0.0), stats.get((id(down), "out"), 0.0)
        k = kmax
        if a > 0:
            k = min(k, math.floor(math.log2(inter_max / a)))
        if b > 0:
            k = min(k, math.floor(math.log2(out_max / b)))
        return max(0, k)
    k_for.stats = stats
    return k_for, out
