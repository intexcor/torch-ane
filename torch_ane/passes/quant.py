"""Weight compression for ANE (palettes, int8) and scale against subnormal fp16."""
import torch


def palettize_lut(model, n_bits=4, group_size=8, skip=("lm_head",), example_inputs=None, per_channel_scale=False,
                  bits_by_name=None, num_workers=1):
    """Weight compression with a lookup table per Apple's recipe (coreai-models, preset
    4bit_weight_palettized_group{8,32}): at the torch level before export, k-means over groups
    of output channels, embeddings and lm_head are left untouched. The graph keeps
    coreai.lut_to_dense — the Core AI compiler decompresses it."""
    from coreai_opt import ExportBackend
    from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
    def spec(bits):
        sp = {"n_bits": bits, "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": group_size}}
        if per_channel_scale:  # rows are normalized by their own scale before clustering
            sp["enable_per_channel_scale"] = True
        return {"op_state_spec": {"weight": sp}}
    names = {n: None for n in skip}
    # bits_by_name: {module name substring: bits} — e.g. 6 bits where GGUF had Q6_K
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) and bits_by_name:
            for pat, b in bits_by_name.items():
                if pat in name:  # b=None — don't palettize the module (e.g. int8 later)
                    names[name] = None if b is None else spec(b)
    cfg = KMeansPalettizerConfig.from_dict({"kmeans_palettization_config": {
        "global_config": spec(n_bits),
        "module_type_configs": {"torch.nn.modules.sparse.Embedding": None},
        "module_name_configs": names}})
    p = KMeansPalettizer(model, cfg)
    ex = example_inputs if example_inputs is not None else (torch.zeros(1, 8, dtype=torch.long),)
    out = p.finalize(p.prepare(example_inputs=ex, num_workers=num_workers), backend=ExportBackend.CoreAI)
    # tables/indices computed from weights with requires_grad remain autograd nodes —
    # such tensors aren't copied (deepcopy during export); we only need the weights as constants
    for mod in out.modules():
        for name, b in list(mod._buffers.items()):
            if b is not None and (b.requires_grad or b.grad_fn is not None):
                mod._buffers[name] = b.detach()
    for prm in out.parameters():
        prm.requires_grad_(False)
    return out


class _ScaleOut(torch.nn.Module):
    """Output of the wrapped projection multiplied by s (the scale is a power of two, exact)."""

    def __init__(self, inner, s):
        super().__init__()
        self.inner, self.s = inner, float(s)

    def forward(self, x):
        return self.inner(x) * self.s


def rescale_for_fp16(model, k=8, k_for=None):
    """ANE flushes tiny (subnormal) fp16 values to zero, and in some models
    (Devstral/Mistral) the intermediate MLP and V/attention activations are exactly that: a layer on
    ANE was off by 2% of the output vs 0.04% for fp16 in torch. An exact fix without changing
    the math: up_proj and v_proj weights are multiplied by 2^k (the MLP is linear in up, attention —
    in V), the down_proj and o_proj output by 2^-k (down/o weights are left alone: dividing them
    would itself push them into subnormals). For Llama-like layers."""
    count = 0
    for mod in list(model.modules()):
        for up_name, down_name, owner in (("up_proj", "down_proj", "mlp"), ("v_proj", "o_proj", "self_attn")):
            up, down = getattr(mod, up_name, None), getattr(mod, down_name, None)
            if not (isinstance(up, torch.nn.Linear) and isinstance(down, torch.nn.Linear)):
                continue
            if down.bias is not None:  # a bias on the output would break the scale
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
    """1D k-means for each row of x [G, N] -> (centers [G, k], indices [G, N]).
    Initialized from quantiles; assignment — searchsorted over midpoints between sorted
    centers (O(N log k)), update — scatter_add."""
    G, N = x.shape
    qs = torch.linspace(0.5 / k, 1 - 0.5 / k, k, device=x.device, dtype=x.dtype)
    sample = x if N <= 16384 else x[:, torch.randint(0, N, (16384,), device=x.device)]
    c = torch.quantile(sample, qs, dim=1).t().contiguous()  # [G, k]
    for _ in range(iters):
        c, _ = c.sort(dim=1)
        mid = (c[:, 1:] + c[:, :-1]) / 2
        idx = torch.searchsorted(mid, x)  # [G, N] in [0, k)
        s = torch.zeros(G, k, device=x.device, dtype=x.dtype).scatter_add_(1, idx, x)
        n = torch.zeros(G, k, device=x.device, dtype=x.dtype).scatter_add_(1, idx, torch.ones_like(x))
        c = torch.where(n > 0, s / n.clamp(min=1), c)  # an empty cluster stays in place
    c, _ = c.sort(dim=1)
    idx = torch.searchsorted((c[:, 1:] + c[:, :-1]) / 2, x)
    return c, idx


def prequantize_lut(model, n_bits=4, group_size=8, skip=(), device=None, batch_values=1 << 26, iters=20):
    """Fast replacement for the slow part of palettization: k-means for all groups at once on
    GPU (MPS), weights are replaced by their centers — each group of group_size output
    channels keeps <= 2^n_bits values. Then the regular palettize_lut (public coreai-opt
    API) on such weights converges immediately: on Devstral the coreai-opt k-means took
    ~40-90 s per layer on a single core. skip — module name substrings to leave untouched."""
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    k, count = 2 ** n_bits, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Conv2d) or any(s in name for s in skip):
            continue
        w = mod.weight.detach()
        O = w.shape[0]
        if O % group_size:
            continue
        flat = w.reshape(O // group_size, -1)  # group = group_size whole output channels
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
    """Per-channel int8 (weights only, coreai-opt w8 preset) for modules whose name
    contains one of the patterns substrings; the rest is left untouched. After this their weights are int8
    constants with dequantization, and the finished program's table compression doesn't touch them.
    patterns=None — all nn.Linear and nn.Conv2d (with axis 0 explicitly: for Linear subclasses,
    as in Whisper, Quantizer doesn't infer it itself). Conv1d is not quantized: int8 in it crashes
    the ANE compiler, and the whole function silently falls back to GPU."""
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
    for mod in out.modules():  # as in palettize_lut: weights are constants, no autograd
        for k, b in list(mod._buffers.items()):
            if b is not None and (b.requires_grad or b.grad_fn is not None):
                mod._buffers[k] = b.detach()
    for prm in out.parameters():
        prm.requires_grad_(False)
    return out, len(names)


def calibrate_rescale(model, run, inter_max=2048.0, out_max=8192.0, kmax=8):
    """Scale 2^k for rescale_for_fp16 — per projection pair, from calibration:
    run() runs the model in fp32 on real text. Without it a shared 2^8 overflowed
    fp16 in layers with "massive activations" (Devstral layer 2: silu(gate)*up up to 282,
    x256 = 72198 > 65504) — ANE clamps to the maximum, and the chunk output turned into garbage.
    k = the largest <= kmax for which the down_proj/o_proj input (and v_proj output) x2^k <= inter_max
    and the down_proj/o_proj output x2^k <= out_max. Returns k_for(down_module) -> k."""
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
