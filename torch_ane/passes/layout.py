"""Layout per Apple's rules for ANE: 1x1 convs instead of Linear, per-head attention,
splitting overly wide projections."""
import torch


# --- Apple's rules for ANE (apple/coreai-models, neural_engine_rules.md) ----------
#
#  * "Conv2d instead of Linear": nn.Linear on ANE is lowered into less
#    efficient ops, a 1x1 conv goes straight into the conv engine;
#  * "Per-head attention": ANE has no fused SDPA — heads are computed
#    separately, Q@K — einsum "bchq,bkhc->bkhq" (no transposes),
#    softmax along the channel axis (keys), mask in layout (1, K, 1, Q);
#  * mask -40000, not -inf: ANE computes softmax with -inf incorrectly.

MASK_NEG = -40000.0


class ConvLinear(torch.nn.Module):
    """nn.Linear as a 1x1 conv. layout="bc1s": tokens along the width — (1, I, 1, N),
    as the ANE layout (B, C, 1, S) requires; "n111": tokens along the batch — (N, I, 1, 1),
    as in the MLP from Apple's examples."""

    def __init__(self, lin, layout="bc1s"):
        super().__init__()
        # a real nn.Conv2d: coreai-opt compression recognizes it (palettes on ANE are for convs)
        self.conv = torch.nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
        with torch.no_grad():
            self.conv.weight.copy_(lin.weight[:, :, None, None])
            if lin.bias is not None:
                self.conv.bias.copy_(lin.bias)
        self.layout = layout
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        if self.layout == "bc1s":
            y = self.conv(x.t()[None, :, None, :])  # (1, O, 1, N)
            y = y[0, :, 0].t()
        else:
            y = self.conv(x[:, :, None, None])[:, :, 0, 0]
        return y.reshape(*lead, self.out_features)


def swap_linear(model, layout="bc1s", skip=("lm_head",)):
    """All nn.Linear -> ConvLinear (except lm_head and huge outputs: a conv with
    O=150k channels gains nothing on ANE and bloats compilation)."""
    count = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if cname in skip:
                continue
            if type(child) is torch.nn.Linear and child.out_features <= 65536:
                setattr(mod, cname, ConvLinear(child, layout))
                count += 1
    return count


def _ane_mask(mask, B, H, Q, K, causal):
    """A transformers mask (bool "visible" or additive, (B|1, 1|H, Q, K)) or None ->
    an additive fp mask (B, K, 1, Q) with -40000."""
    if mask is None:
        if not causal or Q == 1:
            return None
        mask = torch.ones(Q, K, dtype=torch.bool).tril(K - Q)[None, None]
    if mask.dtype == torch.bool:
        add = torch.where(mask, 0.0, MASK_NEG)
    else:
        add = mask.float().clamp(min=MASK_NEG)
    add = add[..., :K]
    return add[:, 0].transpose(1, 2)[:, :, None, :]  # (B, K, 1, Q)


def ane_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """Apple-style attention for transformers: per head, einsum in the ANE layout.
    query (B, H, Q, D), key/value (B, Hkv, K, D) -> ((B, Q, H, D), None)."""
    B, H, Q, D = query.shape
    Hkv, K = key.shape[1], key.shape[2]
    group = H // Hkv
    scale = D ** -0.5 if scaling is None else scaling
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    mask = _ane_mask(attention_mask, B, H, Q, K, causal)
    # scale is applied to Q before the matmul: in Qwen2.5 etc. unscaled Q·K overflows fp16
    q = (query * scale).permute(0, 1, 3, 2)[:, :, :, None, :]   # (B, H, D, 1, Q): head -> bchq
    k = key[:, :, :, None, :]                          # (B, Hkv, K, 1, D): bkhc
    v = value[:, :, :, None, :]
    outs = []
    for h in range(H):
        kv = h // group
        w = torch.einsum("bchq,bkhc->bkhq", q[:, h], k[:, kv])   # (B, K, 1, Q)
        if mask is not None:
            w = w + mask.to(w.dtype)
        w = torch.softmax(w, dim=1)
        outs.append(torch.einsum("bkhq,bkhc->bchq", w, v[:, kv]))       # (B, D, 1, Q)
    out = torch.cat(outs, dim=1)                                          # (B, H*D, 1, Q)
    out = out.reshape(B, H, D, Q).permute(0, 3, 1, 2)                     # (B, Q, H, D)
    return out, None


_REGISTERED = False


def _register_attention():
    """attn_implementation="ane" for transformers models (the mask is as for sdpa, but
    always a tensor, without the "is_causal" skip)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    def mask_fn(*args, **kwargs):
        kwargs["allow_is_causal_skip"] = False
        return sdpa_mask(*args, **kwargs)

    AttentionInterface.register("ane", ane_attention)
    AttentionMaskInterface.register("ane", mask_fn)
    _REGISTERED = True


def use_ane_attention(model):
    """Switch all transformers configs inside the model to "ane" attention."""
    try:
        _register_attention()
    except ImportError:
        return 0
    count = 0
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation") and cfg._attn_implementation != "ane":
            if cfg._attn_implementation not in ("sdpa", "eager"):
                continue
            cfg._attn_implementation = "ane"
            count += 1
    return count


class SplitConvLinear(torch.nn.Module):
    """A ConvLinear wider than the ANE per-dimension limit — several convs of limit channels each:
    along the output (concatenation) or along the input (sum). The same math."""

    def __init__(self, cl, limit=16384):
        super().__init__()
        w = cl.conv.weight.detach()
        b = None if cl.conv.bias is None else cl.conv.bias.detach()
        self.layout, self.in_features, self.out_features = cl.layout, cl.in_features, cl.out_features
        self.by_out = cl.out_features > limit
        self.parts = torch.nn.ModuleList()
        n = cl.out_features if self.by_out else cl.in_features
        for s0 in range(0, n, limit):
            ws = w[s0:s0 + limit] if self.by_out else w[:, s0:s0 + limit]
            conv = torch.nn.Conv2d(ws.shape[1], ws.shape[0], 1, bias=b is not None and (self.by_out or s0 == 0))
            with torch.no_grad():
                conv.weight.copy_(ws)
                if conv.bias is not None:
                    conv.bias.copy_(b[s0:s0 + limit] if self.by_out else b)
            self.parts.append(conv)
        self.limit = limit

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, self.in_features).t()[None, :, None, :]  # (1, I, 1, N)
        if self.by_out:
            y = torch.cat([c(x) for c in self.parts], dim=1)
        else:
            y = sum(c(x[:, i * self.limit:(i + 1) * self.limit]) for i, c in enumerate(self.parts))
        return y[0, :, 0].t().reshape(*lead, self.out_features)


def split_wide(model, limit=16384):
    """ConvLinear with a dimension larger than limit -> SplitConvLinear (ANE limit ~16384)."""
    count = 0
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, ConvLinear) and child.layout == "bc1s" and max(child.in_features,
                                                                                 child.out_features) > limit:
                setattr(mod, name, SplitConvLinear(child, limit))
                count += 1
    return count


# --- splitting the MLP along the hidden dimension to fit ANE on-chip memory --------
#
# A Linear weight larger than a few MB together with the intermediate activation [S, hidden] doesn't
# fit into ANE on-chip memory (~32 MB on M4, per maderix/ANE), and ANE at any
# number of positions is bound by the bus (~50 GB/s). Whisper large MLP 1280 -> 5120 -> 1280 at
# 1500 positions: 6.6 TFLOP/s; hidden dimension in chunks of 3.3 MB of weight, each up chunk
# straight into its own down chunk, summed — 9.5.
# Splitting each Linear separately (concatenation after up) doesn't help: the activation
# [S, hidden] is still assembled in full.

class ChunkedMLP(torch.nn.Module):
    """MLP with chunks of the hidden dimension: y = sum_i down_i(act(up_i(x)) [* gate part]).
    The same math, partial results summed in chunk order."""

    def __init__(self, up, down, act, gate=None, k=4, scale=1.0):
        super().__init__()
        self.scale = float(scale)  # multiplier after the sum (_ScaleOut from rescale_for_fp16)
        hidden = up.out_features
        step = hidden // k

        def part(lin, i, by_out, bias):
            w = lin.weight.detach()
            ws = w[i * step:(i + 1) * step] if by_out else w[:, i * step:(i + 1) * step]
            out = torch.nn.Linear(ws.shape[1], ws.shape[0], bias=bias and lin.bias is not None,
                                  dtype=w.dtype, device=w.device)
            with torch.no_grad():
                out.weight.copy_(ws)
                if out.bias is not None:
                    out.bias.copy_(lin.bias.detach()[i * step:(i + 1) * step] if by_out else lin.bias.detach())
            return out

        self.up = torch.nn.ModuleList(part(up, i, True, True) for i in range(k))
        self.down = torch.nn.ModuleList(part(down, i, False, i == 0) for i in range(k))
        self.gate = None if gate is None else torch.nn.ModuleList(part(gate, i, True, True) for i in range(k))
        self.act = act

    def forward(self, x):
        y = None
        for i, (u, d) in enumerate(zip(self.up, self.down)):
            h = self.act(u(x)) if self.gate is None else self.act(self.gate[i](x)) * u(x)
            p = d(h)
            y = p if y is None else y + p
        return y if self.scale == 1.0 else y * self.scale


def _mlp_chunks(hidden, in_features, max_bytes):
    """Number of chunks: the up chunk weight (fp16) is at most max_bytes, the chunk is a multiple of 64."""
    for k in range(1, hidden // 64 + 1):
        if hidden % k == 0 and (hidden // k) % 64 == 0 and hidden // k * in_features * 2 <= max_bytes:
            return k
    return 1


def _halves(proj):
    """A Linear with output [hidden | gate] (diffusers GEGLU) -> two Linears over the halves."""
    n = proj.out_features // 2
    out = []
    for i in range(2):
        lin = torch.nn.Linear(proj.in_features, n, bias=proj.bias is not None, dtype=proj.weight.dtype,
                              device=proj.weight.device)
        with torch.no_grad():
            lin.weight.copy_(proj.weight[i * n:(i + 1) * n])
            if lin.bias is not None:
                lin.bias.copy_(proj.bias[i * n:(i + 1) * n])
        out.append(lin)
    return out


class _Gelu(torch.nn.Module):
    """diffusers GEGLU/GELU activation: F.gelu with their approximate."""

    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate=self.approximate)


def _find_mlp(child):
    """(up, act, down, gate) for a recognizable MLP, or None."""
    L = torch.nn.Linear
    if (isinstance(child, torch.nn.Sequential) and len(child) == 3 and isinstance(child[0], L)
            and isinstance(child[2], L) and not isinstance(child[1], L)
            and child[0].out_features == child[2].in_features):  # Whisper, ViT
        return child[0], child[1], child[2], None
    if hasattr(child, "act_fn") and all(isinstance(getattr(child, a, None), L) for a in ("gate_proj", "up_proj")):
        return child.up_proj, child.act_fn, child.down_proj, child.gate_proj  # LLaMA/Qwen
    if hasattr(child, "activation_fn") and isinstance(getattr(child, "fc1", None), L) \
            and isinstance(getattr(child, "fc2", None), L):  # CLIP
        return child.fc1, child.activation_fn, child.fc2, None
    net = getattr(child, "net", None)  # diffusers FeedForward: [GEGLU | GELU, Dropout, Linear, (Dropout)]
    if isinstance(net, torch.nn.ModuleList) and isinstance(getattr(net[0], "proj", None), L):
        lins = [m for m in net if isinstance(m, L)]
        if len(lins) != 1 or any(not isinstance(m, (L, torch.nn.Dropout)) for m in list(net)[1:]):
            return None
        kind = type(net[0]).__name__
        if kind == "GEGLU":
            hidden, gate = _halves(net[0].proj)
            return hidden, _Gelu(getattr(net[0], "approximate", "none")), lins[0], gate
        if kind == "GELU":
            return net[0].proj, _Gelu(getattr(net[0], "approximate", "none")), lins[0], None
    return None


def chunk_mlp(model, max_bytes=4 << 20):
    """MLP with a projection weight larger than max_bytes -> ChunkedMLP. Recognizes nn.Sequential
    (Linear, activation, Linear) (Whisper, ViT), gate_proj/up_proj/down_proj + act_fn
    (LLaMA/Qwen), fc1/fc2 + activation_fn (CLIP) and diffusers FeedForward with GEGLU/GELU
    (UNet); a down_proj in a scale wrapper (rescale_for_fp16) is unwrapped. Returns
    the number of replacements."""
    count = 0
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            found = _find_mlp(child)
            if found is None:
                continue
            up, act, down, gate = found
            scale = 1.0
            if not isinstance(down, torch.nn.Linear):  # _ScaleOut: inner Linear and multiplier s
                if not isinstance(getattr(down, "inner", None), torch.nn.Linear):
                    continue
                down, scale = down.inner, down.s
            k = _mlp_chunks(up.out_features, up.in_features, max_bytes)
            if k > 1:
                setattr(mod, name, ChunkedMLP(up, down, act, gate, k, scale))
                count += 1
    return count
