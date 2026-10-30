"""Passes that make an arbitrary torch model suitable for the Neural Engine.

ANE executes only fp16 (and int8). Models from transformers don't run on it as is:
  * RMSNorm internally goes to float32 (`hidden_states.to(torch.float32)`);
  * the attention mask is built from int64/bool comparisons, RoPE (cos/sin) in float32.
Core AI splits such a graph between ANE/GPU/CPU — slow and (coreai-torch #115) silently
wrong. So before conversion, without knowledge of the specific model:

  1. swap_rmsnorm: modules that BEHAVE like RMSNorm (checked numerically,
     not by name) are replaced with RMSNormImpl — a Core AI composite op:
     the compiler substitutes the native kernel, there's no fp32 branch in the graph;
  2. fold_constants: everything that depends on neither the input nor the weights (mask, RoPE
     tables) is precomputed in fp32 and put into the graph as constants (like freezing
     in Inductor and constant folding in XLA) — with a static shape that's all of it;
  3. causal_mask: a folded SDPA mask numerically equal to the causal one is replaced
     with the flag is_causal=True;
  4. only then are the model and constants converted to fp16 — RoPE precision isn't lost.
"""
import copy

import torch

from .layout import chunk_mlp, swap_linear, use_ane_attention


aten = torch.ops.aten
# Apple's rules (below) are off by default. Measured on M4, macOS 27.0.1, Qwen2.5-0.5B: the
# Core AI compiler puts SDPA and Linear on ANE by itself; 1x1 conv is on par (+0..7%); per-head
# attention is 1.1-2.6x slower than fused SDPA even in a clean BC1S layout, and from 12 layers on
# ANE also silently zeroes the outputs after the last layer.
# One rule is on — "mlp_chunk" (MLP in chunks to fit ANE on-chip memory, passes.chunk_mlp):
# Whisper large-v3-turbo encoder 368 -> 280 ms, a Qwen2.5-0.5B pass on 256 tokens
# 30.5 -> 26.3 ms, on 8 tokens on par.
DEFAULT_RULES = ("mlp_chunk",)


# --- 1. RMSNorm -> Core AI composite op -----------------------------------------

class ANERMSNorm(torch.nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        from coreai_torch.composite_ops import RMSNormImpl
        self.weight = weight  # the same Parameter, not a copy
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
    """Replaces in place; returns the number of replacements."""
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


# --- 2–3. constant folding and causal mask (on the unrolled graph) -------------

def fold_constants(gm, param_names):
    """Nodes that depend on neither placeholders nor parameters are computed and
    replaced with constants (gm buffers). Returns the number of folded frontier nodes."""
    g = gm.graph
    live = set()
    for n in g.nodes:
        if n.op == "placeholder" or (n.op == "get_attr" and n.target in param_names):
            live.add(n)
        elif n.op == "call_function" and str(n.target).startswith("coreai."):
            # weight decompression (coreai.lut_to_dense etc.) — don't fold: otherwise compressed
            # weights silently turn back into dense constants
            live.add(n)
        elif n.op == "call_function" and (any(a in live for a in n.all_input_nodes)
                                          or n.target is aten._assert_tensor_metadata.default
                                          or getattr(n.target, "_schema", None) is not None
                                          and n.target._schema.is_mutable):
            live.add(n)
        elif n.op == "output":
            live.add(n)
    foldable = [n for n in g.nodes if n.op in ("call_function", "get_attr") and n not in live]
    # frontier: a foldable node whose result is needed by the "live" part of the graph
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
    """SDPA attn_mask — a constant matching the causal one -> is_causal=True."""
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
        else:  # additive 0 / -inf mask
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
    """Parameters, buffers and folded constants -> fp16; explicit casts to fp32
    inside the graph (except a cast straight into the output) -> fp16 too."""
    # export's internal checks like "this tensor is fp32" are false after conversion to fp16
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target is aten._assert_tensor_metadata.default:
            gm.graph.erase_node(n)
    for name, t in list(gm.named_buffers()) + list(gm.named_parameters()):
        if t.is_floating_point():
            t.data = t.data.half()
    # the graph input stays in its original type (after extracting modules the converter itself
    # re-exports the graph with the original inputs), the cast to fp16 is the first node:
    # ANE supports an F32 <-> F16 cast at input/output
    # placeholders of the unrolled graph have no meta["val"] — types are taken from the example
    # inputs (placeholders come in the same order as the arguments)
    for ph, v in zip([n for n in gm.graph.nodes if n.op == "placeholder"], example_args):
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            with gm.graph.inserting_after(ph):
                cast = gm.graph.call_function(aten._to_copy.default, (ph,), {"dtype": torch.float16})
            ph.replace_all_uses_with(cast)
            cast.args = (ph,)
    out_args = set(next(n for n in gm.graph.nodes if n.op == "output").all_input_nodes)
    # implicit float32 from integer inputs (division of positions etc. — Llama 4 attention scale
    # in Ministral3): the node has no explicit dtype, so a cast to fp16 is inserted after it
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


# --- build ---------------------------------------------------------------------

# version of the export_for_ane passes — part of the build cache keys (when it changes, builds are rebuilt)
PASSES_VERSION = "s1"  # s1: SiLU via tanh


def silu_to_tanh(gm):
    """SiLU -> x·0.5·(1 + tanh(x/2)). Native SiLU on ANE is wrong near zero: at the
    GroupNorm output (|x| < 2) ~5% relative error, because of this the tiny-sd UNet diverged from
    eager by 2.5e-2 per call. The compiler recognizes the form x·sigmoid(x) and substitutes the same
    SiLU; via tanh it doesn't and computes exactly (7.8e-4). Returns the number of replacements."""
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
    """export_fn for TorchConverter.add_pytorch_module: fp32 export -> constant
    folding -> causal mask -> fp16 -> re-export with decompositions."""
    from coreai_torch import get_decomp_table

    def export_fn(m):
        # inputs are fp16 (fed by _HalfInputs from runtime.py, inside they're cast to fp32,
        # after _to_fp16 this cast disappears): ANE computes an fp32 input with a cast straight into an op
        # incorrectly ("ANE I/O op can only do F16 MemRef <-> F32 Tensor cast")
        ep = torch.export.export(m, args).run_decompositions(get_decomp_table())
        # a copy: ep.module() shares parameters with the model, and _to_fp16 changes them in place —
        # the converter later exports the original model with fp32 inputs once more
        gm = copy.deepcopy(ep.module())
        params = {n for n, _ in gm.named_parameters()}
        # mutable buffers (KV cache as model state) are not constants
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
    """A copy of the model for ANE. rules — a set of Apple's rules: "conv" (Linear -> 1x1 conv,
    bc1s layout), "conv_n111" (the same, tokens along the batch), "heads" (per-head
    attention), "mlp_chunk" (MLP in chunks of the hidden dimension to fit ANE on-chip memory).
    Returns (model, stats)."""
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
