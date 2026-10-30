"""Core AI composite ops: SDPA and RoPE for transformers models, GatherMM (MoE),
GatedDeltaUpdate (DeltaNet)."""
import torch


def _composite_sdpa_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """attn_implementation="coreai_sdpa": attention as a single Core AI composite op.
    The mask is boolean (True — visible): the composite SDPA doesn't understand an additive one;
    GQA — handled itself."""
    keep = None if attention_mask is None else (attention_mask > -1.0 if attention_mask.is_floating_point()
                                                 else attention_mask)
    out = module._coreai_sdpa(query, key, value, attn_mask=keep)
    return out.transpose(1, 2).contiguous(), None


class CompositeSwitchMLP(torch.nn.Module):
    """MoE expert layer on the Core AI composite GatherMM: only the selected experts are taken from the shared
    weight tensor by the router indices (input/weight shapes as in Apple's switch.py).
    On ANE it builds as a fused kernel, but on a microbenchmark it's slower than GPU."""

    def __init__(self, num_experts, din, dout, bias=False):
        super().__init__()
        from coreai_torch.composite_ops import GatherMM
        self.gather_mm = GatherMM(num_batch_axes=1)
        # weights in float32: the type is cast to the input type in forward, and passes.prepare converts all to fp16
        self.weight = torch.nn.Parameter(torch.randn(1, num_experts, dout, din) * din ** -0.5)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, num_experts, dout))

    def forward(self, x, indices):  # x: [T,1,1,din], indices: [T, active experts] int
        from coreai_torch.composite_ops import _gather_mm
        w = self.weight.to(x.dtype).transpose(-1, -2)  # weight cast to input type: fp16 (probe) or fp32 (prepare)
        y = self.gather_mm(x, w, rhs_indices=indices)
        if hasattr(self, "bias"):
            b = _gather_mm._gather(self.bias.to(x.dtype), indices, num_batch_axes=1)
            y = y + b.unsqueeze(-2)
        return y


class CompositeGatedDelta(torch.nn.Module):
    """DeltaNet state update (Qwen3-Next/Qwen3.5) on the composite GatedDeltaUpdate.
    The state [B,H,D,D] doesn't grow with context. It builds on ANE, but inside there's a loop over
    positions — on long context slower than GPU; for MoE/DeltaNet GPU is preferable."""

    def __init__(self, use_qk_l2_norm=True):
        super().__init__()
        from coreai_torch.composite_ops import GatedDeltaUpdate
        self.gdu = GatedDeltaUpdate(use_qk_l2_norm=use_qk_l2_norm)

    def forward(self, query, key, value, g, beta, initial_state):
        return self.gdu(query, key, value, g, beta, initial_state)


def use_composite_ops(model, ops=("sdpa", "rope")):
    """Attach Core AI composite ops to the model. SDPA/RoPE substitute calls in the HF model;
    gather/gdu only declare specs — the CompositeSwitchMLP/CompositeGatedDelta layers are set by the
    model itself. Returns a list of ExternalizeSpec for the converter (RMSNorm is attached anyway)."""
    import inspect

    from coreai_torch import ExternalizeSpec
    from coreai_torch.composite_ops import GatedDeltaUpdate, GatherMM, RoPE, SDPA
    specs = []
    if "sdpa" in ops:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        AttentionInterface.register("coreai_sdpa", _composite_sdpa_attention)
        AttentionMaskInterface.register("coreai_sdpa", lambda *a, **k: sdpa_mask(*a, **{**k, "allow_is_causal_skip": False}))
        for mod in model.modules():
            cfg = getattr(mod, "config", None)
            if cfg is not None and getattr(cfg, "_attn_implementation", None) in ("sdpa", "eager"):
                cfg._attn_implementation = "coreai_sdpa"
            if hasattr(mod, "scaling") and hasattr(mod, "o_proj"):  # attention module
                mod._coreai_sdpa = SDPA(scale=float(mod.scaling))
        specs.append(ExternalizeSpec(target_class=SDPA, composite_op_name="scaled_dot_product_attention",
                                     composite_attrs=["scale", "is_causal", "window_size"]))
    if "rope" in ops:
        # the composite RoPE computes the rotation itself from position_ids. It also works via cos/sin,
        # but then HF's rotary_emb computes them in float32 inside the graph — and ANE doesn't take float32
        # (the I/O-cast validator rejects it), and the op falls back to GPU. position_ids mode puts it on ANE.
        cfg = model.config
        base = float(cfg.rope_parameters["rope_theta"]) if getattr(cfg, "rope_parameters", None) \
            else float(getattr(cfg, "rope_theta", 1e4))
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        rope = RoPE(base=base, dims=head_dim)
        model._coreai_rope = rope
        # rotary_emb no longer computes cos/sin — pass position_ids (int16) through the cos slot
        rot = getattr(getattr(model, "model", None), "rotary_emb", None)
        if rot is not None:
            rot.forward = lambda x, position_ids, **_k: (position_ids, position_ids)
        attn = next(m for m in model.modules() if hasattr(m, "o_proj") and hasattr(m, "q_proj"))
        mm = inspect.getmodule(type(attn))
        if hasattr(mm, "apply_rotary_pos_emb"):
            def apply(q, k, cos, sin, unsqueeze_dim=1, _rope=rope):
                pos = cos  # position_ids are passed in here
                return _rope(q, position_ids=pos), _rope(k, position_ids=pos)
            mm.apply_rotary_pos_emb = apply
            specs.append(ExternalizeSpec(target_class=RoPE, composite_op_name="rope",
                                         composite_attrs=["scale", "base", "dims", "interleaved"]))
    if "gather" in ops:  # MoE: CompositeSwitchMLP layers are already in the model, only the spec here
        specs.append(ExternalizeSpec(target_class=GatherMM, composite_op_name="gather_mm",
                                     composite_attrs=["num_batch_axes"]))
    if "gdu" in ops:  # DeltaNet: CompositeGatedDelta layers are already in the model
        specs.append(ExternalizeSpec(target_class=GatedDeltaUpdate, composite_op_name="gated_delta_update",
                                     composite_attrs=[]))
    return specs
