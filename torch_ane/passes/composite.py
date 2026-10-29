"""Составные операции Core AI: SDPA и RoPE для моделей transformers, GatherMM (MoE),
GatedDeltaUpdate (DeltaNet)."""
import torch


def _composite_sdpa_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """attn_implementation="coreai_sdpa": внимание одной составной операцией Core AI.
    Маска — логическая (True — видно): аддитивную составной SDPA не понимает; GQA — сам."""
    keep = None if attention_mask is None else (attention_mask > -1.0 if attention_mask.is_floating_point()
                                                 else attention_mask)
    out = module._coreai_sdpa(query, key, value, attn_mask=keep)
    return out.transpose(1, 2).contiguous(), None


class CompositeSwitchMLP(torch.nn.Module):
    """Экспертный слой MoE на составном GatherMM Core AI: из общего тензора весов по
    индексам маршрутизатора достаются только выбранные эксперты (форма входа/весов — как у
    Apple в switch.py). На ANE собирается слитым ядром, но на микротесте медленнее GPU."""

    def __init__(self, num_experts, din, dout, bias=False):
        super().__init__()
        from coreai_torch.composite_ops import GatherMM
        self.gather_mm = GatherMM(num_batch_axes=1)
        # веса в float32: тип приводится к типу входа в forward, а в fp16 всё переводит passes.prepare
        self.weight = torch.nn.Parameter(torch.randn(1, num_experts, dout, din) * din ** -0.5)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, num_experts, dout))

    def forward(self, x, indices):  # x: [T,1,1,din], indices: [T, активных экспертов] int
        from coreai_torch.composite_ops import _gather_mm
        w = self.weight.to(x.dtype).transpose(-1, -2)  # вес под тип входа: fp16 (проба) или fp32 (prepare)
        y = self.gather_mm(x, w, rhs_indices=indices)
        if hasattr(self, "bias"):
            b = _gather_mm._gather(self.bias.to(x.dtype), indices, num_batch_axes=1)
            y = y + b.unsqueeze(-2)
        return y


class CompositeGatedDelta(torch.nn.Module):
    """Обновление состояния DeltaNet (Qwen3-Next/Qwen3.5) на составном GatedDeltaUpdate.
    Состояние [B,H,D,D] не растёт с контекстом. На ANE собирается, но внутри цикл по
    позициям — на длинном контексте медленнее GPU; для MoE/DeltaNet предпочтителен GPU."""

    def __init__(self, use_qk_l2_norm=True):
        super().__init__()
        from coreai_torch.composite_ops import GatedDeltaUpdate
        self.gdu = GatedDeltaUpdate(use_qk_l2_norm=use_qk_l2_norm)

    def forward(self, query, key, value, g, beta, initial_state):
        return self.gdu(query, key, value, g, beta, initial_state)


def use_composite_ops(model, ops=("sdpa", "rope")):
    """Подключить составные операции Core AI к модели. SDPA/RoPE подменяют вызовы в HF-модели;
    gather/gdu лишь объявляют спеки — слои CompositeSwitchMLP/CompositeGatedDelta ставит сама
    модель. Возвращает список ExternalizeSpec для конвертера (RMSNorm подключается и так)."""
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
            if hasattr(mod, "scaling") and hasattr(mod, "o_proj"):  # модуль внимания
                mod._coreai_sdpa = SDPA(scale=float(mod.scaling))
        specs.append(ExternalizeSpec(target_class=SDPA, composite_op_name="scaled_dot_product_attention",
                                     composite_attrs=["scale", "is_causal", "window_size"]))
    if "rope" in ops:
        # составной RoPE считает поворот сам из position_ids. Через cos/sin он тоже работает,
        # но тогда их вычисляет rotary_emb HF в float32 внутри графа — а float32 ANE не берёт
        # (I/O-каст валидатор режет), и операция падает на GPU. Режим position_ids кладёт её на ANE.
        cfg = model.config
        base = float(cfg.rope_parameters["rope_theta"]) if getattr(cfg, "rope_parameters", None) \
            else float(getattr(cfg, "rope_theta", 1e4))
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        rope = RoPE(base=base, dims=head_dim)
        model._coreai_rope = rope
        # rotary_emb больше не считает cos/sin — пробрасываем position_ids (int16) в слот cos
        rot = getattr(getattr(model, "model", None), "rotary_emb", None)
        if rot is not None:
            rot.forward = lambda x, position_ids, **_k: (position_ids, position_ids)
        attn = next(m for m in model.modules() if hasattr(m, "o_proj") and hasattr(m, "q_proj"))
        mm = inspect.getmodule(type(attn))
        if hasattr(mm, "apply_rotary_pos_emb"):
            def apply(q, k, cos, sin, unsqueeze_dim=1, _rope=rope):
                pos = cos  # сюда проброшены position_ids
                return _rope(q, position_ids=pos), _rope(k, position_ids=pos)
            mm.apply_rotary_pos_emb = apply
            specs.append(ExternalizeSpec(target_class=RoPE, composite_op_name="rope",
                                         composite_attrs=["scale", "base", "dims", "interleaved"]))
    if "gather" in ops:  # MoE: слои CompositeSwitchMLP уже в модели, здесь только спек
        specs.append(ExternalizeSpec(target_class=GatherMM, composite_op_name="gather_mm",
                                     composite_attrs=["num_batch_axes"]))
    if "gdu" in ops:  # DeltaNet: слои CompositeGatedDelta уже в модели
        specs.append(ExternalizeSpec(target_class=GatedDeltaUpdate, composite_op_name="gated_delta_update",
                                     composite_attrs=[]))
    return specs
