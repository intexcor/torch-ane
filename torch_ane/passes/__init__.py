"""Passes over torch models for the Neural Engine — the low level of torch_ane.

  export     — export for Core AI: RMSNorm -> composite op, constant folding,
               causal mask -> is_causal, SiLU via tanh, conversion to fp16;
  layout     — Apple's rules: Linear -> 1x1 conv (BC1S), per-head attention,
               splitting overly wide projections, MLP in chunks to fit ANE on-chip memory;
  quant      — palettes (lut) and int8 per Apple's recipe, scale against subnormal fp16;
  composite  — Core AI composite SDPA/RoPE/GatherMM/GatedDeltaUpdate.
"""
from .composite import CompositeGatedDelta, CompositeSwitchMLP, use_composite_ops
from .export import (DEFAULT_RULES, PASSES_VERSION, ANERMSNorm, _to_fp16, causal_mask_to_flag, export_for_ane,
                     fold_constants, prepare, silu_to_tanh, swap_rmsnorm)
from .layout import (MASK_NEG, ChunkedMLP, ConvLinear, SplitConvLinear, ane_attention, chunk_mlp, split_wide,
                     swap_linear, use_ane_attention)
from .quant import (calibrate_rescale, palettize_lut, prequantize_lut, quantize_int8_modules,
                    rescale_for_fp16)

__all__ = [
    "export_for_ane", "prepare", "swap_rmsnorm", "fold_constants", "causal_mask_to_flag", "silu_to_tanh",
    "ANERMSNorm", "DEFAULT_RULES", "PASSES_VERSION",
    "swap_linear", "split_wide", "use_ane_attention", "ane_attention", "ConvLinear", "SplitConvLinear", "MASK_NEG",
    "chunk_mlp", "ChunkedMLP",
    "palettize_lut", "prequantize_lut", "quantize_int8_modules", "rescale_for_fp16", "calibrate_rescale",
    "use_composite_ops", "CompositeSwitchMLP", "CompositeGatedDelta",
]
