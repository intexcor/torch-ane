"""Проходы над torch-моделями под Neural Engine — нижний уровень torch_ane.

  export     — экспорт для Core AI: RMSNorm -> составная операция, свёртка констант,
               причинная маска -> is_causal, SiLU через tanh, перевод в fp16;
  layout     — правила Apple: Linear -> свёртка 1x1 (BC1S), внимание по головам,
               деление слишком широких проекций, MLP кусками под внутреннюю память ANE;
  quant      — палитры (lut) и int8 по рецепту Apple, масштаб против субнормальных fp16;
  composite  — составные SDPA/RoPE/GatherMM/GatedDeltaUpdate Core AI.
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
