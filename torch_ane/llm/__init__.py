"""Language models on the Neural Engine.

  CausalLM         — a transformers model entirely in one program (up to ~1-3B):
                     chunked prefill, windowed decode, KV cache as Core AI states,
                     batch of up to 8 streams per step (generate_batch);
  ChunkedCausalLM  — large models (24B) in chunks of several layers, weights from GGUF,
                     4-bit palettes / int8.
"""
from ._kv import fresh_rope
from .causal import CausalLM
from .chunked import ChunkedCausalLM

__all__ = ["CausalLM", "ChunkedCausalLM", "fresh_rope"]
