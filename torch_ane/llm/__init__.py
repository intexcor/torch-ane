"""Языковые модели на Neural Engine.

  CausalLM         — модель из transformers целиком в одной программе (до ~1-3B):
                     prefill кусками, декод окнами, KV-кеш — состояния Core AI,
                     батч до 8 потоков за шаг (generate_batch);
  ChunkedCausalLM  — большие модели (24B) кусками по несколько слоёв, веса из GGUF,
                     палитры 4 бит / int8.
"""
from ._kv import fresh_rope
from .causal import CausalLM
from .chunked import ChunkedCausalLM

__all__ = ["CausalLM", "ChunkedCausalLM", "fresh_rope"]
