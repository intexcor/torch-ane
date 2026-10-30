"""torch_ane — PyTorch models on the Apple Neural Engine via Core AI (macOS 27+).

    import torch_ane

    m = torch_ane.compile(model.eval(), example_inputs=(x,))     # any nn.Module -> ANE
    y = m(x)

    lm = torch_ane.CausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ids = lm.generate(prompt_ids, max_new_tokens=64)

    sd = torch_ane.StableDiffusion("segmind/tiny-sd")
    image, timings = sd("a cute robot", steps=20, seed=0)

Lower level — torch_ane.passes (export passes, Apple layout, palettes/int8,
composite ops). Builds are cached in torch_ane.CACHE_DIR (~/.cache/torch_ane,
variable TORCH_ANE_CACHE). The library does not depend on MLX.
"""
from . import passes
from .runtime import (CACHE_DIR, CompiledModule, available, cleanup_mpsgraph_tmp, clear_system_cache, compile,
                      compress_program, model_digest, prune_cache)

__all__ = ["compile", "CompiledModule", "available", "compress_program", "model_digest", "cleanup_mpsgraph_tmp",
           "CACHE_DIR", "prune_cache", "clear_system_cache", "passes", "CausalLM", "ChunkedCausalLM", "StableDiffusion"]


def __getattr__(name):
    # heavy parts (transformers, diffusers) — only on access
    if name in ("CausalLM", "ChunkedCausalLM"):
        from . import llm
        return getattr(llm, name)
    if name == "StableDiffusion":
        from .diffusion import StableDiffusion
        return StableDiffusion
    raise AttributeError(f"module 'torch_ane' has no attribute {name!r}")
