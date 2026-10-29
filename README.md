# torch-ane

Run PyTorch models on the **Apple Neural Engine** through Apple's Core AI (macOS 27+).

Pass in a plain `nn.Module`, or a model from transformers or diffusers. torch-ane rewrites it
into a form the ANE runs fast and correctly: fp16, static shapes, and the layout and ops from
Apple's ANE guidance. It also works around the ANE's numeric quirks. You get back a module
that behaves like the original.

```python
import torch_ane

m = torch_ane.compile(model.eval(), example_inputs=(x,))  # any nn.Module -> ANE
y = m(x)                                                   # torch tensors in, torch tensors out
```

On an M4 MacBook, ANE prefill and convolutions are 2–3× faster than the GPU (MLX/MPS).
Decoding is limited by memory bandwidth, and the ANE reads ~57 GB/s.

## Install

Requires macOS 27+ on Apple silicon, and Python 3.11+.

```sh
pip install torch-ane              # core: compile()
pip install "torch-ane[llm]"       # + CausalLM / ChunkedCausalLM (transformers, gguf)
pip install "torch-ane[diffusion]" # + StableDiffusion (diffusers)
```

Core AI's Python packages (`coreai-core`, `coreai-torch`, `coreai-opt`) come from PyPI
as dependencies.

## What's inside

```python
import torch_ane

torch_ane.available()          # is Core AI usable here

# Any module. The first call per input shape builds a program (seconds); then it's cached on disk.
m = torch_ane.compile(model, example_inputs=(x,), device="ane")   # or device="gpu"

# A whole LLM on the ANE (up to ~1-3B). The KV cache is Core AI state, so it is never copied.
lm = torch_ane.CausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", batch=True)
ids = lm.generate(prompt_ids, max_new_tokens=64, eos=tok.eos_token_id)
outs = lm.generate_batch([ids1, ids2, ids3], max_new_tokens=64)   # up to 8 streams per step
logits = lm.feed(more_ids)                                        # append to the cache -> last logits

# A large model from GGUF, in 4-layer chunks with 4-bit palettes (Devstral 24B etc.)
big = torch_ane.ChunkedCausalLM.from_gguf("model.gguf", "config_dir/")

# Stable Diffusion: text encoder, UNet and VAE on the ANE
sd = torch_ane.StableDiffusion("segmind/tiny-sd")
image, timings = sd("a cute robot", steps=20, seed=0)
```

| | |
|---|---|
| `compile(model, example_inputs=None, device="ane", validate=True, compress=None)` | → `CompiledModule`. `validate` checks the first call of each shape against eager. If they disagree, that shape falls back to GPU, then to eager. `compress="int8"` stores Linear/Conv2d weights in int8 |
| `CompiledModule` | torch module over Core AI programs, one program per input shape (up to 8) |
| `CausalLM(hf_model, max_len=2048, compress=None, batch=False, ...)` | prefill in chunks of 256, decode in windows of 8, a ladder of context sizes 512…`max_len` in one program |
| `ChunkedCausalLM.from_gguf(path, config)` | 4-bit palettes, plus int8 where the GGUF is more precise (Q6_K). First build takes ~35 min for 24B |
| `StableDiffusion(model_id, device="ane", size=512)` | diffusers pipeline with all three parts on the ANE |
| `passes` | the building blocks: `export_for_ane`, `swap_linear`, `split_wide`, `chunk_mlp`, `palettize_lut`, `quantize_int8_modules`, `calibrate_rescale`, `use_composite_ops`, … |
| `CACHE_DIR`, `prune_cache()`, `clear_system_cache()` | build cache at `~/.cache/torch_ane` (`TORCH_ANE_CACHE`). It is capped at `TORCH_ANE_CACHE_GB` (40) and least recently used builds are removed. `clear_system_cache()` wipes Core AI's own compile cache, which can only be cleared as a whole |

## What it does to your model

These passes need no knowledge of the specific model:

1. RMSNorm, found by a numerical check rather than by name, becomes a Core AI composite op.
2. Everything that doesn't depend on the input (masks, RoPE tables) is precomputed in fp32.
3. A causal mask becomes `is_causal`.
4. SiLU becomes `x·0.5·(1 + tanh(x/2))`. The native ANE SiLU is off by ~5% near zero, which
   wrecks GroupNorm→SiLU in diffusion UNets.
5. The model is converted to fp16, the only float type the ANE runs.
6. MLPs whose projection weight exceeds 4 MB are split along the hidden dimension. Each chunk
   of `up` feeds its own chunk of `down`, and the results are summed. Otherwise the weight plus
   the `[S, hidden]` activation doesn't fit the ANE's on-chip memory (~32 MB on M4), and the
   ANE stalls on the bus at ~50 GB/s. This cuts the Whisper large encoder by 24% and
   Qwen2.5-0.5B prefill by 14%. Disable it with `rules=()`.

LLMs get two more passes:
- Per-projection 2^k scales, calibrated on real text, so small activations don't become fp16
  subnormals while large ones don't overflow.
- int16 positions, embeddings on the CPU, and logits for one row only.

UNet/VAE attention uses Apple's layout: BC1S, heads split, `einsum` with no transposes.

## Speed (MacBook M4, 24 GB)

| task | ANE | for comparison |
|---|---|---|
| ResNet-18, batch 16 | 4.5 ms | MLX 24.5, MPS 32.7 |
| BERT-base 8×128 | 20 ms | MLX 71, MPS 75 |
| Whisper large-v3-turbo encoder | 285 ms (int8: 264) | |
| Qwen2.5-0.5B prefill, 512 tokens | 9065 tok/s | mlx-lm 3279 |
| Qwen2.5-0.5B decode, fp16 | 54 tok/s; 8 streams: 351 | mlx-lm bf16 70 |
| Devstral Small 24B decode | 4.8 tok/s | mlx-lm 4-bit 6.7 |
| tiny-sd 512×512, 20 steps | 4.9 s | VAE 3× faster than GPU |

## Examples

- [`examples/compile_bert.py`](examples/compile_bert.py): BERT-base, eager vs ANE
- [`examples/generate.py`](examples/generate.py): chat with Qwen2.5 on the ANE
- [`examples/chat_gguf.py`](examples/chat_gguf.py): chat with a 24B GGUF model, with the engine in a restartable process
- [`examples/imagine.py`](examples/imagine.py): Stable Diffusion

## Limitations

- Shapes are static, so a new input shape means a new build.
- The ANE computes in fp16 only (weights can be palettized or int8). `-inf` in masks is
  replaced by −40000.
- int8 weights in Conv1d make the ANE compiler fail silently, and the whole function moves to
  the GPU. `compress="int8"` therefore quantizes only Linear/Conv2d.
- The coreai-core Python wrapper aborts the process after 16 384 allocated buffers
  ([apple/coreai-torch#75](https://github.com/apple/coreai-torch/issues/75)).
  `ChunkedCausalLM.steps_left()` lets you restart ahead of time; see `examples/chat_gguf.py`.
- The ANE unloads programs after ~10 s idle, which makes the next call ~3× slower. Use
  `ChunkedCausalLM.keep_warm()` to avoid this.
- Copying an input into Core AI runs at ~0.5 GB/s, about 10 ms per 4 MB input. The public API
  has no way around this.
- Code comments are in Russian.

## Tests

```sh
pip install -e ".[llm,diffusion,dev]"
pytest
```

## License

MIT
