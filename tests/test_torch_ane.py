"""torch_ane: compilation for the Neural Engine, passes, LLM with the KV cache as state."""
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

import torch_ane

needs_coreai = pytest.mark.skipif(not torch_ane.available(), reason="no Core AI (macOS 27 + coreai-torch)")


def test_no_mlx_dependency():
    """The library is standalone: importing torch_ane and its LLM part does not pull in MLX or torch_mlx."""
    code = ("import sys, torch_ane; torch_ane.CausalLM; torch_ane.passes.swap_linear; "
            "print(any(m == 'mlx' or m.startswith(('mlx.', 'torch_mlx')) for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_prune_cache(tmp_path, monkeypatch):
    """The build cache stays under the limit: least recently used builds are removed, keep and
    unfinished (.partial) ones stay."""
    import os

    from torch_ane import runtime
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path)
    for i, name in enumerate(("a.aimodel", "b.aimodel", "c.aimodel", "d.partial.aimodel")):
        d = tmp_path / name
        d.mkdir()
        (d / "w").write_bytes(b"x" * 1000)
        os.utime(d, (1000 + i, 1000 + i))
    freed = runtime.prune_cache(max_gb=2.5e-6, keep=(tmp_path / "a.aimodel",))  # limit of 2500 bytes
    assert freed == 2000 and sorted(p.name for p in tmp_path.iterdir()) == ["a.aimodel", "d.partial.aimodel"]


def test_device_names():
    m = nn.Linear(4, 4)
    if torch_ane.available():
        assert torch_ane.CompiledModule(m, device="ane").device == "neural_engine"
        assert torch_ane.CompiledModule(m, device="gpu").device == "gpu"
    with pytest.raises(ValueError):
        torch_ane.runtime._device("tpu")


@needs_coreai
def test_compile_matches_eager():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).eval()
    x = torch.randn(4, 32)
    cm = torch_ane.compile(m, example_inputs=(x,))
    with torch.no_grad():
        ref = m(x)
    out = cm(x)
    assert out.dtype == torch.float32 and ((out - ref).norm() / ref.norm()) < 2e-2


@needs_coreai
def test_compile_int8_with_conv1d():
    """compress="int8" on ANE: int8 only in Linear/Conv2d. Compressing the finished program also
    quantized Conv1d: the ANE compiler failed and the model fell back to GPU with garbage output."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(16, 64, 3, padding=1)
            self.mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))

        def forward(self, x):  # [1, 16, T]
            h = torch.nn.functional.gelu(self.conv(x)).transpose(1, 2)
            return h + self.mlp(h)

    torch.manual_seed(0)
    m = Net().eval()
    x = torch.randn(1, 16, 128)
    cm = torch_ane.compile(m, example_inputs=(x,), validate=False, compress="int8")
    with torch.no_grad():
        ref = m(x)
    out = cm(x)
    if cm.ane_stats:
        assert cm.ane_stats["int8"] == 2  # two Linear, Conv1d untouched
    assert ((out - ref).norm() / ref.norm()) < 3e-2


@needs_coreai
def test_ane_passes_llm():
    """ANE passes on an LLM from transformers with no model-specific edits: RMSNorm -> composite op,
    constant folding (mask, RoPE), causal mask -> is_causal; the next token matches eager."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    class Prefill(nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.lm = lm

        def forward(self, ids):
            return self.lm.lm_head(self.lm.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1])

    torch.manual_seed(0)
    lm = Qwen2ForCausalLM(Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=1000)).eval()
    model = Prefill(lm).eval()
    ids = torch.randint(0, 1000, (1, 64))
    m = torch_ane.compile(model, validate=False)
    out = m(ids)
    with torch.no_grad():
        ref = model(ids)
    if m.ane_stats:  # passes do not run when loading from the disk cache
        assert {k: m.ane_stats[k] for k in ("rms_norm", "causal")} == {"rms_norm": 5, "causal": 2}
        assert m.ane_stats["folded"] > 0
    assert int(out.argmax()) == int(ref.argmax())
    assert ((out.float() - ref).norm() / ref.norm()) < 5e-2


def test_ane_rules_eager_equivalent():
    """Apple's ANE rules (Linear -> 1x1 conv, per-head attention) do not change the model's
    math: checked in eager, without Core AI."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from torch_ane import passes
    torch.manual_seed(0)
    lm = Qwen2ForCausalLM(Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=100)).eval()
    ids = torch.randint(0, 100, (1, 23))
    with torch.no_grad():
        ref = lm(ids).logits
        for rules in (("conv",), ("conv_n111",), ("heads",), ("conv", "heads")):
            m, st = passes.prepare(lm, rules)
            assert st.get("conv", 0) or st.get("heads", 0)
            torch.testing.assert_close(m(ids).logits, ref, rtol=1e-4, atol=1e-4)


def test_chunk_mlp_eager_equivalent():
    """MLP in chunks of the hidden dimension: Sequential (Whisper) and gate/up/down (Qwen) keep the
    same math; small MLPs are left alone."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from torch_ane import passes
    torch.manual_seed(0)
    seq = nn.Sequential(nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))).eval()
    lm = Qwen2ForCausalLM(Qwen2Config(hidden_size=64, intermediate_size=256, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=100)).eval()
    x, ids = torch.randn(3, 256), torch.randint(0, 100, (1, 11))
    with torch.no_grad():
        ref_s, ref_l = seq(x), lm(ids).logits
        m, st = passes.prepare(seq, ("mlp_chunk",))
        assert st["mlp_chunk"] == 0  # 0.5 MB weight is below the threshold
        assert passes.chunk_mlp(m, max_bytes=128 << 10) == 1 and len(m[0].up) == 4
        torch.testing.assert_close(m(x), ref_s, rtol=1e-4, atol=1e-5)
        m, _ = passes.prepare(lm, ())
        assert passes.chunk_mlp(m, max_bytes=8 << 10) == 2 and len(m.model.layers[0].mlp.up) == 4
        torch.testing.assert_close(m(ids).logits, ref_l, rtol=1e-4, atol=1e-4)
        m, _ = passes.prepare(lm, ())  # down_proj inside the scale wrapper (CausalLM)
        passes.rescale_for_fp16(m, k=4)
        assert passes.chunk_mlp(m, max_bytes=8 << 10) == 2 and m.model.layers[0].mlp.scale == 2 ** -4
        torch.testing.assert_close(m(ids).logits, ref_l, rtol=1e-4, atol=1e-4)


@needs_coreai
def test_causal_lm_matches_eager():
    """KV cache as Core AI state (Apple's scheme): chunked prefill + decode in windows of 8
    give the same logits as transformers on the full context."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import numpy as np

    torch.manual_seed(0)
    lm = Qwen2ForCausalLM(Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=1000)).eval()
    ids = torch.randint(0, 1000, (100,)).tolist()  # 37 + 63: crosses from the 64 rung to 128
    st = torch_ane.CausalLM(lm, max_len=256, prefill_q=32, ladder=(64, 128))
    st.reset()
    st._hist = []
    got = [st.feed(ids[:37])] + [st.feed([t]) for t in ids[37:]]
    with torch.no_grad():
        ref = lm(torch.tensor([ids])).logits[0, 36:].numpy()
    assert st.M == 128
    for g, r in zip(got, ref):
        assert np.linalg.norm(g - r) / np.linalg.norm(r) < 0.05


@needs_coreai
def test_causal_lm_batch_matches_single():
    """Streams in the decode window (generate_batch): each has its own positions and mask, K/V
    interleaved in a shared buffer; tokens match generating each prompt alone."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    lm = Qwen2ForCausalLM(Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=1000)).eval()
    prompts = [torch.randint(0, 1000, (n,)).tolist() for n in (5, 40, 17)]  # 40 = two prefill chunks
    st = torch_ane.CausalLM(lm, max_len=256, prefill_q=32, ladder=(128,), batch=True)
    assert st.generate_batch(prompts, 12) == [st.generate(p, 12) for p in prompts]


@needs_coreai
def test_silu_after_groupnorm():
    """Native SiLU on ANE is off near zero (~5% after GroupNorm); export_for_ane replaces it with a
    tanh form, so GroupNorm -> SiLU -> conv matches eager."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1)).eval()
    x = torch.randn(2, 64, 16, 16) * 3
    with torch.no_grad():
        ref = m(x)
    cm = torch_ane.compile(m, validate=False)
    out = cm(x)
    assert ((out - ref).norm() / ref.norm()) < 5e-3
