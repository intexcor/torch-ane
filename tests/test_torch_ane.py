"""torch_ane: компиляция на Neural Engine, проходы, LLM с KV-кешем-состоянием."""
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

import torch_ane

needs_coreai = pytest.mark.skipif(not torch_ane.available(), reason="нет Core AI (macOS 27 + coreai-torch)")


def test_no_mlx_dependency():
    """Библиотека самостоятельна: импорт torch_ane и его LLM-части не тянет MLX и torch_mlx."""
    code = ("import sys, torch_ane; torch_ane.CausalLM; torch_ane.passes.swap_linear; "
            "print(any(m == 'mlx' or m.startswith(('mlx.', 'torch_mlx')) for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_prune_cache(tmp_path, monkeypatch):
    """Кеш сборок не больше лимита: удаляются давно не использованные, keep и
    недописанные (.partial) остаются."""
    import os

    from torch_ane import runtime
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path)
    for i, name in enumerate(("a.aimodel", "b.aimodel", "c.aimodel", "d.partial.aimodel")):
        d = tmp_path / name
        d.mkdir()
        (d / "w").write_bytes(b"x" * 1000)
        os.utime(d, (1000 + i, 1000 + i))
    freed = runtime.prune_cache(max_gb=2.5e-6, keep=(tmp_path / "a.aimodel",))  # лимит 2500 байт
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
    """compress="int8" на ANE: int8 только в Linear/Conv2d. Сжатие готовой программы
    квантовало и Conv1d — компилятор ANE падал, модель уходила на GPU с мусором на выходе."""
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
        assert cm.ane_stats["int8"] == 2  # две Linear, Conv1d не тронута
    assert ((out - ref).norm() / ref.norm()) < 3e-2


@needs_coreai
def test_ane_passes_llm():
    """ANE-проходы на LLM из transformers без правок под модель: RMSNorm -> составная операция,
    свёртка констант (маска, RoPE), причинная маска -> is_causal; следующий токен как у eager."""
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
    if m.ane_stats:  # при загрузке из дискового кеша проходы не запускаются
        assert {k: m.ane_stats[k] for k in ("rms_norm", "causal")} == {"rms_norm": 5, "causal": 2}
        assert m.ane_stats["folded"] > 0
    assert int(out.argmax()) == int(ref.argmax())
    assert ((out.float() - ref).norm() / ref.norm()) < 5e-2


def test_ane_rules_eager_equivalent():
    """Правила Apple для ANE (Linear -> свёртка 1x1, внимание по головам) не меняют
    математику модели: проверка в eager, без Core AI."""
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
    """MLP кусками скрытого измерения: Sequential (Whisper) и gate/up/down (Qwen) — та же
    математика; маленькие MLP не трогаются."""
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
        assert st["mlp_chunk"] == 0  # вес 0.5 МБ — меньше порога
        assert passes.chunk_mlp(m, max_bytes=128 << 10) == 1 and len(m[0].up) == 4
        torch.testing.assert_close(m(x), ref_s, rtol=1e-4, atol=1e-5)
        m, _ = passes.prepare(lm, ())
        assert passes.chunk_mlp(m, max_bytes=8 << 10) == 2 and len(m.model.layers[0].mlp.up) == 4
        torch.testing.assert_close(m(ids).logits, ref_l, rtol=1e-4, atol=1e-4)
        m, _ = passes.prepare(lm, ())  # down_proj в обёртке масштаба (CausalLM)
        passes.rescale_for_fp16(m, k=4)
        assert passes.chunk_mlp(m, max_bytes=8 << 10) == 2 and m.model.layers[0].mlp.scale == 2 ** -4
        torch.testing.assert_close(m(ids).logits, ref_l, rtol=1e-4, atol=1e-4)


@needs_coreai
def test_causal_lm_matches_eager():
    """KV-кеш — состояния Core AI (схема Apple): prefill куском + декод окнами по 8
    дают те же логиты, что transformers на всём контексте."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import numpy as np

    torch.manual_seed(0)
    lm = Qwen2ForCausalLM(Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=1000)).eval()
    ids = torch.randint(0, 1000, (100,)).tolist()  # 37 + 63: переход со ступени 64 на 128
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
    """Потоки в окне декода (generate_batch): у каждого свои позиции и маска, K/V вперемешку
    в общем буфере — токены совпадают с одиночной генерацией каждого промпта."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    lm = Qwen2ForCausalLM(Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=1000)).eval()
    prompts = [torch.randint(0, 1000, (n,)).tolist() for n in (5, 40, 17)]  # 40 — два куска prefill
    st = torch_ane.CausalLM(lm, max_len=256, prefill_q=32, ladder=(128,), batch=True)
    assert st.generate_batch(prompts, 12) == [st.generate(p, 12) for p in prompts]


@needs_coreai
def test_silu_after_groupnorm():
    """Родная SiLU на ANE врёт около нуля (~5% после GroupNorm); export_for_ane заменяет её
    записью через tanh — связка GroupNorm -> SiLU -> свёртка совпадает с eager."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1)).eval()
    x = torch.randn(2, 64, 16, 16) * 3
    with torch.no_grad():
        ref = m(x)
    cm = torch_ane.compile(m, validate=False)
    out = cm(x)
    assert ((out - ref).norm() / ref.norm()) < 5e-3
