"""Общее для LLM на ANE: KV-кеш — состояния Core AI (схема Apple: весь K одним
состоянием, весь V другим, запись куска — coreai.slice_update со смещением из входа),
слой кеша transformers поверх этих буферов и маска окна."""
import numpy as np
import torch
from transformers.cache_utils import CacheLayerMixin

DECODE_Q = 8
MASK_NEG = -40000.0  # не -inf: ANE неверно считает softmax с -inf

_lib = torch.library.Library("torch_ane", "FRAGMENT")
_lib.define("kv_write(Tensor cache, Tensor new, int layer, Tensor step) -> Tensor")


@torch.library.impl(_lib, "kv_write", "CompositeExplicitAutograd")
def _kv_write(cache, new, layer, step):
    """cache [L, 1, kv, M, hd], new [1, 1, kv, C, hd] (ранг как у кеша — требование
    slice_update): cache[layer, :, :, step:step+C] = new."""
    out = cache.clone()
    s = int(step.reshape(-1)[0])
    out[layer:layer + 1, :, :, s:s + new.shape[3]] = new.to(out.dtype)
    return out


@torch.library.register_fake("torch_ane::kv_write")
def _kv_write_fake(cache, new, layer, step):
    return torch.empty_like(cache)


def register_lowering(conv):
    from coreai._compiler.dialects import coreai
    from coreai_torch._utils import get_operand

    @conv.register_torch_lowering("torch_ane::kv_write.default")
    def _(values_map, node, loc):
        cache = get_operand(values_map, node, 0)
        new = get_operand(values_map, node, 1)
        layer = node.args[2]
        step = get_operand(values_map, node, 3)  # [1] int32
        _, _, kv, n, hd = tuple(node.args[1].meta["val"].shape)
        begin = coreai.concat(0, [[layer], [0], [0], step, [0]])
        end = coreai.concat(0, [[layer + 1], [1], [kv], coreai.broadcasting_add(step, n), [hd]])
        return coreai.slice_update(cache, begin, end, [1, 1, 1, 1, 1], new)


class StateLayer(CacheLayerMixin):
    """Слой кеша transformers поверх общих буферов модуля-владельца."""

    is_sliding = False

    def __init__(self, owner, idx, step):
        super().__init__()
        self.owner, self.idx, self.step = owner, idx, step
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        pass

    def update(self, key_states, value_states, *args, **kwargs):
        o, i = self.owner, self.idx
        o.key_cache.copy_(torch.ops.torch_ane.kv_write(o.key_cache, key_states[None], i, self.step))
        o.value_cache.copy_(torch.ops.torch_ane.kv_write(o.value_cache, value_states[None], i, self.step))
        dt = key_states.dtype
        return o.key_cache[i].to(dt), o.value_cache[i].to(dt)  # [1, kv, M, hd]

    def get_mask_sizes(self, query_length):
        return self.owner.max_len, 0

    def get_seq_length(self):
        return 0

    def get_max_length(self):
        return self.owner.max_len



def window_mask(start, n, q, max_len):
    """Строка j (позиция start+j) видит ключи [0, start+j]; строки паддинга — ключ 0."""
    m = np.full((1, 1, q, max_len), MASK_NEG, np.float16)
    for j in range(q):
        m[0, 0, j, :(start + j + 1 if j < n else 1)] = 0
    return m




def fresh_rope(lm):
    """Пересоздать модуль RoPE из конфига во fp32. Если модель загружена или хоть раз
    переведена в bf16/fp16, её inv_freq округлён — ключи в кеше от ANE-prefill
    слегка расходились бы с RoPE, который mlx-lm считает при генерации точно."""
    rope = getattr(getattr(lm, "model", None), "rotary_emb", None)
    if rope is not None:
        try:
            lm.model.rotary_emb = type(rope)(config=lm.config)
        except TypeError:
            pass
    return lm

