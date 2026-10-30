"""GGUF (llama.cpp) weights for ChunkedCausalLM: layer tensor names and the inverse Q/K
permutation. Reading and decompression — via the gguf package (numpy), without MLX."""

# GGUF layer tensor blk.{i}.{kind} -> transformers decoder-layer module
BLOCK = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def unpermute(w, n_head):
    """Inverse of the Q/K permutation in convert_hf_to_gguf.py (LlamaModel.permute)."""
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
