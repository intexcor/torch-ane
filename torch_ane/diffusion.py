"""Stable Diffusion (diffusers) on the Neural Engine: UNet, text encoder and VAE decoder
via CompiledModule, step scheduler and tokenizer — on CPU.

The key to speed is attention in the ANE layout (AttnProcessor): on ANE, permute/reshape
of large tensors is a memory copy, and attention at 64x64 (4096 tokens) in the usual
formulation was 9x slower than GPU. Following Apple's recipe (BC1S, per head, einsum without
transposes, softmax over keys, query chunks) tiny-sd's UNet on ANE is faster than GPU.

    sd = StableDiffusion("segmind/tiny-sd")
    image, t = sd("a cute robot sitting on a desk", steps=20, seed=0)
"""
import hashlib
import inspect
import time

import torch
import torch.nn as nn

Q_CHUNK = 512  # query chunks in attention when T > 1024


class AttnProcessor:
    """diffusers attention processor in the ANE layout (like Apple's SPLIT_EINSUM):
    Q and V — [B, C, 1, T], K — [B, T, 1, C] without transposing; per-head einsum
    bchq,bkhc->bkhq, softmax over the key axis, long queries — in chunks."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        res = hidden_states
        shape4 = hidden_states.shape if hidden_states.ndim == 4 else None
        if shape4 is not None:
            b, c, hh, ww = shape4
            hidden_states = hidden_states.view(b, c, hh * ww).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            ctx = attn.norm_encoder_hidden_states(ctx)
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        heads = attn.heads
        d, tq = q.shape[-1] // heads, q.shape[1]
        qb = q.transpose(1, 2).unsqueeze(2)  # [B, C, 1, Tq]
        kb = k.unsqueeze(2)                  # [B, Tk, 1, C]
        vb = v.transpose(1, 2).unsqueeze(2)  # [B, C, 1, Tk]
        n = Q_CHUNK if tq > 1024 else tq
        outs = []
        for h in range(heads):
            qh, kh, vh = qb[:, h * d:(h + 1) * d], kb[..., h * d:(h + 1) * d], vb[:, h * d:(h + 1) * d]
            parts = []
            for i in range(0, tq, n):
                w = torch.einsum("bchq,bkhc->bkhq", qh[..., i:i + n], kh) * attn.scale
                parts.append(torch.einsum("bkhq,bchk->bchq", w.softmax(dim=1), vh))
            outs.append(torch.cat(parts, dim=3) if len(parts) > 1 else parts[0])
        o = torch.cat(outs, dim=1).squeeze(2).transpose(1, 2)  # [B, Tq, C]
        o = attn.to_out[1](attn.to_out[0](o))
        if shape4 is not None:
            o = o.transpose(1, 2).reshape(shape4)
        if attn.residual_connection:
            o = o + res
        return o / attn.rescale_output_factor


# processor version — in a plain attribute of the wrappers, so the CompiledModule build cache changes with it
_ATTN_TAG = hashlib.sha256(inspect.getsource(AttnProcessor).encode()).hexdigest()[:8]


class _UNet(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.u = unet
        self.attn_tag = _ATTN_TAG

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.u(sample, timestep, encoder_hidden_states, return_dict=False)[0]


class _TextEncoder(nn.Module):
    """CLIP encoder without the embedding table: input — token+position embeddings (from CPU)."""

    def __init__(self, te):
        super().__init__()
        self.te = te

    def forward(self, hidden):
        h = self.te.encoder(inputs_embeds=hidden, attention_mask=None, is_causal=True).last_hidden_state
        return self.te.final_layer_norm(h)


class _VAEDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.pqc, self.dec = vae.post_quant_conv, vae.decoder
        self.attn_tag = _ATTN_TAG

    def forward(self, z):
        if self.pqc is not None:
            z = self.pqc(z)
        return self.dec(z)


class StableDiffusion:
    """diffusers Stable Diffusion pipeline with the UNet, text encoder and VAE decoder
    on the given Core AI compute device ("ane" or "gpu"). The first program build takes
    minutes, after that — from cache."""

    def __init__(self, model_id="segmind/tiny-sd", device="ane", size=512):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        from .runtime import CompiledModule
        self.device, self.size = device, size
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None,
                                                       requires_safety_checker=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.eval().set_attn_processor(AttnProcessor())
        pipe.vae.eval().set_attn_processor(AttnProcessor())
        te = pipe.text_encoder.eval()
        self.t = {}  # per-part timings of the last generation
        self._unet = CompiledModule(_UNet(pipe.unet).eval(), device=device, validate=False)
        self._text = CompiledModule(_TextEncoder(te).eval(), device=device, validate=False)
        self._vae = CompiledModule(_VAEDecoder(pipe.vae).eval(), device=device, validate=False)
        sd = self

        class UNetShim:
            config, dtype = pipe.unet.config, pipe.unet.dtype

            def __call__(self, sample, timestep, encoder_hidden_states=None, return_dict=True, **kw):
                from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
                timestep = torch.as_tensor(timestep)
                if timestep.dim() == 0:
                    timestep = timestep.float().expand(sample.shape[0])
                a = time.perf_counter()
                out = sd._unet(sample.float(), timestep.float(), encoder_hidden_states.float())
                sd.t.setdefault("unet", []).append(time.perf_counter() - a)
                return UNet2DConditionOutput(sample=out) if return_dict else (out,)

        class TextShim:
            config, dtype = te.config, te.dtype

            def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
                a = time.perf_counter()
                with torch.no_grad():  # embedding table lookup — on CPU
                    hidden = te.embeddings(input_ids=input_ids)
                out = sd._text(hidden.float())
                sd.t["text"] = sd.t.get("text", 0) + time.perf_counter() - a
                return (out,)

        def vae_decode(z, return_dict=True, generator=None, **kw):
            from diffusers.models.autoencoders.vae import DecoderOutput
            a = time.perf_counter()
            out = sd._vae(z.float())
            sd.t["vae"] = time.perf_counter() - a
            return DecoderOutput(sample=out) if return_dict else (out,)

        pipe.unet, pipe.text_encoder = UNetShim(), TextShim()
        pipe.vae.decode = vae_decode
        self.pipe = pipe

    def __call__(self, prompt, steps=20, seed=None, negative_prompt="", guidance=7.5):
        """-> (PIL image, {"text": s, "unet": [s per step], "vae": s, "total": s})."""
        self.t = {}
        g = torch.Generator("cpu").manual_seed(seed if seed is not None else int(time.time()))
        a = time.perf_counter()
        img = self.pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                        guidance_scale=guidance, height=self.size, width=self.size, generator=g).images[0]
        self.t["total"] = time.perf_counter() - a
        return img, self.t
