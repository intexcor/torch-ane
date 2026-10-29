"""Stable Diffusion on the Neural Engine: text encoder, UNet and VAE all on ANE.

    python examples/imagine.py "a red fox in the snow"
    python examples/imagine.py --gpu "..."          # same pipeline on the GPU, for comparison

Images go to ./images/. The first run builds three programs (~5 min), then they load from cache.
"""
import argparse
import datetime
import pathlib
import time
import warnings

from torch_ane import StableDiffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="segmind/tiny-sd")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    t = time.perf_counter()
    sd = StableDiffusion(a.model, device="gpu" if a.gpu else "ane")
    sd("warmup", steps=1, seed=0)  # build or load all three programs
    print(f"ready in {time.perf_counter() - t:.0f} s")

    img, tm = sd(a.prompt, steps=a.steps, seed=a.seed)
    out = pathlib.Path("images")
    out.mkdir(exist_ok=True)
    path = out / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.png"
    img.save(path)
    u = sorted(tm.get("unet", []))
    print(f"text {tm.get('text', 0) * 1e3:.0f} ms, UNet {len(u)} x {u[len(u) // 2] * 1e3 if u else 0:.0f} ms, "
          f"VAE {tm.get('vae', 0) * 1e3:.0f} ms, total {tm['total']:.1f} s -> {path}")


if __name__ == "__main__":
    main()
