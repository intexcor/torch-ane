"""Running torch models on the Neural Engine via Core AI (macOS 27+, coreai-torch package).

    import torch_ane
    m = torch_ane.compile(model.eval(), example_inputs=(x,))   # device="ane" | "gpu"
    y = m(x)          # torch.Tensor; first call per shape compiles (up to ~15 s), then cache

How it works:
  * torch.export -> get_decomp_table() -> TorchConverter -> .aimodel in fp16:
    on a Mac Core AI runs fp32 models on the GPU, only fp16 reaches the ANE;
  * the ANE works only with static shapes — each input shape gets its own
    model (following RF-DETR: "one .aimodel per batch size"), at most
    MAX_SHAPES, compiled models are cached on disk (CACHE_DIR);
  * the first call per shape is checked against eager torch (fp32): fp16 on the ANE has
    known silent corruptions (coreai-torch #115, topk/gather) — on mismatch
    the model for that shape moves to the Core AI GPU, and on a repeat — to eager.

Measured on M4 (inference, fp16 ANE / MLX backend fp32 / MPS): ResNet-18 b16 4.5 / 24.5 / 32.7 ms,
BERT-base 8x128 20 / 71 / 75, ViT-base b8 40 / 109 / 114; the ANE doesn't help the Whisper
encoder (14 vs 17), it's better kept on the GPU.
"""
import asyncio
import contextlib
import copy
import hashlib
import io
import os
import pathlib
import shutil
import warnings

import numpy as np
import torch

# cache of built programs; TORCH_ANE_CACHE — a different location
CACHE_DIR = pathlib.Path(os.environ.get("TORCH_ANE_CACHE", pathlib.Path.home() / ".cache" / "torch_ane"))
MAX_SHAPES = 8
REL_TOL = 5e-2  # fp16 on ANE gives ~1e-3..1e-2; more means corruption


DEVICES = {"ane": "neural_engine", "neural_engine": "neural_engine", "gpu": "gpu"}


def _device(name):
    """"ane" / "gpu" (and the Core AI name "neural_engine") -> Core AI compute device name."""
    try:
        return DEVICES[name]
    except KeyError:
        raise ValueError(f"device: 'ane' or 'gpu', not {name!r}") from None


def available():
    """Whether Core AI is available (macOS 27+ and the coreai-core/coreai-torch packages)."""
    try:
        import coreai.runtime  # noqa: F401
        import coreai_torch  # noqa: F401
        return True
    except ImportError:
        return False


def compress_program(prog, kind):
    """Compress AIProgram weights with coreai-opt (post-training, data-free)."""
    from coreai_opt.coreai_utils import CompressionGranularity as G
    from coreai_opt.coreai_utils import DType, palettize_weights, quantize_weights
    if kind.startswith("pal"):  # "pal4" (group 16), "pal4g32", "pal4ch" (per channel)
        bits, _, rest = kind[3:].partition("g")
        if bits.endswith("ch"):
            return palettize_weights(prog, lut_dtype=None, n_bits=int(bits[:-2]), granularity=G.PER_CHANNEL)
        return palettize_weights(prog, lut_dtype=None, n_bits=int(bits), granularity=G.PER_GROUPED_CHANNEL,
                                 group_size=int(rest or 16),
                                 num_kmeans_workers=int(os.environ.get("TORCH_ANE_KM_WORKERS", max(1, (os.cpu_count() or 4) - 2))))
    if kind == "int8":
        return quantize_weights(prog, DType.INT8, granularity=G.PER_CHANNEL)
    if kind == "int4":
        return quantize_weights(prog, DType.INT4, granularity=G.PER_BLOCK, block_size=32)
    if kind in ("int4ch", "int4ch_a"):  # per channel — the format Apple recommends for ANE
        from coreai_opt.coreai_utils.common import QScheme
        qs = QScheme.ASYMMETRIC if kind == "int4ch_a" else QScheme.SYMMETRIC
        return quantize_weights(prog, DType.INT4, qscheme=qs, granularity=G.PER_CHANNEL)
    raise ValueError(f"unknown compression: {kind}")


def model_digest(model):
    """Model fingerprint for the cache key: weights, buffers, class code and simple module attributes."""
    h = hashlib.sha256()
    # all buffers, including non-persistent ones (RoPE inv_freq): state_dict doesn't contain them,
    # and a model with changed RoPE frequencies would get someone else's compiled cache
    tensors = dict(model.state_dict())
    tensors.update({f"buffer:{k}": v for k, v in model.named_buffers()})
    for k, v in sorted(tensors.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    # module code: without it an edited forward (or a weightless module) would take the old model from cache
    import inspect
    for cls in sorted({type(m) for m in model.modules()}, key=lambda c: (c.__module__, c.__qualname__)):
        h.update(f"{cls.__module__}.{cls.__qualname__}".encode())
        try:
            h.update(inspect.getsource(cls).encode())
        except (OSError, TypeError):
            pass
    # simple module attributes (flags, modes, sizes): the same class with the same weights
    # but a different mode is a different model
    for name, mod in model.named_modules():
        attrs = {k: v for k, v in vars(mod).items()
                 if not k.startswith("_") and isinstance(v, (bool, int, float, str, tuple, type(None)))}
        h.update(f"{name}:{sorted(attrs.items(), key=lambda kv: kv[0])!r}".encode())
    return h.hexdigest()[:16]


class _HalfInputs(torch.nn.Module):
    """ANE wrapper: floating-point inputs arrive in fp16 (as in Apple's models) and
    are cast to fp32 inside; the passes._to_fp16 pass turns that cast into a no-op.
    An fp32 input cast to fp16 by the first graph node gets fused by the compiler into the
    input op — if the input goes straight into SDPA, the ANE silently computes garbage."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*[a.float() if isinstance(a, torch.Tensor) and a.dtype == torch.float16 else a
                            for a in args])


class _Fp16Interface(torch.nn.Module):
    """fp16 model with fp32 output (cast inside the graph). Inputs are fed in fp16:
    NDArray copies the input element by element (~1 GB/s, coreai-core 1.0.0b3), and an fp16 input
    with the cast on CPU (0.2 ms) comes out cheaper than an fp32 input with the cast on ANE."""

    def __init__(self, m16):
        super().__init__()
        self.m = m16

    def forward(self, *args):
        out = self.m(*args)
        if isinstance(out, torch.Tensor):
            return out.float() if out.is_floating_point() else out
        return tuple(o.float() if o.is_floating_point() else o for o in out)


class _Compiled:
    def __init__(self, path, device, half_inputs=True):
        self.half_inputs = half_inputs
        import coreai.runtime as rt
        self.loop = asyncio.new_event_loop()
        kind = getattr(rt.ComputeUnitKind, device)()
        self.model = self.loop.run_until_complete(
            rt.AIModel.load(path, rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)))
        self.fn = self.model.load_function(self.model.function_names[0])
        self.device = device

    def __call__(self, args):
        import coreai.runtime as rt
        inputs = {}
        for name, a in zip(self.fn.desc.input_names, args):
            a = a.detach().cpu()
            if a.is_floating_point():
                a = a.half() if self.half_inputs else a
            else:
                a = a.to(torch.int32)
            inputs[name] = rt.NDArray(a.contiguous())  # copies element by element, see _Fp16Interface
        out = self.loop.run_until_complete(self.fn(inputs))
        return [torch.from_numpy(np.array(out[n].numpy())) for n in self.fn.desc.output_names]


class CompiledModule(torch.nn.Module):
    """torch module executed by Core AI: torch.Tensor in/out (output in the source model's
    dtype), a separate static program per input shape.

    device — "ane" or "gpu"; validate — check the first call per shape against eager (on
    mismatch the shape goes to GPU, then to eager); compress — weight compression of the built
    program ("int8", "int4", "pal4"...; see compress_program). "int8" on ANE is done
    in torch before export and only in Linear/Conv2d (passes.quantize_int8_modules):
    compressing the built program also hits Conv1d, and with int8 there the ANE compiler crashes."""

    def __init__(self, model, device="ane", validate=True, check=None, rules=None, compress=None,
                 externalize_modules=()):
        super().__init__()
        if not available():
            raise RuntimeError("Core AI unavailable: requires macOS 27 and the coreai-torch package")
        self.model = model.eval()
        self.device = _device(device)
        self.validate = validate
        # check(ref_outs, got_outs, args) -> bool: custom check of the first call; by default —
        # relative error of each output < REL_TOL and everything finite
        self.check = check
        self.rules = rules  # Apple's rules for ANE (passes.prepare); None — default
        # weight compression of the built program (coreai-opt, data-free): "pal4"/"pal6"/"pal8" —
        # a lookup table per group of 16 channels (what the ANE decompresses in hardware),
        # "int8" — per channel, "int4" — in blocks of 32; None — fp16
        self.compress = compress
        # extra Core AI composite ops (passes.use_composite_ops): SDPA/RoPE/GatherMM/GatedDeltaUpdate
        self.externalize_modules = tuple(externalize_modules)
        self.digest = model_digest(model)
        self._compiled = {}  # shape signature -> _Compiled | "eager"
        self.ane_stats = {}

    def _path(self, sig):
        from .passes import DEFAULT_RULES, PASSES_VERSION
        rules = tuple(sorted(DEFAULT_RULES if self.rules is None else self.rules))
        ext = tuple(s.composite_op_name for s in self.externalize_modules)
        compress = "int8t" if self.compress == "int8" and self.device == "neural_engine" else self.compress
        extra = tuple(x for x in (rules, compress, ext) if x)
        h = hashlib.sha256(repr((sig, *extra) if extra else sig).encode()).hexdigest()[:12]
        # v4: fp16 inputs on ANE; PASSES_VERSION — export_for_ane passes version (doesn't affect GPU)
        ver = f"v4{PASSES_VERSION}" if self.device == "neural_engine" else "v4"
        return CACHE_DIR / f"{self.digest}_{h}_{self.device}_{ver}.aimodel"

    def _build(self, args, sig):
        from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
        path = self._path(sig)
        if path.exists():
            touch_cache(path)
            return path
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings(), torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            if self.device == "neural_engine":
                # ANE passes (passes): RMSNorm -> composite op, constant folding
                # (mask, RoPE), causal mask -> is_causal, then fp16
                from coreai_torch.composite_ops import RMSNormImpl

                from . import passes
                m, st = passes.prepare(self.model, self.rules)
                if self.compress == "int8":
                    m, st["int8"] = passes.quantize_int8_modules(m, None, tuple(args))
                self.ane_stats.update(st)
                m = _HalfInputs(m)
                a16 = tuple(a.half() if isinstance(a, torch.Tensor) and a.dtype == torch.float32 else a for a in args)
                conv = TorchConverter().add_pytorch_module(
                    m, export_fn=passes.export_for_ane(m, a16, self.ane_stats),
                    externalize_modules=[ExternalizeSpec(target_class=RMSNormImpl, composite_op_name="rms_norm",
                                                         composite_attrs=["axes", "eps"])]
                    + list(self.externalize_modules))
            else:
                wrapped = _Fp16Interface(copy.deepcopy(self.model).half())
                a16 = tuple(a.half() if a.is_floating_point() else a for a in args)
                ep = torch.export.export(wrapped, a16).run_decompositions(get_decomp_table())
                conv = TorchConverter().add_exported_program(ep)
            prog = conv.to_coreai()
            if self.compress and not (self.compress == "int8" and self.device == "neural_engine"):
                prog = compress_program(prog, self.compress)
            prog.save_asset(path)
        prune_cache(keep=(path,))
        return path

    def _check(self, compiled, args):
        """Check the first call against eager fp32: catches silent fp16 corruptions on ANE."""
        with torch.no_grad():
            ref = self.model(*args)
        ref = [ref] if isinstance(ref, torch.Tensor) else list(ref)
        got = compiled(args)
        if self.check is not None:
            return bool(self.check(ref, got, args))
        for r, g in zip(ref, got):
            r, g = r.float(), g.float()
            if r.shape != g.shape or not torch.isfinite(g).all():
                return False
            if (g - r).norm() / (r.norm() + 1e-12) > REL_TOL:
                return False
        return True

    def _get(self, args):
        sig = tuple((tuple(a.shape), str(a.dtype)) for a in args)
        c = self._compiled.get(sig)
        if c is not None:
            return c
        if len(self._compiled) >= MAX_SHAPES:
            warnings.warn(f"torch_ane: more than {MAX_SHAPES} input shapes — new shapes go to eager torch; "
                          "ANE needs static shapes (pad to fixed sizes)")
            return "eager"
        path = self._build(args, sig)
        for device in dict.fromkeys([self.device, "gpu"]):
            # the ANE-passes graph takes inputs in the original dtype (fp16 cast inside),
            # the _Fp16Interface graph — in fp16; on GPU fallback the file is the same
            try:
                c = _Compiled(path, device)
            except RuntimeError:
                # an old cached build stopped loading (when low on space macOS partially
                # purges the Core AI system specialization cache) — rebuild once
                warnings.warn(f"torch_ane: build {path.name} fails to load — rebuilding")
                shutil.rmtree(path, ignore_errors=True)
                path = self._build(args, sig)
                c = _Compiled(path, device)
            if not self.validate or self._check(c, args):
                if device != self.device:
                    warnings.warn(f"torch_ane: on {self.device} the result diverges from eager — shape {sig} "
                                  f"moved to {device}")
                self._compiled[sig] = c
                return c
        warnings.warn(f"torch_ane: Core AI diverges from eager for shape {sig} — eager torch")
        self._compiled[sig] = "eager"
        return "eager"

    @torch.no_grad()
    def forward(self, *args):
        c = self._get(args)
        if c == "eager":
            return self.model(*args)
        outs = c(args)
        ref_dtype = next(self.model.parameters(), torch.empty(0)).dtype  # weightless module — fp32
        outs = [o.to(ref_dtype) if o.is_floating_point() else o for o in outs]
        return outs[0] if len(outs) == 1 else tuple(outs)


def compile(model, example_inputs=None, device="ane", **kw):  # noqa: A001 — like torch.compile
    """model -> CompiledModule. With example_inputs, compilation and checking happen immediately rather than
    on the first call. Other arguments — as in CompiledModule (validate, compress, rules...)."""
    m = CompiledModule(model, device=device, **kw)
    if example_inputs is not None:
        m(*example_inputs)
    return m


def cleanup_mpsgraph_tmp(include_self=False):
    """Core AI/MPSGraph puts a mpsgraph-<pid>-<time> folder into $TMPDIR/com.apple.MetalPerformanceShadersGraph
    per loaded program (a copy of the weights, up to a gigabyte) and never deletes them —
    neither on exit nor on crash. Chunked Devstral left ~10 GB per run. We delete folders
    of finished processes (and our own — on exit). Since macOS 27.0.1 some of these folders
    (ANE region compilation) appear in ...GraphHost — we check both."""
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
    dirs = [d for r in ("com.apple.MetalPerformanceShadersGraph", "com.apple.MetalPerformanceShadersGraphHost")
            if (tmp / r).is_dir() for d in (tmp / r).glob("mpsgraph-*")]
    me, freed = os.getpid(), 0
    for d in dirs:
        try:
            pid = int(d.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        alive = pid != me or not include_self
        if pid != me:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                pass
        if not alive:
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return freed


def _install_tmp_cleanup():
    import atexit
    cleanup_mpsgraph_tmp()
    atexit.register(cleanup_mpsgraph_tmp, include_self=True)


_install_tmp_cleanup()


def touch_cache(path):
    """Mark a build as used (modification time — for prune_cache)."""
    try:
        os.utime(path)
    except OSError:
        pass


def prune_cache(max_gb=None, keep=()):
    """Keep the build cache at most max_gb (default TORCH_ANE_CACHE_GB, 40): the least recently
    used builds are deleted (by modification time; loading updates it), except paths in keep.
    Each build is ~1 GB for a mid-sized model, and without a limit a day of experiments
    ate a hundred gigabytes. Returns the freed bytes."""
    max_gb = float(os.environ.get("TORCH_ANE_CACHE_GB", 40) if max_gb is None else max_gb)
    if not CACHE_DIR.is_dir():
        return 0

    def size(p):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size

    keep = {pathlib.Path(k).resolve() for k in keep}
    entries = sorted(((e.stat().st_mtime, e, size(e)) for e in CACHE_DIR.iterdir()), key=lambda t: t[0])
    total, freed = sum(s for _, _, s in entries), 0
    for _, e, s in entries:
        if total <= max_gb * 1e9:
            break
        if e.resolve() in keep or e.name.endswith(".partial.aimodel"):
            continue
        shutil.rmtree(e, ignore_errors=True) if e.is_dir() else e.unlink(missing_ok=True)
        total, freed = total - s, freed + s
    return freed


def clear_system_cache():
    """Core AI system compilation cache (~/Library/Caches/coreai-cache): a compiled copy
    of every model ever loaded, never cleaned up by itself (a week of experiments —
    87 GB). Can only be deleted entirely: after a partial deletion Core AI stops
    loading models. The first load of each model after clearing includes compilation."""
    root = pathlib.Path.home() / "Library" / "Caches" / "coreai-cache"
    freed = 0
    for d in root.iterdir() if root.is_dir() else ():
        if d.is_dir() and d.name not in ("sync", "tmp"):
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return freed


def probe_tmpdir():
    """Temporary folder for experiments, deleted on process exit (previously probes
    used tempfile.mkdtemp() and left a gigabyte-sized model copy per run)."""
    import atexit
    import tempfile
    d = tempfile.mkdtemp(prefix="torch_ane_probe_")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d
