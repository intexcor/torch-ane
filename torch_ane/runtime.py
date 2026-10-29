"""Исполнение torch-моделей на Neural Engine через Core AI (macOS 27+, пакет coreai-torch).

    import torch_ane
    m = torch_ane.compile(model.eval(), example_inputs=(x,))   # device="ane" | "gpu"
    y = m(x)          # torch.Tensor; первый вызов на форму — компиляция (до ~15 с), дальше кеш

Как это устроено:
  * torch.export -> get_decomp_table() -> TorchConverter -> .aimodel в fp16:
    на Маке fp32-модели Core AI исполняет на GPU, на ANE попадает только fp16;
  * ANE работает только со статическими формами — на каждую форму входа своя
    модель (по образцу RF-DETR: «один .aimodel на размер батча»), не больше
    MAX_SHAPES, скомпилированные модели кешируются на диске (CACHE_DIR);
  * первый вызов на форму сверяется с eager torch (fp32): у fp16 на ANE есть
    известные тихие порчи (coreai-torch #115, topk/gather) — при расхождении
    модель для этой формы переезжает на GPU Core AI, а при повторе — на eager.

Замер на M4 (инференс, fp16 ANE / backend MLX fp32 / MPS): ResNet-18 б16 4.5 / 24.5 / 32.7 мс,
BERT-base 8x128 20 / 71 / 75, ViT-base б8 40 / 109 / 114; энкодеру Whisper ANE не
помогает (14 против 17), его выгоднее держать на GPU.
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

# кеш собранных программ; TORCH_ANE_CACHE — другое место
CACHE_DIR = pathlib.Path(os.environ.get("TORCH_ANE_CACHE", pathlib.Path.home() / ".cache" / "torch_ane"))
MAX_SHAPES = 8
REL_TOL = 5e-2  # fp16 на ANE даёт ~1e-3..1e-2; больше — значит порча


DEVICES = {"ane": "neural_engine", "neural_engine": "neural_engine", "gpu": "gpu"}


def _device(name):
    """"ane" / "gpu" (и имя Core AI "neural_engine") -> имя блока Core AI."""
    try:
        return DEVICES[name]
    except KeyError:
        raise ValueError(f"device: 'ane' или 'gpu', а не {name!r}") from None


def available():
    """Есть ли Core AI (macOS 27+ и пакеты coreai-core/coreai-torch)."""
    try:
        import coreai.runtime  # noqa: F401
        import coreai_torch  # noqa: F401
        return True
    except ImportError:
        return False


def compress_program(prog, kind):
    """Сжатие весов AIProgram средствами coreai-opt (post-training, без данных)."""
    from coreai_opt.coreai_utils import CompressionGranularity as G
    from coreai_opt.coreai_utils import DType, palettize_weights, quantize_weights
    if kind.startswith("pal"):  # "pal4" (группа 16), "pal4g32", "pal4ch" (по каналам)
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
    if kind in ("int4ch", "int4ch_a"):  # по каналам — формат, который Apple рекомендует для ANE
        from coreai_opt.coreai_utils.common import QScheme
        qs = QScheme.ASYMMETRIC if kind == "int4ch_a" else QScheme.SYMMETRIC
        return quantize_weights(prog, DType.INT4, qscheme=qs, granularity=G.PER_CHANNEL)
    raise ValueError(f"неизвестное сжатие: {kind}")


def model_digest(model):
    """Отпечаток модели для ключа кеша: веса, буферы, код классов и простые атрибуты модулей."""
    h = hashlib.sha256()
    # все буферы, включая непостоянные (inv_freq у RoPE): state_dict их не содержит,
    # и модель с изменёнными частотами RoPE получила бы чужой скомпилированный кеш
    tensors = dict(model.state_dict())
    tensors.update({f"buffer:{k}": v for k, v in model.named_buffers()})
    for k, v in sorted(tensors.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    # код модулей: без него правка forward (или модуль без весов) взяла бы старую модель из кеша
    import inspect
    for cls in sorted({type(m) for m in model.modules()}, key=lambda c: (c.__module__, c.__qualname__)):
        h.update(f"{cls.__module__}.{cls.__qualname__}".encode())
        try:
            h.update(inspect.getsource(cls).encode())
        except (OSError, TypeError):
            pass
    # простые атрибуты модулей (флаги, режимы, размеры): тот же класс с теми же весами,
    # но другим режимом — другая модель
    for name, mod in model.named_modules():
        attrs = {k: v for k, v in vars(mod).items()
                 if not k.startswith("_") and isinstance(v, (bool, int, float, str, tuple, type(None)))}
        h.update(f"{name}:{sorted(attrs.items(), key=lambda kv: kv[0])!r}".encode())
    return h.hexdigest()[:16]


class _HalfInputs(torch.nn.Module):
    """Обёртка для ANE: вещественные входы приходят в fp16 (как у моделей Apple) и
    внутри приводятся к fp32; проход passes._to_fp16 превращает этот каст в пустой.
    fp32-вход, приведённый к fp16 первым узлом графа, компилятор вливает в операцию
    ввода — если вход сразу идёт в SDPA, ANE тихо считает мусор."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*[a.float() if isinstance(a, torch.Tensor) and a.dtype == torch.float16 else a
                            for a in args])


class _Fp16Interface(torch.nn.Module):
    """fp16-модель с fp32-выходом (каст внутри графа). Входы подаются в fp16:
    NDArray копирует вход поэлементно (~1 ГБ/с, coreai-core 1.0.0b3), и fp16-вход
    с кастом на CPU (0.2 мс) выходит дешевле, чем fp32-вход с кастом на ANE."""

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
            inputs[name] = rt.NDArray(a.contiguous())  # копирует поэлементно, см. _Fp16Interface
        out = self.loop.run_until_complete(self.fn(inputs))
        return [torch.from_numpy(np.array(out[n].numpy())) for n in self.fn.desc.output_names]


class CompiledModule(torch.nn.Module):
    """torch-модуль, исполняемый Core AI: на вход/выход — torch.Tensor (выход в dtype
    исходной модели), на каждую форму входа — своя статическая программа.

    device — "ane" или "gpu"; validate — сверять первый вызов формы с eager (при
    расхождении форма уходит на GPU, потом на eager); compress — сжатие весов готовой
    программы ("int8", "int4", "pal4"...; см. compress_program). "int8" на ANE делается
    в torch до экспорта и только в Linear/Conv2d (passes.quantize_int8_modules):
    сжатие готовой программы задевает и Conv1d, а с int8 в ней компилятор ANE падает."""

    def __init__(self, model, device="ane", validate=True, check=None, rules=None, compress=None,
                 externalize_modules=()):
        super().__init__()
        if not available():
            raise RuntimeError("Core AI недоступен: нужна macOS 27 и пакет coreai-torch")
        self.model = model.eval()
        self.device = _device(device)
        self.validate = validate
        # check(ref_outs, got_outs, args) -> bool: своя сверка первого вызова; по умолчанию —
        # относительная ошибка каждого выхода < REL_TOL и всё конечно
        self.check = check
        self.rules = rules  # правила Apple для ANE (passes.prepare); None — по умолчанию
        # сжатие весов готовой программы (coreai-opt, без данных): "pal4"/"pal6"/"pal8" —
        # таблица значений на группу из 16 каналов (то, что ANE разжимает железом),
        # "int8" — по каналам, "int4" — блоками по 32; None — fp16
        self.compress = compress
        # доп. составные операции Core AI (passes.use_composite_ops): SDPA/RoPE/GatherMM/GatedDeltaUpdate
        self.externalize_modules = tuple(externalize_modules)
        self.digest = model_digest(model)
        self._compiled = {}  # сигнатура формы -> _Compiled | "eager"
        self.ane_stats = {}

    def _path(self, sig):
        from .passes import DEFAULT_RULES, PASSES_VERSION
        rules = tuple(sorted(DEFAULT_RULES if self.rules is None else self.rules))
        ext = tuple(s.composite_op_name for s in self.externalize_modules)
        compress = "int8t" if self.compress == "int8" and self.device == "neural_engine" else self.compress
        extra = tuple(x for x in (rules, compress, ext) if x)
        h = hashlib.sha256(repr((sig, *extra) if extra else sig).encode()).hexdigest()[:12]
        # v4: fp16-входы на ANE; PASSES_VERSION — версия проходов export_for_ane (на GPU не влияет)
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
                # ANE-проходы (passes): RMSNorm -> составная операция, свёртка констант
                # (маска, RoPE), причинная маска -> is_causal, затем fp16
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
        """Сверка первого вызова с eager fp32: ловит тихие порчи fp16 на ANE."""
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
            warnings.warn(f"torch_ane: больше {MAX_SHAPES} форм входа — новые формы идут в eager torch; "
                          "для ANE нужны статические формы (дополняйте до фиксированных размеров)")
            return "eager"
        path = self._build(args, sig)
        for device in dict.fromkeys([self.device, "gpu"]):
            # граф ANE-проходов принимает входы в исходном типе (каст в fp16 внутри),
            # граф _Fp16Interface — в fp16; при откате на GPU файл тот же
            try:
                c = _Compiled(path, device)
            except RuntimeError:
                # старая сборка из кеша перестала грузиться (macOS при нехватке места частично
                # вычищает системный кеш специализаций Core AI) — пересобираем один раз
                warnings.warn(f"torch_ane: сборка {path.name} не загружается — пересобираю")
                shutil.rmtree(path, ignore_errors=True)
                path = self._build(args, sig)
                c = _Compiled(path, device)
            if not self.validate or self._check(c, args):
                if device != self.device:
                    warnings.warn(f"torch_ane: на {self.device} результат расходится с eager — форма {sig} "
                                  f"переведена на {device}")
                self._compiled[sig] = c
                return c
        warnings.warn(f"torch_ane: Core AI расходится с eager для формы {sig} — eager torch")
        self._compiled[sig] = "eager"
        return "eager"

    @torch.no_grad()
    def forward(self, *args):
        c = self._get(args)
        if c == "eager":
            return self.model(*args)
        outs = c(args)
        ref_dtype = next(self.model.parameters(), torch.empty(0)).dtype  # модуль без весов — fp32
        outs = [o.to(ref_dtype) if o.is_floating_point() else o for o in outs]
        return outs[0] if len(outs) == 1 else tuple(outs)


def compile(model, example_inputs=None, device="ane", **kw):  # noqa: A001 — как torch.compile
    """model -> CompiledModule. С example_inputs компиляция и сверка — сразу, а не на первом
    вызове. Остальные аргументы — как у CompiledModule (validate, compress, rules...)."""
    m = CompiledModule(model, device=device, **kw)
    if example_inputs is not None:
        m(*example_inputs)
    return m


def cleanup_mpsgraph_tmp(include_self=False):
    """Core AI/MPSGraph кладёт в $TMPDIR/com.apple.MetalPerformanceShadersGraph по папке
    mpsgraph-<pid>-<время> на загруженную программу (копия весов, до гигабайта) и не
    удаляет их — ни при выходе, ни при падении. Devstral кусками оставлял ~10 ГБ за
    каждый запуск. Удаляем папки завершившихся процессов (и свои — при выходе). С macOS 27.0.1
    часть таких папок (компиляция ANE-областей) появляется в ...GraphHost — смотрим обе."""
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
    """Отметить сборку как использованную (время изменения — для prune_cache)."""
    try:
        os.utime(path)
    except OSError:
        pass


def prune_cache(max_gb=None, keep=()):
    """Кеш сборок не больше max_gb (по умолчанию TORCH_ANE_CACHE_GB, 40): удаляются давно
    не использованные (по времени изменения; загрузка его обновляет), кроме путей keep.
    Каждая сборка — ~1 ГБ на модель среднего размера, и без лимита эксперименты за сутки
    съедали сотню гигабайт. Возвращает освобождённые байты."""
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
    """Системный кеш компиляции Core AI (~/Library/Caches/coreai-cache): по скомпилированной
    копии каждой когда-либо загруженной модели, сам не чистится (за неделю экспериментов —
    87 ГБ). Удалять можно только целиком: после частичного удаления Core AI перестаёт
    загружать модели. Первая загрузка каждой модели после очистки — с компиляцией."""
    root = pathlib.Path.home() / "Library" / "Caches" / "coreai-cache"
    freed = 0
    for d in root.iterdir() if root.is_dir() else ():
        if d.is_dir() and d.name not in ("sync", "tmp"):
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return freed


def probe_tmpdir():
    """Временная папка для экспериментов, удаляется при выходе из процесса (раньше пробы
    брали tempfile.mkdtemp() и оставляли по копии модели на гигабайт за запуск)."""
    import atexit
    import tempfile
    d = tempfile.mkdtemp(prefix="torch_ane_probe_")
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d
