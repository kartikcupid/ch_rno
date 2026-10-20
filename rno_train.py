#!/usr/bin/env python
"""RNO for the Cahn-Hilliard-Cook equation -- training.

Reads the dataset ch_gpu.py writes (Work/data/test_t<END>.npz) and writes the
model, its logs and its float32 memmaps into Work/models/train_t<END>_lap/
(the Laplacian-output model, the default) or Work/models/train_t<END>/
(--variant base, the direct-increment head).
Validation ground truth goes to the shared cache Work/gt_cache/.

    python rno_train.py                     END = CONFIG['train_end'] (1000)
    python rno_train.py --end 300
    python rno_train.py --end 1000 --variant base   direct-increment head
    python rno_train.py --end 1000 --width 24 --modes 12 --layers 3
    python rno_train.py --npz test.npz      override the dataset path
    python rno_train.py --smoke             writes only under Work/smoke/
    python rno_train.py --bench
    torchrun --nproc_per_node=4 rno_train.py --end 1000

The network is the Recurrent Neural Operator of neuralop.models.RNO
(Liu-Schiaffini et al. 2023, arXiv:2308.08794): a lifting channel MLP, a
stack of RNO layers whose GRU gates z, r and the SELU candidate are each the
sum of two FNO blocks (one on the input, one on the hidden state) and a
scalar bias, and a projection channel MLP.  RNOCore is a line-by-line port
of that class with its defaults (no positional embedding, no domain
padding); tests/test_rno.py loads the same weights into neuralop's RNO and
checks that both give the same output on the training grid.  Two things are
changed, both outside the GRU equations:

* SpectralConv keeps neuralop's weight layout and Hermitian projection on
  the reference grid n_ref = 64, but on any other grid it resamples the
  weight to the modes that carry the same physical wavenumbers 2 pi j / n.
  neuralop indexes its weights by mode number, so a model trained on the
  64-box would act on the 128-box at half the wavenumbers it learned (the
  defect described in section 1 of the UNO project's README).  Here the
  128-box operator is exactly the 64-box operator: model(tile(psi)) =
  tile(model(psi)) for the state and every hidden state.
* CHRNO wraps the core for the Cahn-Hilliard step: the input channels are
  psi, psi^3, the 5-point Laplacian of psi, the mean of psi and a learned
  embedding of the step dt broadcast as constant channels; the output M is
  scaled by a learned gain(dt), turned into the increment by the periodic
  5-point Laplacian (CONFIG['laplacian_output']) and made mass conserving.
  The hidden states of every layer are returned and carried from step to
  step along a rollout, as RNO.predict does.

Training draws chains psi(t0), psi(t1), ... from the l = 64 trajectories,
with a constant dt or with a stride that grows like alpha t as the rollout
schedule does (CONFIG['chain_sched_frac']), starts each chain from an empty
memory and backpropagates through the last grad_steps transitions; the
unroll curriculum lengthens the chains so the memory sees more history.

Checkpoint selection (CONFIG['select'] = "survival"): after every validation
rollout the epoch with the longest mean survival time (linear interpolation of
the first crossing of val_threshold, see interp_survival) becomes best.pt; an
epoch within select_min_delta of the best only replaces it with a higher
in-range accuracy (mean R^2 clipped to [0, 1] up to t = 2 END).  The unclipped
mean R^2 over the whole validation horizon is still logged, for reference.
A validation rollout that goes non-finite or past max |psi| = diverge_abs is
scored per seed (R^2 = -1 and failure from its own step on).

The path helpers (RESULTS, WORK, GT_DIR, variant_tag, variant_name, run_tag,
train_label, run_label, data_path, model_dir, predict_dir, rollout_dir,
report_dir, PAIRS) and the ground-truth cache (ground_truth, ground_truth_batch,
find_gt, load_gt) are imported by rno_predict.py and make_submission.py.  The
ground-truth file format and names are those of the UNO project, so its
Work/gt_cache can be shared (README, "HPC").
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import sys
import time
import zipfile
from datetime import datetime
from typing import Any

if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib import format as npformat


PRECISION = "fp32"

RESULTS = "Results"
WORK = "Work"
ARCHIVE = "Archive"
GT_DIR = os.path.join(WORK, "gt_cache")
SMOKE_ROOT = os.path.join(WORK, "smoke")
SMOKE_RESULTS = os.path.join(SMOKE_ROOT, RESULTS)
DEFAULT_WIDTH = 32
DEFAULT_MODES = 16
DEFAULT_LAYERS = 4
PAIRS = {300: 10000.0, 500: 10000.0, 1000: 10000.0, 2000: 10000.0}


VARIANTS = ("lap", "base")


def variant_tag(variant=""):
    """'_lap' for the Laplacian-output model, '' for the base one."""
    v = (variant or "").lower()
    if v in ("", "base"):
        return ""
    if v == "lap":
        return "_lap"
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def variant_name(laplacian_output):
    """'lap' or 'base', the --variant value of an architecture."""
    return "lap" if laplacian_output else "base"


def run_tag(width=DEFAULT_WIDTH, modes=DEFAULT_MODES, seed=0,
            layers=DEFAULT_LAYERS, context=0):
    """'' for the default (width 32, modes 16, 4 layers, no context, seed 0);
    else 'w<W>', 'm<M>', 'd<L>' (depth, the number of RNO layers), 'c<K>'
    (K ground-truth frames before t_start warm the memory) and 's<N>' in
    that order, without the underscore: 'm24', 'w24', 'w24m12d3', 'c5',
    'w32s1', 's3'."""
    if int(seed) < 0:
        raise SystemExit(f"--seed must be >= 0 (got {int(seed)})")
    if int(context) < 0:
        raise SystemExit(f"--context must be >= 0 (got {int(context)})")
    tag = ""
    if int(width) != DEFAULT_WIDTH:
        tag += f"w{int(width)}"
    if int(modes) != DEFAULT_MODES:
        tag += f"m{int(modes)}"
    if int(layers) != DEFAULT_LAYERS:
        tag += f"d{int(layers)}"
    if int(context) != 0:
        tag += f"c{int(context)}"
    if int(seed) != 0:
        tag += f"s{int(seed)}"
    return tag


def train_label(end, variant="", tag=""):
    """rno_t0300 for END = 300; rno_t0300_lap for the lap variant;
    rno_t0300_lap_m24 for the lap variant with 24 Fourier modes.  The rno_
    prefix keeps every folder apart from the UNO project's train_t... runs,
    so the two can be copied side by side."""
    t = f"_{tag}" if tag else ""
    return f"rno_t{int(round(end)):04d}{variant_tag(variant)}{t}"


def run_label(end, t_end, variant="", tag=""):
    """rno_t1000_lap_pred_t10000 for a lap model trained to 1000 and
    predicted to 10000; rno_t1000_pred_t10000 for the base head;
    rno_t1000_lap_m24_pred_t10000 for a size-tagged one."""
    return (f"{train_label(end, variant, tag)}_pred_"
            f"t{int(round(t_end)):05d}")


def data_path(end, root=WORK):
    return os.path.join(root, "data", f"test_t{int(round(end)):04d}.npz")


def memmap_dir(end, root=WORK):
    """float32 memmaps of one dataset, shared by every model of that END."""
    return os.path.join(root, "data", f"memmap_t{int(round(end)):04d}")


def model_dir(end, root=WORK, variant="", tag=""):
    return os.path.join(root, "models", train_label(end, variant, tag))


def gt_dir(root=WORK):
    return os.path.join(root, "gt_cache")


def predict_dir(label, root=WORK):
    return os.path.join(root, "predict", label)


def rollout_dir(label, root=WORK):
    return os.path.join(root, "rollout_cache", label)


def report_dir(label, results=RESULTS):
    return os.path.join(results, label)


def default_t_end(end):
    """PAIRS[END]: the horizon a model trained to END is evaluated on.

    Every model is evaluated to t = 10000, also an END PAIRS does not list.
    """
    return PAIRS.get(int(round(end)), 10000.0)


def horizon_frames(end):
    """Frames per trajectory in train_X of a ch_gpu.py dataset made to END."""
    return 10 * int(round(end)) - 10


def data_horizon(n_frames):
    """Inverse of horizon_frames: END of a dataset with n_frames in train_X."""
    return float(round(0.1 * (n_frames + 10)))


CONFIG: dict[str, Any] = dict(
    train_end      = 1000,
    npz            = "",
    run_name       = "rno_ch",

    solver_dt      = 0.01,
    gap            = 10,
    off_values     = (-0.4, -0.2, 0.0, 0.2, 0.4),
    train_off      = None,

    t_min          = 35.0,
    dt_min         = 1.0,
    dt_max         = 50.0,
    chain_sched_frac = 0.5,
    sched_alphas   = (0.05, 0.70),
    context        = 0,
    context_dt     = 4.0,
    in_ram         = True,
    data_on_gpu    = True,

    width          = DEFAULT_WIDTH,
    modes          = DEFAULT_MODES,
    layers         = DEFAULT_LAYERS,
    lifting_ratio  = 2.0,
    projection_ratio = 2.0,
    mlp_expansion  = 0.5,
    rno_skip       = False,
    cond_dim       = 32,
    cond_ch        = 8,
    conserve_mass  = True,
    laplacian_output = True,

    epochs         = 60,
    steps_per_epoch= 200,
    batch          = 128,
    lr             = 2.0e-3,
    lr_min         = 1.0e-5,
    unroll_stages  = ((0.00, 1), (0.08, 2), (0.19, 4), (0.45, 8)),
    stage_lr_mult  = 0.65,
    grad_steps     = 4,
    weight_decay   = 1.0e-4,
    grad_clip      = 1.0,
    ema_decay      = 0.999,
    noise          = 0.015,
    w_inc          = 0.25,
    w_grad         = 0.25,
    seed           = 0,

    val_seeds      = (31337, 24601, 8675309, 20250, 42424,
                      90210, 23571, 55555, 48151, 62831),
    val_l          = 128,
    val_t0         = 100.0,
    val_t1         = 0.0,
    val_alpha      = 0.20,
    val_dt_max     = 40.0,
    val_save_every = 4.0,
    val_threshold  = 0.80,
    diverge_abs    = 10.0,
    val_every      = 2,
    select         = "survival",
    select_min_delta = 1.0,
    patience       = 0,

    resume         = True,
    gt_require_cuda= "auto",
)

SMOKE: dict[str, Any] = dict(
    epochs=6, steps_per_epoch=25, val_every=3,
    patience=3, val_t1=400.0, val_save_every=10.0, val_seeds=(31337,),
    width=8, modes=8, layers=2, cond_ch=4, batch=8, grad_steps=2,
    unroll_stages=((0.0, 1), (0.3, 2), (0.6, 4)), train_off=(0.0,))

_DTYPES = {"fp32": torch.float32, "tf32": torch.float32, "fp64": torch.float64}


def set_precision(mode=None):
    """Apply PRECISION globally.  Returns (mode, torch dtype)."""
    mode = (mode or PRECISION).lower()
    if mode not in _DTYPES:
        raise ValueError(
            f"PRECISION must be one of {sorted(_DTYPES)}, got {mode!r}")
    use_tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    return mode, _DTYPES[mode]


class Tee:
    """Duplicate a stream to a file so the run log lands next to its outputs."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


def init_distributed():
    """Returns (rank, world_size, local_rank, device); works with torchrun."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def to_project_dir():
    """Work from the folder holding these scripts, whatever the launch cwd.

    Results/, Work/ and Archive/ are relative paths, so an IDE run button or
    `python ch_rno/rno_predict.py` would otherwise miss the checkpoint and
    scatter empty folders into the launch directory.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


def setup_log(out_dir, stem, rank=0):
    """Create out_dir and tee stdout/stderr into <out_dir>/<stem>.log."""
    out = os.path.abspath(out_dir)
    os.makedirs(out, exist_ok=True)
    name = f"{stem}.log" if rank == 0 else f"{stem}.rank{rank}.log"
    fh = open(os.path.join(out, name), "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    return out


def setup_output(cfg, rank=0, kind="train"):
    """Create cfg['results_dir'] (+cache) and tee into <run_name>_<kind>.log."""
    out = os.path.abspath(cfg["results_dir"])
    os.makedirs(os.path.join(out, "cache"), exist_ok=True)
    return setup_log(out, f"{cfg['run_name']}_{kind}", rank)


def log(*a, rank=0):
    if rank == 0:
        print(*a, flush=True)


def make_ic(l, seed, off=0.0):
    """Reproduces PhaseOrdering.set_ic from ch.py."""
    np.random.seed(seed)
    ic = np.random.uniform(-0.1, 0.1, size=(l, l))
    return ic - ic.mean() + off


def _del_sq_t(psi):
    return (torch.roll(psi, 1, -2) + torch.roll(psi, -1, -2)
            + torch.roll(psi, 1, -1) + torch.roll(psi, -1, -1) - 4.0 * psi)


@torch.no_grad()
def integrate(ic, t_end, save_times, dt=0.01, device="cuda", sink=None):
    """Explicit Euler in float64, operation for operation as in ch.py.

    Returns (times, frames) with the frames cast to float32.  With sink, each
    saved frame goes to sink(j, frame) as soon as it is computed instead, and
    frames is None, so host memory holds one frame rather than all of them.
    """
    dev = torch.device(device)
    if dev.type == "mps":
        dev = torch.device("cpu")
    psi = torch.as_tensor(np.ascontiguousarray(ic), dtype=torch.float64,
                          device=dev)
    save = {int(round(t / dt)): i for i, t in enumerate(save_times)}
    out = None
    if sink is None:
        out = np.empty((len(save_times),) + tuple(psi.shape), dtype=np.float32)
    if 0 in save:
        if out is not None:
            out[save[0]] = psi.cpu().numpy()
        else:
            sink(save[0], psi.cpu().numpy().astype(np.float32))
    for step in range(1, int(round(t_end / dt)) + 1):
        mu = psi**3 - psi - _del_sq_t(psi)
        psi = psi + _del_sq_t(mu) * dt
        j = save.get(step)
        if j is not None:
            frame = psi.cpu().numpy().astype(np.float32)
            if out is not None:
                out[j] = frame
            else:
                sink(j, frame)
    return np.asarray(save_times, dtype=np.float64), out


def savez_atomic(path, **arrays):
    """np.savez to a temporary name, then rename: a killed job never leaves a
    truncated file behind that a later run would mistake for a cached one."""
    tmp = path[:-4] + ".partial.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def free_name(path):
    """path, or path with a timestamp before its extension if that exists.

    Rotations (best -> _prev, state -> _stale) use it so they never
    overwrite an earlier rotated file.
    """
    root, ext = os.path.splitext(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cand, k = path, 1
    while os.path.exists(cand):
        cand = f"{root}_{stamp}{ext}" if k == 1 else f"{root}_{stamp}-{k}{ext}"
        k += 1
    return cand


def gt_path(cache_dir, l, off, seed, t0, t1, save_every):
    return os.path.join(cache_dir, f"gt_l{l}_off{off:+.1f}_seed{seed}"
                                   f"_t{t0:g}-{t1:g}_ev{save_every:g}.npz")


def gt_times(t0, t1, save_every):
    """The frame times of a ground-truth file, exactly as it stores them."""
    return np.arange(t0, t1 + 0.5 * save_every, save_every).round(6)


def find_gt(cache_dir, l, off, seed, t0, t1, save_every):
    """Path of a cached file that holds this trajectory, or None.

    The exact name wins.  Otherwise any file with the same (l, off, seed, t0,
    save_every) and a longer t1 serves, the shortest such: its first frames
    are the same solve, bit for bit.  load_gt(path, t1) slices it.
    """
    path = gt_path(cache_dir, l, off, seed, t0, t1, save_every)
    if os.path.exists(path):
        return path
    k = (t1 - t0) / save_every
    if abs(k - round(k)) > 1e-6 or not os.path.isdir(cache_dir):
        return None
    pre = f"gt_l{l}_off{off:+.1f}_seed{seed}_t{t0:g}-"
    suf = f"_ev{save_every:g}.npz"
    best = None
    for name in os.listdir(cache_dir):
        if not (name.startswith(pre) and name.endswith(suf)):
            continue
        try:
            t_long = float(name[len(pre):-len(suf)])
        except ValueError:
            continue
        if t_long >= t1 - 1e-9 and (best is None or t_long < best[0]):
            best = (t_long, name)
    return None if best is None else os.path.join(cache_dir, best[1])


def _stored_npy_offset(path, member):
    """Byte offset of an uncompressed .npz member's .npy data, else None."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        start = info.header_offset
    with open(path, "rb") as f:
        f.seek(start)
        head = f.read(30)
        if head[:4] != b"PK\x03\x04":
            return None
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        f.seek(start + 30 + n_name + n_extra)
        major, _ = npformat.read_magic(f)
        if major == 1:
            shape, fortran, dtype = npformat.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = npformat.read_array_header_2_0(f)
        return f.tell(), tuple(shape), fortran, dtype


def load_gt(path, t1=None):
    """(times, fields) of a ground-truth file; fields is a read-only memmap.

    Nothing but the frames that get indexed is read from disk.  With t1 the
    arrays stop at times <= t1 (a longer cached solve serves a shorter one).
    A compressed member falls back to np.load.
    """
    with np.load(path) as z:
        times = np.asarray(z["times"])
    loc = _stored_npy_offset(path, "fields.npy")
    if loc is None or loc[2]:
        with np.load(path) as z:
            fields = z["fields"]
    else:
        start, shape, _, dtype = loc
        fields = np.memmap(path, dtype=dtype, mode="r", offset=start,
                           shape=shape)
    if t1 is not None:
        n = int(np.count_nonzero(times <= t1 + 1e-6))
        times, fields = times[:n], fields[:n]
    return times, fields


def gt_provenance(path):
    """Solver metadata a ground-truth file records; None where it predates it."""
    keys = ("device", "solver_dtype", "solver_dt", "torch_version")
    with np.load(path) as z:
        return {k: (z[k].item() if k in z.files else None) for k in keys}


_GT_CPU_NOTED = [False]


def gt_require(require=None):
    """Whether ground truth must be solved on CUDA; 'auto' = inside Slurm."""
    if require is None:
        require = CONFIG["gt_require_cuda"]
    if isinstance(require, str) and require.lower() == "auto":
        return bool(os.environ.get("SLURM_JOB_ID"))
    return bool(require)


def gt_device(device, require=None):
    """The device ground truth really runs on (MPS falls back to the CPU).

    When CUDA is required (see gt_require) and not available, stop instead of
    spending hours on a silent CPU solve.
    """
    dev = torch.device(device)
    if dev.type == "mps":
        dev = torch.device("cpu")
    if dev.type == "cuda" and torch.cuda.is_available():
        return dev
    if gt_require(require):
        raise SystemExit(
            f"[gt] ground truth must be solved on CUDA here "
            f"(gt_require_cuda={require or CONFIG['gt_require_cuda']!r}, "
            f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')!r}) but the device "
            f"is {device} and torch.cuda.is_available() is "
            f"{torch.cuda.is_available()}. Request a GPU node, or set "
            f"gt_require_cuda = False to allow a CPU solve.")
    if not _GT_CPU_NOTED[0]:
        log("[gt] solving ground truth on the CPU (float64)")
        _GT_CPU_NOTED[0] = True
    return torch.device("cpu")


def device_label(dev):
    """'cuda:0 NVIDIA A100-SXM4-40GB', or 'cpu'."""
    dev = torch.device(dev)
    if dev.type != "cuda":
        return dev.type
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return f"cuda:{idx} {torch.cuda.get_device_name(idx)}"


class _GTWriter:
    """One ground-truth .npz written frame by frame, then renamed into place.

    The zip is uncompressed (ZIP_STORED) with the members np.savez writes, so
    np.load and load_gt read it like any other cached file.  The temporary
    name carries the host and pid: two jobs solving the same file write
    separate partials, and the last rename installs identical bytes.
    """

    def __init__(self, path, times, shape):
        tag = f"{socket.gethostname()}-{os.getpid()}"
        self.path, self.tmp = path, f"{path[:-4]}.{tag}.partial.npz"
        self.n, self.shape, self.done = len(times), tuple(shape), 0
        self.zf = zipfile.ZipFile(self.tmp, "w", zipfile.ZIP_STORED,
                                  allowZip64=True)
        self._put("times", np.asarray(times, dtype=np.float64))
        self.fh = self.zf.open("fields.npy", "w", force_zip64=True)
        npformat.write_array_header_1_0(self.fh, dict(
            descr=npformat.dtype_to_descr(np.dtype(np.float32)),
            fortran_order=False, shape=(self.n,) + self.shape))

    def _put(self, name, value):
        with self.zf.open(f"{name}.npy", "w", force_zip64=True) as f:
            npformat.write_array(f, np.asanyarray(value), allow_pickle=False)

    def frame(self, j, arr):
        assert j == self.done and arr.shape == self.shape, (j, self.done)
        self.fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        self.done += 1

    def close(self, **small):
        assert self.done == self.n, (self.path, self.done, self.n)
        self.fh.close()
        for k, v in small.items():
            self._put(k, v)
        self.zf.close()
        os.replace(self.tmp, self.path)

    def abort(self):
        try:
            self.fh.close()
            self.zf.close()
        except Exception:
            pass
        try:
            os.remove(self.tmp)
        except OSError:
            pass


def ground_truth(cache_dir, l, off, seed, t0, t1, save_every, dt, device,
                 rank=0, quiet=False, require_cuda=None):
    """Exact trajectory, cached to .npz and reused on later runs.

    Returns the path of a file that holds it, possibly a longer solve of the
    same seed: read it with load_gt(path, t1).
    """
    path = find_gt(cache_dir, l, off, seed, t0, t1, save_every)
    if path is not None:
        if not quiet:
            log(f"[data] ground truth cached: {os.path.basename(path)}",
                rank=rank)
        return path
    if rank == 0:
        ground_truth_batch(cache_dir, l, off, [seed], t0, t1, save_every, dt,
                           device, require_cuda=require_cuda)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    path = find_gt(cache_dir, l, off, seed, t0, t1, save_every)
    return path or gt_path(cache_dir, l, off, seed, t0, t1, save_every)


def ground_truth_batch(cache_dir, l, off, seeds, t0, t1, save_every, dt,
                       device, cells=1.6e7, max_open=128, require_cuda=None):
    """Solve every uncached seed together; each file matches ground_truth().

    The Laplacian acts on the last two axes, so a batch of trajectories is
    integrated elementwise exactly as each would be alone: the stored fields
    are bit-identical to one-at-a-time solves.  Frames stream straight into
    per-seed files, so host memory holds one frame per seed and a batch is
    capped only by cells (l*l per seed, the device-side state) and by
    max_open files.  A seed already held by a longer cached solve counts as
    present.  Returns the number of trajectories solved.
    """
    todo: list[int] = []
    for s in seeds:
        s = int(s)
        if s not in todo and find_gt(cache_dir, l, off, s, t0, t1,
                                     save_every) is None:
            todo.append(s)
    if not todo:
        return 0
    os.makedirs(cache_dir, exist_ok=True)
    dev = gt_device(device, require_cuda)
    label = device_label(dev)
    times = gt_times(t0, t1, save_every)
    steps = int(round(t1 / dt))
    per = max(1, min(int(cells // (l * l)), int(max_open)))
    n_batch = -(-len(todo) // per)
    log(f"[gt] solving {len(todo)} ground-truth trajectories at l={l} "
        f"off={off:+.1f} to t={t1:g} ({len(times)} frames, every "
        f"{save_every:g}) on {label}, {n_batch} batch(es) of <= {per}")
    small = dict(off=off, l=l, device=label, solver_dtype="float64",
                 solver_dt=float(dt), torch_version=torch.__version__)
    for k in range(n_batch):
        chunk = [s for s in todo[k * per:(k + 1) * per]
                 if find_gt(cache_dir, l, off, s, t0, t1, save_every) is None]
        if not chunk:
            log(f"[gt]   batch {k + 1}/{n_batch}: solved meanwhile by another "
                f"job, skipped")
            continue
        t_wall = time.time()
        ws = [_GTWriter(gt_path(cache_dir, l, off, s, t0, t1, save_every),
                        times, (l, l)) for s in chunk]

        def sink(j, frame, ws=ws):
            for b, w in enumerate(ws):
                w.frame(j, frame[b])

        try:
            ic = np.stack([make_ic(l, s, off) for s in chunk])
            integrate(ic, t1, list(times), dt=dt, device=dev, sink=sink)
            for w, s in zip(ws, chunk):
                w.close(seed=s, **small)
        except BaseException:
            for w in ws:
                w.abort()
            raise
        log(f"[gt]   batch {k + 1}/{n_batch} on {label}: {len(chunk)} seeds "
            f"{chunk}, {steps:,} steps, {time.time() - t_wall:.0f}s")
    return len(todo)


def frame_dt(cfg):
    return cfg["gap"] * cfg["solver_dt"]


def npz_member_shape(src, member):
    """Shape of one array inside an .npz, read from its header alone."""
    with zipfile.ZipFile(src) as zf, zf.open(member) as f:
        npformat.read_magic(f)
        return tuple(npformat.read_array_header_1_0(f)[0])


def build_cache(cfg, rank=0):
    """Stream the dataset into float32 memmaps in cfg['memmap_dir'].

    train_y is a shifted copy of X, so only train_X and val_X are read.  The
    memmaps live in Work/data/memmap_t<END>/ and are shared by every model
    trained to that END (3.3 GB for END = 1000).  Existing memmaps are
    reused only while their shape still matches the dataset.  Without the
    dataset they are reused only if their frame count is the one
    cfg['train_end'] implies; anything else stops the run.  Each writer
    fills its own host-pid partial file and renames it into place, so two
    jobs that start together install identical bytes.
    """
    mdir = cfg["memmap_dir"]
    paths = {s: os.path.join(mdir, f"{s}_X_f32.npy") for s in ("train", "val")}
    src, end = cfg["npz"], int(round(cfg["train_end"]))
    hint = (f"run `python ch_gpu.py --end {end}` (it writes {data_path(end)}), "
            f"or pass --npz PATH")
    if all(os.path.exists(p) for p in paths.values()):
        frames = {s: np.load(p, mmap_mode="r").shape[1]
                  for s, p in paths.items()}
        if not os.path.exists(src):
            if set(frames.values()) == {horizon_frames(end)}:
                log(f"[data] {src} not found; reusing the memmaps in "
                    f"{mdir} ({frames['train']} frames = t {end})", rank=rank)
                return paths
            raise SystemExit(
                f"{src} not found, and the memmaps in {mdir} hold {frames} "
                f"frames, not {horizon_frames(end)} (END = {end}): {hint}")
        stale = [s for s in paths
                 if np.load(paths[s], mmap_mode="r").shape
                 != npz_member_shape(src, f"{s}_X.npy")]
        if not stale:
            log(f"[data] reusing the memmaps in {mdir}", rank=rank)
            return paths
        log(f"[data] cached {stale} memmaps do not match {src}; rebuilding",
            rank=rank)
    if not os.path.exists(src):
        raise SystemExit(f"{src} not found: {hint}")
    if rank == 0:
        os.makedirs(mdir, exist_ok=True)
        log(f"[data] reading {src} "
            f"({os.path.getsize(src) / 1e9:.2f} GB on disk) into {mdir}",
            rank=rank)
        tag = f"{socket.gethostname()}-{os.getpid()}"
        with zipfile.ZipFile(src) as zf:
            for split, member in (("train", "train_X.npy"),
                                  ("val", "val_X.npy")):
                with zf.open(member) as f:
                    npformat.read_magic(f)
                    shape, fortran, dtype = npformat.read_array_header_1_0(f)
                    assert not fortran and dtype == np.float64
                    per = int(np.prod(shape[1:]))
                    tmp = paths[split][:-4] + f".{tag}.partial.npy"
                    arr = np.lib.format.open_memmap(
                        tmp, mode="w+", dtype=np.float32, shape=shape)
                    for j in range(shape[0]):
                        buf = f.read(per * 8)
                        arr[j] = np.frombuffer(buf, np.float64).reshape(
                            shape[1:]).astype(np.float32)
                    arr.flush()
                    del arr
                    os.replace(tmp, paths[split])
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return paths


def off_of_trajectory(n_traj, off_values):
    """Per-trajectory mean order parameter, mirroring ch.py's array_split."""
    off = np.empty(n_traj, dtype=np.float64)
    for value, idx in zip(off_values,
                          np.array_split(np.arange(n_traj), len(off_values))):
        off[idx] = value
    return off


def select_trajectories(n_traj, cfg):
    """Indices whose quench is in cfg['train_off']; None selects all."""
    offs = off_of_trajectory(n_traj, cfg["off_values"])
    want = cfg["train_off"]
    if want is None:
        return np.arange(n_traj), offs, True
    want_arr = np.round(np.asarray(want, dtype=np.float64), 6)
    keep = np.nonzero(np.isin(np.round(offs, 6), want_arr))[0]
    if keep.size == 0:
        raise SystemExit(f"train_off={want} selects no trajectory; "
                         f"available off values are {sorted(set(offs))}")
    symmetric = set(np.round(-want_arr, 6)) == set(want_arr)
    return keep, offs[keep], symmetric


class PairSampler:
    """Draws chains psi(t0), psi(t1), ... from the cached trajectories.

    A chain has a constant stride (log-uniform in [dt_min, max_dt]) or, for
    a fraction sched_frac of the batch, a stride that grows like alpha t_k
    (alpha log-uniform in sched_alphas, clipped to [dt_min, dt_max]) the way
    the rollout schedule does, so the memory also sees strides that change
    from one step to the next.  Every stride is a whole number of frames and
    the chain always fits inside its trajectory.  With cfg['context'] = K
    the K frames context_dt apart before the chain start come back too, for
    warm_memory.
    """

    def __init__(self, path, cfg, device, dtype, seed):
        arr = np.load(path, mmap_mode="r")
        keep, self.offs, self.sign_ok = select_trajectories(arr.shape[0], cfg)
        data = np.ascontiguousarray(arr[keep]) if cfg["in_ram"] else arr[keep]
        self.kept = keep
        self.data = torch.from_numpy(np.ascontiguousarray(data))
        if cfg["data_on_gpu"] and device.type == "cuda":
            self.data = self.data.to(device)
        self.n_traj, self.n_frames = self.data.shape[:2]
        self.l = self.data.shape[-1]
        self.device, self.dtype = device, dtype
        self.rng = np.random.default_rng(seed)
        self.fdt = frame_dt(cfg)
        self.t_first = cfg["solver_dt"]
        self.f_min = int(round((cfg["t_min"] - cfg["solver_dt"]) / self.fdt))
        self.df_min = max(1, int(round(cfg["dt_min"] / self.fdt)))
        self.df_max = int(round(cfg["dt_max"] / self.fdt))
        self.context = int(cfg.get("context", 0))
        self.cdf = max(1, int(round(float(cfg.get("context_dt", 4.0))
                                    / self.fdt)))
        self.f_ctx = self.context * self.cdf
        self.f_min = max(self.f_min, self.f_ctx)
        self.sched_frac = float(cfg.get("chain_sched_frac", 0.0))
        self.alphas = tuple(float(x) for x in cfg.get("sched_alphas",
                                                       (0.05, 0.7)))

    def max_dt(self, n_steps):
        """Largest dt a chain of n_steps fits into one training trajectory."""
        df_hi = max(self.df_min,
                    min(self.df_max,
                        (self.n_frames - 1 - self.f_min) // n_steps))
        return df_hi * self.fdt

    def _strides(self, batch, n_steps, dt):
        """(f0 (b,), df (b, n_steps)) frame indices and frame strides."""
        df_hi = max(self.df_min,
                    min(self.df_max,
                        (self.n_frames - 1 - self.f_min) // n_steps))
        if dt is not None:
            d = min(max(1, int(round(dt / self.fdt))), df_hi)
            df = np.full((batch, n_steps), d, dtype=np.int64)
        else:
            lo, hi = math.log(self.df_min), math.log(df_hi)
            d = np.clip(np.rint(np.exp(self.rng.uniform(lo, hi, batch))),
                        self.df_min, df_hi).astype(np.int64)
            df = np.repeat(d[:, None], n_steps, axis=1)
        top = self.n_frames - 1 - df.sum(axis=1)
        lo = np.maximum(self.f_ctx, np.minimum(self.f_min, top))
        if (top < lo).any():
            raise SystemExit(f"a chain of {n_steps} steps of at least "
                             f"{self.df_min} frames does not fit into "
                             f"{self.n_frames} frames")
        f0 = self.rng.integers(lo, top + 1)
        if dt is not None or self.sched_frac <= 0.0:
            return f0, df
        sched = self.rng.random(batch) < self.sched_frac
        if not sched.any():
            return f0, df
        n = int(sched.sum())
        top = self.n_frames - 1 - n_steps * self.df_min
        lo = max(self.f_ctx, min(self.f_min, top))
        f = self.rng.integers(lo, top + 1, n)
        a = np.exp(self.rng.uniform(math.log(self.alphas[0]),
                                    math.log(self.alphas[1]), n))
        start = f.copy()
        steps = np.empty((n, n_steps), dtype=np.int64)
        for k in range(n_steps):
            room = (self.n_frames - 1 - f) // (n_steps - k)
            want = np.rint(a * (f * self.fdt + self.t_first) / self.fdt)
            d = np.clip(want, self.df_min, self.df_max).astype(np.int64)
            d = np.maximum(self.df_min, np.minimum(d, room))
            steps[:, k] = d
            f = f + d
        f0[sched], df[sched] = start, steps
        return f0, df

    def sample(self, batch, n_steps=1, dt=None):
        """([psi(t0), ..., psi(t_n)] each (b, 1, l, l), dt (b, n_steps),
        [psi(t0 - K cdt), ..., psi(t0 - cdt)] the context, [] for K = 0)."""
        f0, df = self._strides(batch, n_steps, dt)
        traj = self.rng.integers(0, self.n_traj, batch)
        src = self.data.device
        traj_i = torch.as_tensor(traj, device=src)
        frames = np.concatenate([f0[:, None], f0[:, None]
                                 + np.cumsum(df, axis=1)], axis=1)
        chain = []
        for k in range(n_steps + 1):
            idx_i = torch.as_tensor(frames[:, k], device=src)
            x = self.data[traj_i, idx_i]
            chain.append(x.to(self.device, dtype=self.dtype).unsqueeze(1))
        warm = []
        for k in range(self.context, 0, -1):
            idx_i = torch.as_tensor(f0 - k * self.cdf, device=src)
            x = self.data[traj_i, idx_i]
            warm.append(x.to(self.device, dtype=self.dtype).unsqueeze(1))
        dt_t = torch.as_tensor(df * self.fdt, dtype=self.dtype,
                               device=self.device)
        return chain, dt_t, warm

    def augment(self, chain, gen):
        """D4 lattice symmetry, plus psi -> -psi when the quench set allows."""
        k = int(torch.randint(0, 4, (1,), generator=gen, device=self.device))
        if k:
            chain = [torch.rot90(c, k, dims=(-2, -1)) for c in chain]
        if torch.rand((), generator=gen, device=self.device) < 0.5:
            chain = [torch.flip(c, dims=(-1,)) for c in chain]
        if self.sign_ok and torch.rand((), generator=gen,
                                       device=self.device) < 0.5:
            chain = [-c for c in chain]
        return chain


def _hat(u, nodes):
    """(len(u), len(nodes)) linear-interpolation weights, zero outside."""
    return torch.clamp(1.0 - (u[:, None] - nodes[None, :]).abs(), min=0.0)


class SpectralConv(nn.Module):
    """neuralop's SpectralConv (2-D, dense, real data) on physical wavenumbers.

    On the reference grid n_ref it is neuralop.layers.spectral_convolution.
    SpectralConv with n_modes = (m1, m2) and its defaults: the complex weight
    (in, out, m1, m2 // 2 + 1) acts on rows -m1//2 .. m1 - m1//2 - 1 of the
    first FFT axis and columns 0 .. m2//2 of the rfft axis, the inverse is
    an ifft over the first axis, a real m2 = 0 (and Nyquist) column and an
    irfft (enforce_hermitian_symmetry), with norm = 'forward', plus a real
    bias.  The weight is stored as a real (..., 2) tensor.

    On a grid n the row and column j carry the wavenumber 2 pi j / n, i.e.
    the reference index j n_ref / n, and the weight is resampled linearly to
    every such index that lies inside the trained band.  For n = 2 n_ref the
    even rows and columns are exactly the trained weights, so a tiled state
    sees exactly the reference operator.
    """

    def __init__(self, in_ch, out_ch, modes, n_ref):
        super().__init__()
        m1, m2 = (modes, modes) if isinstance(modes, int) else modes
        self.in_ch, self.out_ch, self.n_ref = in_ch, out_ch, int(n_ref)
        self.r_lo, self.r_hi = -(m1 // 2), m1 - m1 // 2 - 1
        self.c_hi = m2 // 2
        std = (2.0 / (in_ch + out_ch)) ** 0.5
        w = std * torch.randn(in_ch, out_ch, m1, self.c_hi + 1,
                              dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(w).contiguous())
        self.bias = nn.Parameter(std * torch.randn(out_ch, 1, 1))
        self._cache: dict[Any, Any] = {}

    def _plan(self, n):
        """Kept rows (frequencies) and columns of grid n, and the index
        coordinates u = j n_ref / n they read on the reference grid."""
        s = n / self.n_ref
        j_lo = int(math.ceil(self.r_lo * s - 1e-9))
        j_hi = int(math.floor(self.r_hi * s + 1e-9))
        c_hi = int(math.floor(self.c_hi * s + 1e-9))
        assert -j_lo < n / 2 and j_hi < n / 2 and c_hi <= n // 2, (
            f"mode band rows {j_lo}..{j_hi}, cols 0..{c_hi} does not fit "
            f"grid {n}")
        rows = torch.arange(j_lo, j_hi + 1)
        cols = torch.arange(0, c_hi + 1)
        return rows, cols, s

    def _runtime_weight(self, n, device, dtype):
        key = (n, device, dtype)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == self.weight._version \
                and not torch.is_grad_enabled():
            return hit[1], hit[2]
        rows, cols, s = self._plan(n)
        real = torch.float64 if dtype == torch.complex128 else torch.float32
        w = torch.view_as_complex(self.weight.to(real))
        if n != self.n_ref:
            a = _hat(rows.to(real) / s,
                     torch.arange(self.r_lo, self.r_hi + 1).to(real))
            c = _hat(cols.to(real) / s,
                     torch.arange(0, self.c_hi + 1).to(real))
            a, c = a.to(w.device), c.to(w.device)
            w = torch.einsum("ab,iobd,cd->ioac", a.to(w.dtype), w,
                             c.to(w.dtype))
        w = w.to(device=device, dtype=dtype)
        idx = (rows % n).to(device)
        if not torch.is_grad_enabled():
            self._cache[key] = (self.weight._version, w, idx)
        return w, idx

    def forward(self, x):
        b, _, n, m = x.shape
        assert n == m, f"square grids only, got {n} x {m}"
        xf = torch.fft.rfft2(x, norm="forward")
        w, idx = self._runtime_weight(n, xf.device, xf.dtype)
        cols = w.shape[-1]
        blk = xf[:, :, idx, :cols]
        out = torch.einsum("bixy,ioxy->boxy", blk, w)
        full = torch.zeros(b, self.out_ch, n, n // 2 + 1, dtype=xf.dtype,
                           device=xf.device)
        full[:, :, idx, :cols] = out
        full = torch.fft.ifft(full, n=n, dim=-2, norm="forward")
        full[..., 0].imag.zero_()
        if n % 2 == 0:
            full[..., -1].imag.zero_()
        y = torch.fft.irfft(full, n=n, dim=-1, norm="forward")
        return y + self.bias.to(y.dtype)


class ChannelMLP(nn.Module):
    """neuralop.layers.channel_mlp.ChannelMLP: pointwise layers, GELU between.

    neuralop uses kernel-1 Conv1d layers on the flattened grid; kernel-1
    Conv2d layers compute the same map.
    """

    def __init__(self, in_ch, out_ch, hidden, n_layers=2):
        super().__init__()
        dims = ([in_ch, out_ch] if n_layers == 1
                else [in_ch] + [hidden] * (n_layers - 1) + [out_ch])
        self.fcs = nn.ModuleList([nn.Conv2d(dims[i], dims[i + 1], 1)
                                  for i in range(n_layers)])

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = F.gelu(x)
        return x


class FNOBlock(nn.Module):
    """neuralop FNOBlocks(n_layers=1) with the RNO defaults.

    fno_skip = 'linear' (bias-free 1x1), use_channel_mlp with expansion
    0.5, channel_mlp_skip = 'soft-gating', no norm, no preactivation, no
    stabilizer: y = MLP(K x + W x) + g * x, and no nonlinearity after the
    last (only) layer.
    """

    def __init__(self, ch, modes, n_ref, expansion=0.5):
        super().__init__()
        self.conv = SpectralConv(ch, ch, modes, n_ref)
        self.skip = nn.Conv2d(ch, ch, 1, bias=False)
        self.mlp = ChannelMLP(ch, ch, round(ch * expansion), 2)
        self.gate = nn.Parameter(torch.ones(1, ch, 1, 1))

    def forward(self, x):
        return self.mlp(self.conv(x) + self.skip(x)) + self.gate * x


class RNOCell(nn.Module):
    """neuralop.layers.rno_block.RNOCell:

        z = sigmoid(f1(x) + f2(h) + b1)
        r = sigmoid(f3(x) + f4(h) + b2)
        c = selu(f5(x) + f6(r * h) + b3)
        h' = (1 - z) * h + z * c

    with f1 .. f6 FNO blocks and b1 .. b3 scalar biases.
    """

    def __init__(self, ch, modes, n_ref, expansion=0.5):
        super().__init__()
        self.input_gates = nn.ModuleList(
            [FNOBlock(ch, modes, n_ref, expansion) for _ in range(3)])
        self.hidden_gates = nn.ModuleList(
            [FNOBlock(ch, modes, n_ref, expansion) for _ in range(3)])
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.randn(())) for _ in range(3)])

    def forward(self, x, h):
        ig, hg, b = self.input_gates, self.hidden_gates, self.biases
        z = torch.sigmoid(ig[0](x) + hg[0](h) + b[0])
        r = torch.sigmoid(ig[1](x) + hg[1](h) + b[1])
        c = F.selu(ig[2](x) + hg[2](r * h) + b[2])
        return (1.0 - z) * h + z * c


class RNOBlock(nn.Module):
    """neuralop.layers.rno_block.RNOBlock: the cell over a sequence.

    An absent hidden state starts as zeros plus the scalar bias_h.
    """

    def __init__(self, ch, modes, n_ref, expansion=0.5):
        super().__init__()
        self.ch = ch
        self.cell = RNOCell(ch, modes, n_ref, expansion)
        self.bias_h = nn.Parameter(torch.randn(()))

    def forward(self, x, h=None, return_sequences=False):
        b, steps = x.shape[:2]
        if h is None:
            h = torch.zeros((b, self.ch) + tuple(x.shape[3:]), dtype=x.dtype,
                            device=x.device) + self.bias_h.to(x.dtype)
        outs = []
        for i in range(steps):
            h = self.cell(x[:, i], h)
            if return_sequences:
                outs.append(h)
        return torch.stack(outs, dim=1) if return_sequences else h


class RNOCore(nn.Module):
    """neuralop.models.RNO with positional_embedding = None, no padding.

    forward(x, states) takes x (b, steps, in_ch, n, n) and one hidden state
    per layer (or None) and returns (pred (b, out_ch, n, n), states), which
    is RNO.forward(x, states, return_hidden_states=True,
    keep_states_padded=True): the state of a lower layer is its output at
    the last step (with rno_skip, the input plus that output, as in
    neuralop), the state of the top layer is its hidden state.
    """

    def __init__(self, in_ch, out_ch, width, modes, layers, n_ref,
                 lifting_ratio=2.0, projection_ratio=2.0, expansion=0.5,
                 rno_skip=False):
        super().__init__()
        self.width, self.n_layers, self.rno_skip = width, layers, rno_skip
        lift = int(lifting_ratio * width)
        self.lifting = (ChannelMLP(in_ch, width, lift, 2) if lift
                        else ChannelMLP(in_ch, width, width, 1))
        self.layers = nn.ModuleList([RNOBlock(width, modes, n_ref, expansion)
                                     for _ in range(layers)])
        self.projection = ChannelMLP(width, out_ch,
                                     int(projection_ratio * width), 2)

    def forward(self, x, states=None):
        b, steps = x.shape[:2]
        x = self.lifting(x.reshape((b * steps,) + tuple(x.shape[2:])))
        x = x.reshape((b, steps) + tuple(x.shape[1:]))
        new = []
        for i, layer in enumerate(self.layers):
            h0 = None if states is None else states[i]
            if i < self.n_layers - 1:
                out = layer(x, h0, return_sequences=True)
                x = x + out if self.rno_skip else out
                new.append(x[:, -1])
            else:
                new.append(layer(x, h0, return_sequences=False))
        return self.projection(new[-1]), new


def lap5(x):
    """Periodic 5-point Laplacian over the last two dims."""
    return (torch.roll(x, 1, -2) + torch.roll(x, -1, -2)
            + torch.roll(x, 1, -1) + torch.roll(x, -1, -1) - 4.0 * x)


def _dt_embedding(dt, dim):
    half = dim // 2
    logdt = torch.log(dt.clamp_min(1e-6))
    freqs = torch.exp(torch.linspace(0.0, math.log(64.0), half,
                                     device=dt.device, dtype=dt.dtype))
    ang = logdt[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang), logdt[:, None]], dim=-1)


class CHRNO(nn.Module):
    """(psi(t), memory) -> (psi(t + dt), memory); box-size agnostic, mass
    conserving.

    The RNO core sees psi, psi^3, lap5(psi), mean(psi) and cond_ch constant
    channels that embed dt, one time step per call, and its hidden states are
    the memory.  Its output M is scaled by gain(dt) = exp(log_gain +
    gain_exp log dt); with laplacian_output the increment is lap5(M), the
    form of every solver update, otherwise M itself.  The last projection
    layer starts at zero, so the untrained model is the identity map.
    """

    def __init__(self, width=DEFAULT_WIDTH, modes=DEFAULT_MODES,
                 layers=DEFAULT_LAYERS, n_ref=64, lifting_ratio=2.0,
                 projection_ratio=2.0, mlp_expansion=0.5, rno_skip=False,
                 cond_dim=32, cond_ch=8, conserve_mass=True,
                 laplacian_output=True):
        super().__init__()
        self.n_ref, self.n_layers = n_ref, layers
        self.conserve_mass = conserve_mass
        self.laplacian_output = laplacian_output
        emb = 2 * (cond_dim // 2) + 1
        self.cond_in = emb
        self.cond = nn.Sequential(nn.Linear(emb, cond_dim), nn.GELU(),
                                  nn.Linear(cond_dim, cond_ch))
        self.core = RNOCore(4 + cond_ch, 1, width, modes, layers, n_ref,
                            lifting_ratio, projection_ratio, mlp_expansion,
                            rno_skip)
        last = self.core.projection.fcs[-1]
        nn.init.zeros_(last.weight)
        assert last.bias is not None
        nn.init.zeros_(last.bias)
        self.log_gain = nn.Parameter(torch.tensor(math.log(0.012)))
        self.gain_exp = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _features(psi):
        lap = lap5(psi)
        mean = psi.mean(dim=(-2, -1), keepdim=True).expand_as(psi)
        return torch.cat([psi, psi ** 3, lap, mean], dim=1)

    def forward(self, psi, dt, state=None):
        if dt.ndim == 0:
            dt = dt.expand(psi.shape[0])
        dt = dt.to(psi.dtype)
        c = self.cond(_dt_embedding(dt, self.cond_in - 1))
        n = psi.shape[-1]
        x = torch.cat([self._features(psi),
                       c[:, :, None, None].expand(-1, -1, n, n)], dim=1)
        m, state = self.core(x.unsqueeze(1), state)
        gain = torch.exp(self.log_gain
                         + self.gain_exp * torch.log(dt.clamp_min(1e-6)))
        m = m * gain[:, None, None, None].to(m.dtype)
        inc = lap5(m) if self.laplacian_output else m
        if self.conserve_mass:
            inc = inc - inc.mean(dim=(-2, -1), keepdim=True)
        return psi + inc, state


def model_arch(cfg, n_ref=64):
    """The CHRNO keyword arguments of a training config (checkpoint 'arch')."""
    return dict(width=int(cfg["width"]), modes=int(cfg["modes"]),
                layers=int(cfg["layers"]), n_ref=int(n_ref),
                lifting_ratio=float(cfg["lifting_ratio"]),
                projection_ratio=float(cfg["projection_ratio"]),
                mlp_expansion=float(cfg["mlp_expansion"]),
                rno_skip=bool(cfg["rno_skip"]), cond_dim=int(cfg["cond_dim"]),
                cond_ch=int(cfg["cond_ch"]),
                conserve_mass=bool(cfg["conserve_mass"]),
                laplacian_output=bool(cfg["laplacian_output"]))


def mask_state(alive, new, old):
    """Per-sample select of two memories: new where alive, old elsewhere."""
    if old is None:
        return new
    keep = alive.view(-1, 1, 1, 1)
    return [torch.where(keep, a, b) for a, b in zip(new, old)]


def detach_state(state):
    return None if state is None else [h.detach() for h in state]


def _sq(x):
    return (x ** 2).sum(dim=(1, 2, 3))


def r2_loss(pred, true):
    den = _sq(true - true.mean(dim=(1, 2, 3), keepdim=True))
    return (_sq(pred - true) / den.clamp_min(1e-12)).mean()


def increment_loss(pred, true, inp):
    return (_sq(pred - true) / _sq(true - inp).clamp_min(1e-12)).mean()


def _fwd_diff(z):
    return torch.roll(z, -1, -2) - z, torch.roll(z, -1, -1) - z


def gradient_loss(pred, true):
    px, py = _fwd_diff(pred)
    tx, ty = _fwd_diff(true)
    return ((_sq(px - tx) + _sq(py - ty))
            / (_sq(tx) + _sq(ty)).clamp_min(1e-12)).mean()


def add_noise(psi, sigma):
    if sigma <= 0:
        return psi
    n = torch.randn_like(psi) * sigma
    return psi + (n - n.mean(dim=(-2, -1), keepdim=True))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)


def make_schedule(t0, t1, alpha, dt_min, dt_max, quantum=1.0, must_hit=()):
    """Times visited by an autoregressive rollout, stride ~ alpha * t."""
    ts, t = [float(t0)], float(t0)
    while t < t1 - 1e-9:
        dt = min(max(alpha * t, dt_min), dt_max)
        dt = max(quantum, round(dt / quantum) * quantum)
        t = min(t1, t + dt)
        ts.append(round(t, 6))
    ts += [float(m) for m in must_hit if t0 < m < t1]
    return np.array(sorted(set(ts)))


@torch.no_grad()
def eval_one_step(model, sampler, dts=(1.0, 10.0, 40.0), batch=8, reps=3):
    model.eval()
    acc = []
    for dt in dts:
        for _ in range(reps):
            chain, dt_t, warm = sampler.sample(batch, 1, dt=dt)
            state = warm_memory(model, warm, sampler.cdf * sampler.fdt)
            pred, true = model(chain[0], dt_t[:, 0], state)[0], chain[1]
            acc.append(float(r2_loss(pred, true)))
    model.train()
    return float(np.mean(acc))


@torch.no_grad()
def interp_survival(times, r2, threshold):
    """Survival time of one rollout with linear interpolation of the crossing.

    The first index i with r2[i] < threshold gives t[i-1] + (t[i] - t[i-1])
    * (r2[i-1] - thr) / (r2[i-1] - r2[i]).  A rollout that never drops below
    the threshold returns t[-1] (censored); a non-finite R^2 fails at its own
    step, and a drop at the first step returns t[0].
    """
    r2 = np.asarray(r2, dtype=np.float64)
    bad = np.nonzero(~(r2 >= threshold))[0]
    if not bad.size:
        return float(times[-1])
    i = int(bad[0])
    if i == 0 or not np.isfinite(r2[i]):
        return float(times[i])
    frac = (r2[i - 1] - threshold) / (r2[i - 1] - r2[i])
    return float(times[i - 1] + (times[i] - times[i - 1]) * frac)


def better_epoch(survive, acc, best_survive, best_acc, delta):
    """True when an epoch replaces the best one under select = "survival"."""
    if survive > best_survive + delta:
        return True
    return survive >= best_survive - delta and acc > best_acc


LOWK_DT = 48.0
LOWK_VAL_T0S = (1000.0, 2000.0)
LOWK_SHELLS = ("below", "first", "mid", "high")


def lowk_masks(l):
    """Boolean l x l masks of the four wavenumber shells (|n| = k l / 2 pi).

    below: |n| in {1, sqrt 2} (the 8 modes under the training box's
    fundamental at l = 128); first: |n| in [2, 2.9]; mid: [3, 8]; high: the
    rest.  The mean (n = 0) belongs to no shell.
    """
    n = np.fft.fftfreq(l) * l
    r = np.hypot(n[:, None], n[None, :])
    below = (r > 0.5) & (r < 1.5)
    first = (r >= 2.0) & (r <= 2.9)
    mid = (r >= 3.0) & (r <= 8.0)
    high = (r > 0.0) & ~(below | first | mid)
    return dict(below=below, first=first, mid=mid, high=high)


def lowk_plan(t, wanted, dt_want=LOWK_DT):
    """(dt, [t0, ...]) on the ground-truth frame grid t.

    dt is dt_want rounded to a multiple of the frame spacing; a wanted t0
    is kept when t0 and t0 + dt are both frames.  When none survives (a
    short smoke run) the middle frame of the usable range is used.
    """
    t = np.asarray(t, dtype=np.float64)
    q = float(t[1] - t[0])
    dt = max(q, round(dt_want / q) * q)
    k = int(round(dt / q))

    def frame(x):
        i = int(np.argmin(np.abs(t - x)))
        return i if abs(t[i] - x) < 1e-6 and i + k < len(t) else None

    ok = [float(x) for x in wanted if frame(x) is not None]
    if not ok and len(t) > k + 1:
        ok = [float(t[(len(t) - 1 - k) // 2])]
    return dt, ok


@torch.no_grad()
def lowk_gains(model, gts, t0s, dt, device, dtype):
    """Teacher-forced increment gain of the model in each wavenumber shell.

    g = sum Re(conj(dP) dT) / sum |dT|^2 over the modes of a shell and over
    the seeds, with dP = model(psi(t0), dt) - psi(t0) (one step from an
    empty memory, as at the start of a rollout) and dT = psi(t0 + dt) -
    psi(t0), both from the ground truth gts = [(times, frames), ...].
    g = 1 is a model whose one-step increment has the true size in that
    shell.  Returns {t0: {shell: g}}.
    """
    out: dict[float, dict[str, float]] = {}
    for t0 in t0s:
        a, b = [], []
        for t, f in gts:
            i0 = int(np.argmin(np.abs(np.asarray(t) - t0)))
            i1 = int(np.argmin(np.abs(np.asarray(t) - (t0 + dt))))
            a.append(np.asarray(f[i0], dtype=np.float64))
            b.append(np.asarray(f[i1], dtype=np.float64))
        p0 = torch.as_tensor(np.stack(a), dtype=torch.float64)
        x = p0.to(device=device, dtype=dtype).unsqueeze(1)
        step = torch.full((x.shape[0],), float(dt), device=device,
                          dtype=dtype)
        dp = model(x, step)[0][:, 0].cpu().double() - p0
        dtru = torch.as_tensor(np.stack(b), dtype=torch.float64) - p0
        fp, ft = torch.fft.fft2(dp), torch.fft.fft2(dtru)
        num = (fp.conj() * ft).real.sum(dim=0).numpy()
        den = (ft.abs() ** 2).sum(dim=0).numpy()
        masks = lowk_masks(p0.shape[-1])
        out[float(t0)] = {k: float(num[m].sum() / max(den[m].sum(), 1e-300))
                          for k, m in masks.items()}
    return out


def lowk_val_text(gains):
    """'2.41/1.12@1000 2.60/1.15@2000': lowest / first shell gain at t0."""
    return " ".join(f"{g['below']:.2f}/{g['first']:.2f}@{t0:g}"
                    for t0, g in gains.items())


def alive_mask(psi, limit):
    """Per-sample bool (b,): finite everywhere and max |psi| <= limit."""
    flat = psi.detach().flatten(1)
    return torch.isfinite(flat).all(dim=1) & (flat.abs().amax(dim=1) <= limit)


def context_span(t_start, k, cdt):
    """(t0, t1) of the ground-truth file that holds k context frames."""
    return round(float(t_start) - int(k) * float(cdt), 6), float(t_start)


def context_frames(cache_dir, l, off, seeds, t_start, k, cdt, solver_dt,
                   device, require_cuda=None):
    """(k, len(seeds), l, l) float64 ground truth at t_start - k cdt, ...,
    t_start - cdt: the history a --context run sees before t_start.

    Solved from t = 0 like every ground truth, into small files of their
    own (t0 = t_start - k cdt, t1 = t_start, every cdt) the first time.
    """
    t0, t1 = context_span(t_start, k, cdt)
    if t0 < 0:
        raise SystemExit(f"context of {k} frames every {cdt:g} starts at "
                         f"t = {t0:g} < 0")
    ground_truth_batch(cache_dir, l, off, seeds, t0, t1, cdt, solver_dt,
                       device, require_cuda=require_cuda)
    out = []
    for sd in seeds:
        path = find_gt(cache_dir, l, off, int(sd), t0, t1, cdt)
        assert path is not None, (l, off, sd, t0, t1, cdt)
        t, f = load_gt(path, t1)
        assert abs(float(t[int(k)]) - t1) < 1e-6, (path, t[:int(k) + 1])
        out.append(np.asarray(f[:int(k)], dtype=np.float64))
    return np.stack(out, axis=1)


def warm_memory(model, frames, dt, state=None):
    """Teacher forcing over a context: feed the ground-truth frames, each a
    (b, 1, l, l) tensor dt after the previous one, and keep only the
    memory the model builds from them."""
    for x in frames:
        d = torch.full((x.shape[0],), float(dt), device=x.device,
                       dtype=x.dtype)
        _, state = model(x, d, state)
    return state


@torch.no_grad()
def eval_rollout(model, gt_paths, cfg, device, dtype, context=None):
    """Full 128-box rollout on the held-out seeds.

    Returns the mean R^2 over every seed and time (reference only), the mean
    interpolated survival time (the selection score), the per-seed survival
    times, and the in-range accuracy: the mean over seeds and rollout times
    t <= min(2 END, val_t1) of R^2 clipped to [0, 1].

    The memory starts empty at val_t0 (or is warmed on the context frames,
    a (k, b, l, l) array, with cfg["context"] > 0) and is carried along
    the rollout.
    A seed diverges when its field goes non-finite or max |psi| exceeds
    cfg["diverge_abs"].  It is frozen at its last good state and memory so
    it cannot feed NaN to the model, gets R^2 = -1 from that step on and
    fails at its own step; the other seeds keep rolling and are scored as
    usual (the model acts on every sample separately).
    """
    model.eval()
    limit = float(cfg.get("diverge_abs", CONFIG["diverge_abs"]))
    truth_list, times = [], None
    for path in gt_paths:
        t, f = load_gt(path, cfg["val_t1"])
        if times is None:
            times = make_schedule(cfg["val_t0"], cfg["val_t1"],
                                  cfg["val_alpha"], cfg["dt_min"],
                                  cfg["val_dt_max"],
                                  quantum=float(t[1] - t[0]))
        idx = [int(np.argmin(np.abs(t - x))) for x in times]
        assert all(abs(t[i] - x) < 1e-6 for i, x in zip(idx, times)), path
        truth_list.append(np.asarray(f[idx], dtype=np.float64))

    assert times is not None
    truth = np.stack(truth_list, axis=1)
    psi = torch.as_tensor(truth[0], dtype=dtype, device=device).unsqueeze(1)
    b = psi.shape[0]
    r2 = np.ones((len(times), b))
    alive = torch.ones(b, dtype=torch.bool, device=device)
    div_k = np.full(b, len(times))
    state = None
    if context is not None and len(context):
        state = warm_memory(model, [torch.as_tensor(c, dtype=dtype,
                                                    device=device)[:, None]
                                    for c in context], cfg["context_dt"])
    for i in range(1, len(times)):
        dt = torch.full((b,), float(times[i] - times[i - 1]), device=device,
                        dtype=dtype)
        new, new_state = model(psi, dt, state)
        newly = alive & ~alive_mask(new, limit)
        if bool(newly.any()):
            div_k[newly.cpu().numpy()] = i
            alive = alive & ~newly
            if not bool(alive.any()):
                break
        psi = torch.where(alive.view(-1, 1, 1, 1), new, psi)
        state = mask_state(alive, new_state, state)
        tr = torch.as_tensor(truth[i], dtype=dtype, device=device).unsqueeze(1)
        num = _sq(psi - tr)
        den = _sq(tr - tr.mean(dim=(1, 2, 3), keepdim=True))
        r2[i] = (1 - num / den).double().cpu().numpy()

    failing = r2.copy()
    for k in range(b):
        r2[div_k[k]:, k] = -1.0
        failing[div_k[k]:, k] = np.nan
    survive = [interp_survival(times, failing[:, k], cfg["val_threshold"])
               for k in range(b)]
    in_range = np.asarray(times) <= min(2.0 * cfg["train_end"], cfg["val_t1"])
    acc = float(np.clip(r2[in_range], 0.0, 1.0).mean())
    model.train()
    return float(r2.mean()), float(np.mean(survive)), survive, acc


def tuple_of(x):
    if isinstance(x, (list, tuple)):
        return tuple(tuple_of(v) for v in x)
    return x


def unroll_for(epoch, cfg):
    """Unroll length and stage index for this epoch."""
    total = cfg["epochs"]
    stage, unroll = 0, 1
    for i, (frac, n) in enumerate(cfg["unroll_stages"]):
        if epoch >= 1 + int(frac * total):
            stage, unroll = i, n
    return unroll, stage


@torch.no_grad()
def tile_check(net, device, dtype, n=64, steps=3, dt=10.0, seed=0):
    """Relative error of the 2 x 2 tiling test over a few carried steps.

    A tiled state is a legitimate state of the twice larger box, and the
    exact solver commutes with tiling, so a box-size agnostic model must
    give tile(model(psi, h)) = model(tile(psi), tile(h)) for the state and
    every hidden state.  Returns max |a - b| / max |a| over the states and
    the increments.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, 1, n, n, generator=g, device=device, dtype=dtype) * 0.6
    q = q - q.mean()
    d1 = torch.tensor([float(dt)], device=device, dtype=dtype)
    small, big = q, q.repeat(1, 1, 2, 2)
    hs = hb = None
    worst = 0.0
    for _ in range(steps):
        ns, hs = net(small, d1, hs)
        nb, hb = net(big, d1, hb)
        inc = (ns - small).repeat(1, 1, 2, 2)
        worst = max(worst, float((inc - (nb - big)).abs().max()
                                 / inc.abs().max().clamp_min(1e-30)))
        for a, b in zip(hs, hb):
            ta = a.repeat(1, 1, 2, 2)
            worst = max(worst, float((ta - b).abs().max()
                                     / ta.abs().max().clamp_min(1e-30)))
        small, big = ns, nb
    return worst


def _bench_step(net, x, d, steps):
    """One fwd + bwd through `steps` carried RNO steps."""
    net.zero_grad(set_to_none=True)
    psi, h, loss = x, None, 0.0
    for _ in range(steps):
        psi, h = net(psi, d, h)
        loss = loss + psi.square().mean()
    loss.backward()


def benchmark(cfg, device, dtype):
    """Time this GPU across the precision modes and check tile invariance."""
    log("=" * 78)
    log(" BENCHMARK")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        log(f" {p.name}  {p.total_memory / 2**30:.0f} GiB  "
            f"{p.multi_processor_count} SMs")
    log("=" * 78)
    bs = int(cfg["batch"])
    steps = int(cfg["grad_steps"])
    log(f" width={cfg['width']} modes={cfg['modes']} layers={cfg['layers']} "
        f"batch={max(1, bs // steps)} x {steps} carried steps with gradient "
        f"(as in every stage with unroll >= grad_steps)")
    log(f" {'mode':>6} {'fwd+bwd':>12} {'samples/s':>11} {'peak mem':>10} "
        f"{'tile-inv':>11}")
    for mode in ("fp32", "tf32", "fp64"):
        if device.type != "cuda" and mode == "tf32":
            continue
        _, dt_ = set_precision(mode)
        torch.manual_seed(0)
        net = CHRNO(**model_arch(cfg)).to(device, dtype=dt_)
        nn.init.normal_(net.core.projection.fcs[-1].weight, std=0.05)
        b = max(1, bs // steps)
        x = (torch.randn(b, 1, 64, 64, device=device, dtype=dt_) * 0.5)
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        d = torch.full((b,), 10.0, device=device, dtype=dt_)
        for _ in range(3):
            _bench_step(net, x, d, steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(10):
            _bench_step(net, x, d, steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per = (time.time() - t0) / 10
        peak = (torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda" else 0.0)
        net.eval()
        tile = tile_check(net, device, dt_)
        log(f" {mode:>6} {per*1e3:10.1f} ms {b/per:11.0f} {peak:9.2f} GiB "
            f"{tile:11.3e}")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log("")
    log(" If fwd+bwd is well under ~20 ms the run is launch-bound: raise batch")
    log(" or widen the model before changing anything else.")


def train(cfg, device, dtype, prec, rank, world, local, out_dir, smoke):
    seed0 = int(cfg["seed"]) * 1000
    torch.manual_seed(seed0 + rank)
    np.random.seed(seed0 + rank)
    torch.backends.cudnn.benchmark = True
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(max(1, int(os.environ["SLURM_CPUS_PER_TASK"])))

    gpu_gb = 0.0
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        gpu_gb = p.total_memory / 2**30
        log(f"[setup] {p.name}, {p.total_memory / 1e9:.1f} GB, "
            f"torch {torch.__version__}", rank=rank)
    if prec == "fp64":
        log("[setup] note: the training memmaps are float32, so fp64 raises "
            "cost "
            "(~5x fewer FLOP/s on an A100, 2x memory) without adding "
            "target accuracy; tf32 or fp32 train the same model faster",
            rank=rank)
        if 0.0 < gpu_gb < 12.0:
            log(f"[setup] WARNING: fp64 on a {gpu_gb:.1f} GiB device is "
                f"likely to run out of memory (fp64 peaked at 7.6 GB on this "
                f"config). Set PRECISION='fp32' or lower batch/width, or "
                f"request a full A100.", rank=rank)
    log(f"[setup] device={device} precision={prec} dtype={dtype} "
        f"world_size={world}", rank=rank)
    log(f"[setup] model dir -> {out_dir}", rank=rank)

    paths = build_cache(cfg, rank)
    shapes = {k: tuple(np.load(p, mmap_mode="r").shape)
              for k, p in paths.items()}
    t_data = {k: data_horizon(v[1]) for k, v in shapes.items()}
    end = float(round(cfg["train_end"]))
    if set(t_data.values()) != {end}:
        raise SystemExit(
            f"{cfg['npz']} holds data to t = {t_data} ({shapes}), not the "
            f"requested END = {end:g}: pass --end "
            f"{t_data['train']:g} or regenerate with "
            f"`python ch_gpu.py --end {end:g}`")
    log(f"[data] {cfg['npz']} -> training data ends at t = {end:g}  "
        f"train_X {shapes['train']}", rank=rank)
    train_s = PairSampler(paths["train"], cfg, device, dtype,
                          seed0 + 100 + rank)
    val_s = PairSampler(paths["val"], cfg, device, dtype,
                        seed0 + 200 + rank)
    gb = train_s.data.numel() * 4 / 1e9
    log(f"[data] train {tuple(train_s.data.shape)}  "
        f"val {tuple(val_s.data.shape)}  float32, {gb:.2f} GB", rank=rank)
    sel = cfg["train_off"]
    log(f"[data] train_off={sel if sel is not None else 'all'} -> "
        f"trajectories {[int(i) for i in train_s.kept]} "
        f"(off={sorted(set(train_s.offs.tolist()))})", rank=rank)
    log(f"[data] sign-flip augmentation "
        f"{'on' if train_s.sign_ok else 'off'}, sampling from "
        f"{train_s.data.device}", rank=rank)

    gt_args = (cfg["val_l"], 0.0)
    gt_span = (cfg["val_t0"], cfg["val_t1"], cfg["val_save_every"])
    if rank == 0:
        ground_truth_batch(cfg["gt_dir"], *gt_args, cfg["val_seeds"], *gt_span,
                           cfg["solver_dt"], device,
                           require_cuda=cfg["gt_require_cuda"])
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    gt_paths = [find_gt(cfg["gt_dir"], *gt_args, int(s), *gt_span)
                for s in cfg["val_seeds"]]
    assert all(gt_paths), gt_paths
    log(f"[val] l={cfg['val_l']} ground truth to t={cfg['val_t1']:g} for "
        f"seeds {list(cfg['val_seeds'])} in {cfg['gt_dir']}", rank=rank)
    val_gts = [load_gt(p, cfg["val_t1"]) for p in gt_paths]
    val_ctx = None
    if cfg["context"] and rank == 0:
        val_ctx = context_frames(cfg["gt_dir"], *gt_args, cfg["val_seeds"],
                                 cfg["val_t0"], cfg["context"],
                                 cfg["context_dt"], cfg["solver_dt"], device,
                                 require_cuda=cfg["gt_require_cuda"])
        log(f"[val] context: {cfg['context']} ground-truth frames every "
            f"{cfg['context_dt']:g} before t = {cfg['val_t0']:g} warm the "
            f"memory (an ablation: the default sees psi(t_start) only)",
            rank=rank)
    lk_dt, lk_t0s = lowk_plan(val_gts[0][0], LOWK_VAL_T0S)
    log(f"[val] lowk g = increment gain of the 8 lowest modes / the first "
        f"trained shell at t0 = {lk_t0s}, dt = {lk_dt:g} (logged only, not "
        f"used for selection)", rank=rank)

    arch: dict[str, Any] = model_arch(cfg, train_s.l)
    raw = CHRNO(**arch).to(device, dtype=dtype)
    n_par = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    log(f"[model] RNO {n_par:,} parameters: width {arch['width']}, modes "
        f"{arch['modes']} (k <= {2 * math.pi * (arch['modes'] // 2) / 64:.3f}"
        f" at dx = 1), {arch['layers']} layers, dtype={dtype}", rank=rank)
    log(f"[model] conserve_mass={cfg['conserve_mass']} "
        f"laplacian_output={cfg['laplacian_output']} dt_conditioned=True "
        f"residual_prediction=True rno_skip={cfg['rno_skip']}", rank=rank)
    log(f"[data] chains: {1 - cfg['chain_sched_frac']:.0%} constant dt, "
        f"{cfg['chain_sched_frac']:.0%} dt ~ alpha t with alpha in "
        f"{tuple(cfg['sched_alphas'])}; memory starts empty at every chain",
        rank=rank)

    ceil = [(n, train_s.max_dt(n)) for _, n in cfg["unroll_stages"]]
    log("[data] max trainable dt by unroll: "
        + ", ".join(f"{n} -> {d:.1f}" for n, d in ceil), rank=rank)
    short = [(n, d) for n, d in ceil if d < cfg["dt_max"] - 1e-9]
    if short:
        log(f"[warn] chain length caps dt below cfg dt_max={cfg['dt_max']:g} "
            f"for unroll {[n for n, _ in short]}: a t <= "
            f"{(train_s.n_frames - 1) * train_s.fdt:.0f} trajectory has no "
            f"room for longer chains at full stride. Those stages train a "
            f"shorter-dt regime than the rollout uses.", rank=rank)

    sched: dict[str, Any] = dict(
        epochs=cfg["epochs"], steps_per_epoch=cfg["steps_per_epoch"],
        lr=cfg["lr"], unroll_stages=tuple_of(cfg["unroll_stages"]),
        train_shape=shapes["train"], val_shape=shapes["val"],
        npz=os.path.basename(cfg["npz"]), val_t1=float(cfg["val_t1"]),
        select=cfg["select"], select_min_delta=float(cfg["select_min_delta"]),
        seed=int(cfg["seed"]), grad_steps=int(cfg["grad_steps"]),
        chain_sched_frac=float(cfg["chain_sched_frac"]),
        sched_alphas=tuple_of(cfg["sched_alphas"]),
        context=int(cfg["context"]), context_dt=float(cfg["context_dt"]))
    horizon = dict(train_t_end=t_data["train"], train_shape=shapes["train"],
                   val_shape=shapes["val"])
    state: dict[str, Any] = {}
    state_path = os.path.join(out_dir, f"{cfg['run_name']}_state.pt")
    if cfg["resume"] and os.path.exists(state_path):
        cand = torch.load(state_path, map_location=device, weights_only=False)
        saved = cand.get("arch")
        reason = None
        if saved is not None:
            diff = {k: (saved[k], arch[k]) for k in saved
                    if k in arch and saved[k] != arch[k]}
            if diff:
                reason = f"architecture changed: {diff}"
        prior = cand.get("sched")
        if reason is None and prior is not None:
            sdiff = {k: (prior.get(k), sched.get(k))
                     for k in sorted(set(prior) | set(sched))
                     if tuple_of(prior.get(k)) != tuple_of(sched.get(k))}
            if sdiff:
                reason = f"training schedule changed: {sdiff}"
        if reason is None:
            try:
                raw.load_state_dict(cand["raw"])
            except RuntimeError as exc:
                reason = f"state_dict does not fit: {str(exc).splitlines()[0]}"
        if reason is None:
            state = cand
            log(f"[resume] {os.path.basename(state_path)} at epoch "
                f"{state['epoch']}", rank=rank)
        elif rank == 0:
            stale = free_name(state_path.replace(".pt", "_stale.pt"))
            os.replace(state_path, stale)
            log(f"[resume] ignoring {os.path.basename(state_path)}: {reason}",
                rank=rank)
            log(f"[resume] kept as {os.path.basename(stale)}; starting fresh",
                rank=rank)
        del cand
    elif cfg["resume"]:
        log(f"[resume] no {os.path.basename(state_path)} in {out_dir}; "
            f"training from epoch 1", rank=rank)

    best_path = os.path.join(out_dir, f"{cfg['run_name']}_best.pt")
    if not state and rank == 0 and os.path.exists(best_path):
        prev = free_name(best_path.replace(".pt", "_prev.pt"))
        os.replace(best_path, prev)
        log(f"[resume] fresh start: kept the existing checkpoint as "
            f"{os.path.basename(prev)} so this run cannot overwrite it with "
            f"an early-epoch model", rank=rank)

    model = raw
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            raw, device_ids=[local] if device.type == "cuda" else None)

    ema = EMA(raw, cfg["ema_decay"])
    if state.get("ema"):
        ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
    gen = torch.Generator(device=device).manual_seed(seed0 + rank)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    if state.get("opt"):
        opt.load_state_dict(state["opt"])

    start_epoch = int(state.get("epoch", 0)) + 1
    best_acc = float(state.get("best", -1.0))
    best_epoch = int(state.get("best_epoch", 0))
    best_survive = float(state.get("best_survive", 0.0))
    stale_vals = int(state.get("stale_vals", 0))
    prev_stage = int(state.get("stage", -1))
    stop_on = int(cfg["patience"]) > 0
    in_t1 = min(2.0 * cfg["train_end"], cfg["val_t1"])
    if not stop_on:
        log("[setup] early stopping disabled (patience=0); the no-gain counter "
            "is still logged, so the epoch any patience would have fired at "
            "can be read straight off the log", rank=rank)

    if start_epoch > cfg["epochs"]:
        log(f"[resume] state is already at epoch {start_epoch - 1} of "
            f"{cfg['epochs']}: nothing left to train. Delete "
            f"{os.path.basename(state_path)} or raise epochs to train more.",
            rank=rank)

    batch = max(2, cfg["batch"] // world)
    t_start = time.time()
    n_bad = 0
    stopped = ""

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        unroll, stage = unroll_for(epoch, cfg)
        if stage != prev_stage:
            if prev_stage >= 0:
                log(f"[stage] unroll -> {unroll}; max trainable dt "
                    f"{train_s.max_dt(unroll):.1f}", rank=rank)
            prev_stage = stage
        bs = max(2, batch // max(1, min(unroll, cfg["grad_steps"])))
        frac = (epoch - 1) / max(1, cfg["epochs"] - 1)
        cos = 0.5 * (1 + math.cos(math.pi * frac))
        lr = (cfg["lr_min"] + (cfg["lr"] - cfg["lr_min"]) * cos) \
            * cfg["stage_lr_mult"] ** stage
        for g in opt.param_groups:
            g["lr"] = lr

        t_epoch = time.time()
        run_loss, run_dpsi, n_seen = 0.0, 0.0, 0
        for _ in range(cfg["steps_per_epoch"]):
            chain, dt_t, warm = train_s.sample(bs, unroll)
            both = train_s.augment(chain + warm, gen)
            chain, warm = both[:len(chain)], both[len(chain):]
            sigma = cfg["noise"] * float(torch.rand((), generator=gen,
                                                    device=device))
            psi = add_noise(chain[0], sigma)
            state = None

            n_nograd = max(0, unroll - cfg["grad_steps"])
            if warm:
                with torch.set_grad_enabled(n_nograd == 0):
                    state = warm_memory(model if n_nograd == 0 else raw,
                                        [add_noise(w, sigma) for w in warm],
                                        cfg["context_dt"])
            if n_nograd:
                with torch.no_grad():
                    for k in range(n_nograd):
                        psi, state = raw(psi, dt_t[:, k], state)
                psi, state = psi.detach(), detach_state(state)

            terms, dpsi = [], 0.0
            for k in range(n_nograd, unroll):
                psi_in = psi
                pred, state = model(psi, dt_t[:, k], state)
                true = chain[k + 1]
                terms.append(r2_loss(pred, true)
                             + cfg["w_inc"] * increment_loss(pred, true,
                                                             psi.detach())
                             + cfg["w_grad"] * gradient_loss(pred, true))
                if k == n_nograd:
                    with torch.no_grad():
                        ref = psi_in - psi_in.mean(dim=(1, 2, 3), keepdim=True)
                        dpsi = float((_sq(pred - psi_in)
                                      / _sq(ref).clamp_min(1e-12)).sqrt().mean())
                psi = pred
            loss = torch.stack(terms).mean()

            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_bad += 1
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                cfg["grad_clip"])
            if not torch.isfinite(gn):
                n_bad += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            ema.update(raw)

            run_loss += float(loss.detach())
            run_dpsi += dpsi
            n_seen += 1

        dt_epoch = time.time() - t_epoch
        ms = dt_epoch / max(1, cfg["steps_per_epoch"]) * 1e3
        line = (f"epoch {epoch:3d}/{cfg['epochs']} | unroll {unroll:2d} | "
                f"train {run_loss / max(1, n_seen):.5f} | ")

        do_val = (epoch % cfg["val_every"] == 0) or epoch == cfg["epochs"]
        if rank == 0:
            evm = CHRNO(**arch).to(device, dtype=dtype)
            evm.load_state_dict(raw.state_dict())
            ema.copy_to(evm)
            val1 = eval_one_step(evm, val_s)
            line += (f"val1 {val1:.5f} | d/psi {run_dpsi / max(1, n_seen):.4f} "
                     f"| lr {lr:.2e} | {dt_epoch:.1f}s ({ms:.0f} ms/step)")
            if do_val:
                score, surv, per_seed, acc = eval_rollout(
                    evm, gt_paths, cfg, device, dtype, context=val_ctx)
                line += (f" | rollout {score:.4f} | in-range R2 {acc:.4f} "
                         f"(t <= {in_t1:g}) | survives t={surv:7.1f} "
                         f"{[int(round(s)) for s in per_seed]}")
                lk = lowk_gains(evm, val_gts, lk_t0s, lk_dt, device, dtype)
                line += f" | lowk g {lowk_val_text(lk)}"
                if better_epoch(surv, acc, best_survive, best_acc,
                                cfg["select_min_delta"]):
                    best_acc, best_epoch, best_survive = acc, epoch, surv
                    stale_vals = 0
                    torch.save(dict(model=evm.state_dict(), arch=arch,
                                    precision=prec, **horizon,
                                    metrics=dict(select="survival",
                                                 survives=surv,
                                                 per_seed=per_seed,
                                                 r2_in_range=acc,
                                                 in_range_t1=in_t1,
                                                 rollout=score, epoch=epoch,
                                                 val1=val1),
                                    config=cfg),
                               best_path)
                    line += "  <- best"
                else:
                    stale_vals += 1
                    cap = cfg["patience"] if stop_on else "off"
                    line += f"  (no gain {stale_vals}/{cap})"
                torch.save(dict(epoch=epoch, best=best_acc,
                                best_epoch=best_epoch,
                                best_survive=best_survive,
                                stale_vals=stale_vals, stage=stage,
                                arch=arch, sched=sched, **horizon,
                                raw=raw.state_dict(), ema=ema.shadow,
                                opt=opt.state_dict()), state_path)
            del evm
            log(line)
        if world > 1:
            dist.barrier()

        if do_val and stop_on and stale_vals >= cfg["patience"]:
            stopped = " [stopped early: no further improvement]"
            log(f"[stop] survival has not improved by "
                f"{cfg['select_min_delta']:g} for {cfg['patience']} "
                f"validations ({cfg['patience'] * cfg['val_every']} epochs); "
                f"best was t = {best_survive:.1f} at epoch {best_epoch}",
                rank=rank)
            break

    total = time.time() - t_start
    if rank == 0:
        if device.type == "cuda":
            log(f"[mem] peak allocated "
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        if n_bad:
            log(f"[warn] skipped {n_bad} non-finite steps")
        log(f"[done] selected epoch {best_epoch} (survives to t = "
            f"{best_survive:.1f}, in-range R2 {best_acc:.4f}) | "
            f"total {total / 3600:.2f} h ({total:.0f}s) | "
            f"{n_par:,} parameters | saved to "
            f"{best_path}{stopped}")
        with open(os.path.join(out_dir,
                               f"{cfg['run_name']}_train_config.json"),
                  "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.items()}, f, indent=2)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Train the RNO on a ch_gpu.py dataset.")
    ap.add_argument("--end", type=float, default=None,
                    help="training horizon END (default CONFIG['train_end'])")
    ap.add_argument("--npz", default="",
                    help="dataset path (default Work/data/test_t<END>.npz)")
    ap.add_argument("--variant", choices=VARIANTS, default=None,
                    help="lap (default, CONFIG['laplacian_output']): the "
                         "head predicts M and the increment is its 5-point "
                         "Laplacian, model in Work/models/rno_t<END>_lap/; "
                         "base: the direct-increment head, rno_t<END>/")
    ap.add_argument("--modes", type=int, default=None,
                    help="Fourier modes n_modes = (M, M) of every spectral "
                         "convolution (default CONFIG['modes'] = 16, i.e. "
                         "|k| <= 0.785 on the 64-box). A non-default value "
                         "is tagged m<M> in the model directory and every "
                         "label")
    ap.add_argument("--width", type=int, default=None,
                    help="hidden channels (default CONFIG['width'] = 32); "
                         "a non-default value is tagged w<W>")
    ap.add_argument("--layers", type=int, default=None,
                    help="number of RNO layers (default CONFIG['layers'] = "
                         "4); a non-default value is tagged d<L>")
    ap.add_argument("--context", type=int, default=None,
                    help="ablation: K ground-truth frames context_dt apart "
                         "before each chain (and before t_start in every "
                         "validation rollout) warm the memory (default "
                         "CONFIG['context'] = 0: psi(t_start) only); tagged "
                         "c<K>")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (default CONFIG['seed'] = 0): weight "
                         "init, training and validation pair draws and "
                         "augmentation noise; a non-zero value is tagged "
                         "s<N>, so w24 seed 1 trains into rno_t<END>_lap_"
                         "w24s1/ next to the seed-0 run")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run; writes only under Work/smoke/")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--local-rank", "--local_rank", type=int,
                    help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def resolve_config(a):
    """CONFIG (+SMOKE) with END, dataset, output dirs and val_t1 filled in."""
    cfg: dict[str, Any] = dict(CONFIG)
    base_width, base_modes = cfg["width"], cfg["modes"]
    base_layers = cfg["layers"]
    if a.smoke:
        cfg.update(SMOKE)
    if a.end is not None:
        cfg["train_end"] = a.end
    end = int(round(cfg["train_end"]))
    cfg["train_end"] = end
    root = SMOKE_ROOT if a.smoke else WORK
    variant = getattr(a, "variant", None)
    if variant:
        cfg["laplacian_output"] = variant == "lap"
    if getattr(a, "modes", None):
        cfg["modes"] = base_modes = int(a.modes)
    if getattr(a, "width", None):
        cfg["width"] = base_width = int(a.width)
    if getattr(a, "layers", None):
        cfg["layers"] = base_layers = int(a.layers)
    if getattr(a, "context", None) is not None:
        cfg["context"] = int(a.context)
    if getattr(a, "seed", None) is not None:
        cfg["seed"] = int(a.seed)
    cfg["laplacian_output"] = bool(cfg["laplacian_output"])
    cfg["variant"] = variant_name(cfg["laplacian_output"])
    cfg["size_tag"] = run_tag(base_width, base_modes, cfg["seed"],
                              base_layers, cfg["context"])
    cfg["npz"] = a.npz or cfg["npz"] or data_path(end)
    cfg["memmap_dir"] = memmap_dir(end, root)
    cfg["model_dir"] = model_dir(end, root, cfg["variant"], cfg["size_tag"])
    cfg["gt_dir"] = gt_dir(root)
    if not float(cfg["val_t1"]):
        cfg["val_t1"] = default_t_end(end)
    cfg["val_t1"] = float(cfg["val_t1"])
    return cfg


def main():
    a = parse_args()
    if a.npz:
        a.npz = os.path.abspath(a.npz)
    to_project_dir()
    if a.npz and not os.path.relpath(a.npz).startswith(".."):
        a.npz = os.path.relpath(a.npz)
    cfg = resolve_config(a)
    smoke = a.smoke

    rank, world, local, device = init_distributed()
    out_dir = setup_log(cfg["model_dir"], f"{cfg['run_name']}_train", rank)
    prec, dtype = set_precision()

    if a.bench:
        benchmark(cfg, device, dtype)
        return

    log("=" * 78, rank=rank)
    log(f" RNO / Cahn-Hilliard -- TRAINING{'  [SMOKE]' if smoke else ''}   "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}", rank=rank)
    log("=" * 78, rank=rank)
    log(f"[setup] END = {cfg['train_end']}  variant {cfg['variant']}  "
        f"width {cfg['width']}  modes {cfg['modes']}  "
        f"layers {cfg['layers']}  context {cfg['context']}  "
        f"seed {cfg['seed']}  data {cfg['npz']}  "
        f"validation to t = {cfg['val_t1']:g}", rank=rank)
    train(cfg, device, dtype, prec, rank, world, local, out_dir, smoke)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
