"""GPU Cahn-Hilliard dataset generator (ch.py's data, frame for frame).

    python ch_gpu.py                  END = 300 -> Work/data/test_t0300.npz
    python ch_gpu.py --end 1000       -> Work/data/test_t1000.npz
    python ch_gpu.py --end 2000 --out PATH
    python ch_gpu.py --verify 3 | --benchmark | --cpu

The .npz holds train_X, train_y, val_X, val_y (float64) as ch.py writes them;
a sidecar .json with the same stem records the device, shapes and timing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

L = 64
DT = 0.01
START = 0
END = 300
N_AHEAD = 100
GAP = 10
NENS = 25
OFF_VALUES = [-0.4, -0.2, 0.0, 0.2, 0.4]
FNAME = "test.npz"

BATCH = 25
DX = 1


def set_ic(l, seed, off, dtype=np.float64):
    """ch.py's PhaseOrdering.set_ic, verbatim, so the RNG stream is identical."""
    np.random.seed(seed)
    ic = np.random.uniform(-0.1, 0.1, size=(l, l)).astype(dtype)
    return ic - ic.mean() + off


def plan(nens=NENS, off_values=OFF_VALUES):
    """ch.py's seed assignment: which (seed, off) pairs, in which order."""
    np.random.seed(42)
    train_ens = int(0.8 * nens)
    train_seeds = np.random.randint(0, 10000, train_ens)
    val_seeds = np.random.randint(10000, 20000, nens - train_ens)

    def pairs(seeds):
        out = []
        for off, group in zip(off_values,
                              np.array_split(seeds, len(off_values))):
            out.extend((int(s), float(off)) for s in group)
        return out

    return pairs(train_seeds), pairs(val_seeds)


def _del_sq(psi):
    """The 5-point periodic Laplacian, on the last two axes so it batches."""
    return (torch.roll(psi, 1, dims=-2) + torch.roll(psi, -1, dims=-2)
            + torch.roll(psi, 1, dims=-1) + torch.roll(psi, -1, dims=-1)
            - 4 * psi) / DX ** 2


def integrate(psi, n_steps, gap, progress=None, out=None):
    """Explicit-Euler Cahn-Hilliard, keeping every ``gap``-th frame.

    ch.py allocates all 30,000 frames (983 MB per trajectory) and then throws
    away nine tenths of them.  prepare_features_labels only ever reads indices
    that are multiples of ``gap``, so only those are kept here: 3,000 frames,
    98 MB.  Frame j of the result is ch.py's data[j * gap].  With out, the
    frames go into that preallocated (batch, n_keep, l, l) array.
    """
    n_keep = (n_steps + gap - 1) // gap
    kept = out
    if kept is None:
        kept = np.empty((psi.shape[0], n_keep) + tuple(psi.shape[1:]),
                        dtype=np.float64)
    j = 0
    for i in range(n_steps):
        psi += _del_sq(psi ** 3 - psi - _del_sq(psi)) * DT
        if i % gap == 0:
            kept[:, j] = psi.cpu().numpy()
            j += 1
        if progress and (i + 1) % progress == 0:
            print(f"    step {i + 1:,}/{n_steps:,}", flush=True)
    return kept


def run_batches(specs, l, n_steps, gap, device, batch=BATCH, tag=""):
    """Roll every (seed, off) in ``specs``, in order, ``batch`` at a time.

    The result is allocated once and each batch writes its slice, so the
    host never holds a second copy of the trajectories.
    """
    n_keep = (n_steps + gap - 1) // gap
    out = np.empty((len(specs), n_keep, l, l), dtype=np.float64)
    for s in range(0, len(specs), batch):
        chunk = specs[s:s + batch]
        ic = np.stack([set_ic(l, seed, off) for seed, off in chunk])
        psi = torch.from_numpy(ic).to(device=device, dtype=torch.float64)
        t0 = time.time()
        integrate(psi, n_steps, gap, out=out[s:s + len(chunk)])
        print(f"  [{tag}] {s + len(chunk)}/{len(specs)} trajectories "
              f"({time.time() - t0:.1f}s)", flush=True)
        del psi
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def features_labels(kept, n_ahead, gap):
    """ch.py's prepare_features_labels, expressed on the kept frames.

    ch.py takes X = data[:-n_ahead][::gap] and y = data[n_ahead:][::gap].  With
    start = 0 and gap dividing n_ahead those are exactly kept[:-shift] and
    kept[shift:] for shift = n_ahead // gap -- the same frames, not a
    re-derivation of them.
    """
    shift = n_ahead // gap
    return kept[:, :-shift], kept[:, shift:]


def device_label(device):
    """'cuda:0 NVIDIA A100-SXM4-40GB', or 'cpu'."""
    if device.type != "cuda":
        return device.type
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{idx} {torch.cuda.get_device_name(idx)}"


def sidecar_path(fname):
    return os.path.splitext(fname)[0] + ".json"


def generate(fname=FNAME, device=None, end=END):
    if START != 0 or N_AHEAD % GAP != 0:
        raise ValueError(
            f"the frame-skipping path assumes START == 0 (got {START}) and "
            f"GAP dividing N_AHEAD (got {N_AHEAD} % {GAP}). Use ch.py, or keep "
            f"every frame, if that changes.")

    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    n_steps = int(end / DT)
    train_specs, val_specs = plan()
    print(f"[gen] device={device_label(device)}, l={L}, END={end:g}, "
          f"{n_steps:,} steps, dt={DT}", flush=True)
    print(f"[gen] {len(train_specs)} train + {len(val_specs)} val "
          f"trajectories over off={OFF_VALUES}", flush=True)

    t0 = time.time()
    train = run_batches(train_specs, L, n_steps, GAP, device, tag="train")
    val = run_batches(val_specs, L, n_steps, GAP, device, tag="val")
    t_gen = time.time() - t0
    train_X, train_y = features_labels(train, N_AHEAD, GAP)
    val_X, val_y = features_labels(val, N_AHEAD, GAP)
    members = dict(train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y)

    total = sum(v.nbytes for v in members.values())
    print(f"[gen] train_X {train_X.shape}  val_X {val_X.shape}  "
          f"{train_X.dtype}", flush=True)
    print(f"[gen] writing {fname} ({total / 1e9:.2f} GB)", flush=True)
    folder = os.path.dirname(os.path.abspath(fname))
    os.makedirs(folder, exist_ok=True)
    tmp = fname[:-4] + ".partial.npz"
    np.savez(tmp, **members)
    os.replace(tmp, fname)
    meta = dict(
        file=os.path.basename(fname), device=device_label(device),
        dtype=str(train_X.dtype), torch_version=torch.__version__,
        END=end, DT=DT, L=L, GAP=GAP, N_AHEAD=N_AHEAD, NENS=NENS,
        OFF_VALUES=list(OFF_VALUES), START=START,
        shapes={k: list(v.shape) for k, v in members.items()},
        bytes=os.path.getsize(fname),
        created=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        generation_seconds=round(t_gen, 2),
        total_seconds=round(time.time() - t0, 2))
    side = sidecar_path(fname)
    with open(side + ".partial", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(side + ".partial", side)
    print(f"[gen] metadata -> {side}", flush=True)
    print(f"[gen] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _del_sq_np(psi):
    return (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0)
            + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1)
            - 4 * psi) / DX ** 2


def _chc_np(psi, n_steps, gap):
    """ch.py's chc, keeping the same frames integrate() keeps."""
    n_keep = (n_steps + gap - 1) // gap
    kept = np.empty((n_keep,) + psi.shape, dtype=np.float64)
    psi, j = psi.copy(), 0
    for i in range(n_steps):
        phi = psi ** 3 - psi - _del_sq_np(psi)
        psi += _del_sq_np(phi) * DT
        if i % gap == 0:
            kept[j] = psi
            j += 1
    return kept


def verify(n_traj, device=None, end=END):
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    n_steps = int(end / DT)
    specs = plan()[0][:n_traj]
    print(f"[verify] {n_traj} trajectories, {n_steps:,} steps, "
          f"ch.py numpy vs {device}\n", flush=True)
    gpu = run_batches(specs, L, n_steps, GAP, device, tag="gpu")
    print()
    worst, ok = 0.0, True
    for i, (seed, off) in enumerate(specs):
        t0 = time.time()
        ref = _chc_np(set_ic(L, seed, off), n_steps, GAP)
        final_ref, final_gpu = ref[-1], gpu[i, -1]
        d = float(np.abs(ref - gpu[i]).max())
        flips = int(np.sum(np.sign(final_ref) != np.sign(final_gpu)))
        worst = max(worst, d)
        ok = ok and flips == 0
        print(f"  seed {seed:>5} off={off:+.1f}: max|diff| = {d:.3e}, "
              f"cells in the wrong phase at t={end:g} = {flips}, "
              f"mass drift = {abs(final_gpu.mean() - final_ref.mean()):.2e}  "
              f"({time.time() - t0:.0f}s for the numpy reference)", flush=True)
    print(f"\n[verify] worst difference {worst:.3e} over {n_traj} "
          f"trajectories; phase agreement "
          f"{'EXACT' if ok else 'BROKEN'}", flush=True)
    return ok


def benchmark(device=None, end=END):
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    n_steps, probe = int(end / DT), 2000
    seed, off = plan()[0][0]
    t0 = time.time()
    _chc_np(set_ic(L, seed, off), probe, GAP)
    t_np = (time.time() - t0) * n_steps / probe * NENS

    ic = np.stack([set_ic(L, s, o) for s, o in plan()[0][:BATCH]])
    psi = torch.from_numpy(ic).to(device=device, dtype=torch.float64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    integrate(psi, probe, GAP)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu = (time.time() - t0) * n_steps / probe

    print(f"projected for all {NENS} trajectories, {n_steps:,} steps, l={L}")
    print(f"  ch.py, numpy, one at a time   {t_np / 60:6.1f} min")
    print(f"  this, {device.type}, {BATCH} batched      "
          f"{t_gpu / 60:6.1f} min   ({t_np / t_gpu:.1f}x)")


def main():
    ap = argparse.ArgumentParser(
        description=(__doc__ or "GPU Cahn-Hilliard simulation").splitlines()[0])
    ap.add_argument("--end", type=float, default=END,
                    help=f"simulation end time END (default {END})")
    ap.add_argument("--verify", type=int, nargs="?", const=3, metavar="N",
                    help="compare N trajectories against ch.py's numpy path")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force the CPU backend")
    ap.add_argument("--out", default="",
                    help="output .npz (default Work/data/test_t<END>.npz)")
    a = ap.parse_args()
    out = os.path.abspath(a.out) if a.out else ""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not out:
        from rno_train import data_path
        out = os.path.abspath(data_path(a.end))
    if not out.endswith(".npz"):
        out += ".npz"
    end = int(a.end) if float(a.end).is_integer() else float(a.end)
    dev = torch.device("cpu") if a.cpu else None

    if a.verify:
        raise SystemExit(0 if verify(a.verify, dev, end) else 1)
    if a.benchmark:
        benchmark(dev, end)
    elif os.path.exists(out):
        print(f"{out} already exists; move it aside or pass --out to "
              f"regenerate", flush=True)
        raise SystemExit(1)
    else:
        generate(out, dev, end)


if __name__ == "__main__":
    main()
