#!/usr/bin/env python
"""RNO for the Cahn-Hilliard-Cook equation -- prediction and diagnostics.

    python rno_predict.py                       END = 1000, t_end = 10000
    python rno_predict.py --end 300             t_end = PAIRS[300] = 10000
    python rno_predict.py --end 1000 --variant base    direct-increment head
    python rno_predict.py --end 1000 --width 24 --modes 12 --layers 3
    python rno_predict.py --end 1000 --seed 1   a training-seed replicate
    python rno_predict.py --gt-only --t-end 10000
    python rno_predict.py --smoke

The protocol, seeds, cases, metrics, cache layout and figures are those of
the UNO project (uno_predict.py), so the two models are scored on exactly
the same trajectories.  One run is labelled run_label(END, t_end, variant,
size tag), e.g. rno_t1000_lap_pred_t10000 (Laplacian-output head, the
default --variant lap), rno_t1000_pred_t10000 (--variant base) or
rno_t1000_lap_w24m12_pred_t10000 (--width 24 --modes 12; --layers L adds
d<L>, --seed N adds s<N>):

1. the log goes to Work/predict/<label>/predict.log;
2. ground-truth prewarm into Work/gt_cache/: test + tune seeds for every
   quench at l = 128 and paper_seeds for every box size, solved in batches
   that stream frames to disk (or found there: the UNO project's cache can
   be linked in).  --gt-only stops here, before any model is loaded;
3. the model Work/models/rno_t<END>_lap/rno_ch_best.pt (base: rno_t<END>/;
   size-tagged: rno_t<END>_lap_m24/; or --ckpt) is loaded, and its arch
   must match --variant, --width, --modes, --layers and --seed;
4. for each quench the stride schedule is tuned on the tune seeds (or
   reused), then every test seed is rolled out from psi(t_start) alone, with
   an empty memory that the RNO then carries from step to step, into the
   per-seed cache Work/rollout_cache/<label>/<case>/seed<N>.npz.  The box
   sizes use the critical quench's schedule.  Seeds already cached for the
   same checkpoint, label and schedule are skipped, so a killed run
   resumes.  A cache of the same label built from another checkpoint is
   superseded: its rollout_cache, predict and Results folders move (never
   deleted) to Archive/superseded/<label>_<old fingerprint>_<time>/ first;
5. diagnostics on the reference trajectory (first tune seed, critical):
   the 2 x 2 tiling test over carried steps, one-step R^2, intrinsic
   predictability, coarsening drift, the low-k response and the speedup;
6. Work/predict/<label>/results.json;
7. make_submission.draw() draws Results/<label>/.

Snapshots: CONFIG['snapshots'] is the fixed main set (those <= t_end) and
every multiple of CONFIG['long_snap_every'] in (t_start, t_end] is a long
snapshot (2000, 4000, ..., 10000 for t_end = 10000); main, long and t_end are
all stored in the per-seed cache and are must-hit times of every schedule.
Tune scores are taken at (t_end/4, t_end/2, 3 t_end/4, t_end).
--smoke loads the model that `rno_train.py --smoke` wrote and writes only
under Work/smoke/.  All settings live in CONFIG below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import zipfile
from datetime import datetime
from typing import Any

import numpy as np
import torch

import make_submission as ms
from make_submission import (a_to_psi, comp_case, domain_length,
                             dump_json, first_below, fingerprint,
                             free_energy_density, pair_correlation, psi_to_a,
                             r2_score, savez_compressed_atomic, seed_file,
                             structure_factor)
from rno_train import (CHRNO, LOWK_SHELLS, LOWK_DT, RESULTS, SMOKE_RESULTS,
                       SMOKE_ROOT, VARIANTS, WORK, alive_mask,
                       context_frames, context_span, data_horizon,
                       default_t_end, find_gt, gt_dir, gt_provenance,
                       ground_truth, ground_truth_batch, integrate, load_gt,
                       lowk_gains, lowk_plan, make_schedule, mask_state,
                       model_dir, predict_dir, report_dir, rollout_dir,
                       run_label, run_tag, set_precision, setup_log,
                       tile_check, to_project_dir, variant_name, warm_memory)
from rno_train import CONFIG as TRAIN_CONFIG
from rno_train import SMOKE as TRAIN_SMOKE

PHYSICS = (a_to_psi, domain_length, first_below, free_energy_density,
           pair_correlation, psi_to_a, r2_score, structure_factor)

CONFIG: dict[str, Any] = dict(
    train_end    = 1000,
    t_end        = 0.0,
    run_name     = "rno_ch",
    ckpt         = "",
    laplacian_output = True,

    l            = 128,
    eval_offs    = (-0.4, -0.2, 0.0, 0.2, 0.4),
    test_seeds   = (99999, 54321, 77777, 31415, 27182, 61803, 41421, 57721,
                    66260, 86420, 20264, 22348, 22911, 27614, 27924, 28096,
                    29935, 31458, 32393, 33052, 33672, 34617, 35064, 35973,
                    36643, 36963, 39748, 42192, 43125, 48417, 51170, 52066,
                    54464, 58713, 60079, 63866, 67517, 68620, 70389, 76145,
                    76214, 77086, 84331, 85507, 88080, 88971, 92280, 94245,
                    98029, 98895),
    t_start      = 100.0,
    threshold    = 0.80,
    diverge_abs  = 10.0,
    snapshots    = (100.0, 500.0, 1000.0, 2000.0),
    long_snap_every = 2000.0,
    solver_dt    = 0.01,
    save_every   = 4.0,
    rollout_cells= 10 * 128 * 128,
    gt_require_cuda = "auto",

    alpha        = 0.20,
    dt_min       = 1.0,
    dt_max       = 40.0,

    do_tune      = True,
    tune_seeds   = (31337, 24601, 8675309, 20250, 42424,
                    90210, 23571, 55555, 48151, 62831),
    tune_report  = (),
    tune_alphas  = (0.10, 0.20, 0.35, 0.50, 0.70),
    tune_dt_maxs = (30.0, 40.0, 45.0, 50.0),

    paper_sizes       = (160, 192, 256),
    paper_size_off    = 0.0,
    paper_seeds       = (99999, 54321, 77777, 31415, 27182, 61803, 41421,
                         57721, 66260, 86420, 20264, 22348, 22911, 27614,
                         27924, 28096, 29935, 31458, 32393, 33052, 33672,
                         34617, 35064, 35973, 36643, 36963, 39748, 42192,
                         43125, 48417, 51170, 52066, 54464, 58713, 60079,
                         63866, 67517, 68620, 70389, 76145, 76214, 77086,
                         84331, 85507, 88080, 88971, 92280, 94245, 98029, 98895),
    paper_save_every  = 0.0,

    do_transfer       = True,
    do_predictability = True,
    predictability_eps= (1e-4, 1e-3, 1e-2),
    do_drift          = True,
    do_lowk           = True,
    lowk_t0s          = (500.0, 1000.0, 2000.0, 5000.0),
    do_speedup        = True,
    drift_dt          = 20.0,
    drift_times       = (100, 150, 200, 300, 400, 600, 800, 1100, 1400,
                         1700, 1900),
    transfer_t0s      = (100.0, 1000.0, 1900.0),
    train_t_end       = 0.0,
    do_draw           = True,

    device       = ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu"),
)

SMOKE: dict[str, Any] = dict(
    t_end=400.0, save_every=10.0, snapshots=(100.0, 200.0),
    long_snap_every=100.0,
    test_seeds=(99999, 54321), eval_offs=(0.0, -0.4),
    do_predictability=False, drift_times=(100, 200, 300),
    tune_seeds=(31337,), tune_alphas=(0.2,), tune_dt_maxs=(40.0,),
    paper_sizes=(160,), paper_seeds=(99999, 54321))

TRAIN_KEYS = ms.TRAIN_KEYS


def log(*a):
    print(*a, flush=True)


def on_grid(x, t0, ev):
    """The ground-truth frame time nearest x: t0 + k * save_every."""
    return round(t0 + round((x - t0) / ev) * ev, 6)


def time_grids(cfg):
    """Snapshot, tune, drift, transfer and predictability times for t_end.

    snapshots holds main + long + t_end; main_snaps and long_snaps the two
    sets the figures draw.  The tune, drift and predictability grids are
    fractions of t_end.
    """
    t0, t1, ev = cfg["t_start"], cfg["t_end"], cfg["save_every"]
    main = sorted({on_grid(float(x), t0, ev) for x in cfg["snapshots"]
                   if t0 - 1e-9 <= x <= t1 + 1e-9})
    step = float(cfg["long_snap_every"] or 0.0)
    n_long = int(math.floor(t1 / step + 1e-9)) if step > 0 else 0
    long = sorted({on_grid(k * step, t0, ev) for k in range(1, n_long + 1)
                   if t0 + 1e-9 < k * step <= t1 + 1e-9})
    snaps = sorted(set(main) | set(long) | {float(t1)})
    rep = cfg["tune_report"] or (t1 / 4, t1 / 2, 0.75 * t1, t1)
    rep = sorted({on_grid(float(x), t0, ev) for x in rep
                  if t0 + 1e-9 < x <= t1 + 1e-9}) or [float(t1)]
    extra = []
    if t1 > 2000:
        extra = [float(round(2000 * (0.95 * t1 / 2000) ** (k / 5)))
                 for k in range(1, 6)]
    marks = sorted({on_grid(x, t0, ev) for x in (t1 / 4, t1 / 2, t1)
                    if x > t0 + 1e-9})
    return dict(snapshots=tuple(snaps), main_snaps=tuple(main),
                long_snaps=tuple(long), tune_report=tuple(rep),
                drift_times=tuple(float(x) for x in cfg["drift_times"])
                + tuple(extra),
                transfer_t0s=tuple(float(x) for x in cfg["transfer_t0s"])
                + tuple(extra),
                predict_marks=tuple(marks))


def resolve_config(a):
    """CONFIG (+SMOKE) with END, t_end, label, paths and time grids."""
    cfg: dict[str, Any] = dict(CONFIG)
    if a.smoke:
        cfg.update(SMOKE)
    if a.end is not None:
        cfg["train_end"] = a.end
    end = int(round(cfg["train_end"]))
    t_end = float(a.t_end or cfg["t_end"] or default_t_end(end))
    k = (t_end - cfg["t_start"]) / cfg["save_every"]
    if t_end <= cfg["t_start"] or abs(k - round(k)) > 1e-6:
        raise SystemExit(f"t_end = {t_end:g} must lie on the ground-truth "
                         f"grid t_start + k * save_every = {cfg['t_start']:g}"
                         f" + k * {cfg['save_every']:g}")
    root = SMOKE_ROOT if a.smoke else WORK
    results = SMOKE_RESULTS if a.smoke else RESULTS
    variant = getattr(a, "variant", None) or variant_name(
        cfg["laplacian_output"])
    cfg["laplacian_output"] = variant == "lap"
    base_width, base_modes = TRAIN_CONFIG["width"], TRAIN_CONFIG["modes"]
    base_layers = TRAIN_CONFIG["layers"]
    want_width, want_modes = base_width, base_modes
    want_layers = base_layers
    if a.smoke:
        want_width = TRAIN_SMOKE.get("width", want_width)
        want_modes = TRAIN_SMOKE.get("modes", want_modes)
        want_layers = TRAIN_SMOKE.get("layers", want_layers)
    if getattr(a, "modes", None):
        want_modes = base_modes = int(a.modes)
    if getattr(a, "width", None):
        want_width = base_width = int(a.width)
    if getattr(a, "layers", None):
        want_layers = base_layers = int(a.layers)
    seed = int(getattr(a, "seed", None) or 0)
    context = int(getattr(a, "context", None) or 0)
    tag = run_tag(base_width, base_modes, seed, base_layers, context)
    label = run_label(end, t_end, variant, tag)
    cfg.update(size_tag=tag, want_width=want_width, want_modes=want_modes,
               want_layers=want_layers, want_seed=seed, context=context,
               context_dt=float(TRAIN_CONFIG["context_dt"]))
    cfg.update(variant=variant, train_end=end, t_end=t_end,
               smoke=bool(a.smoke), root=root,
               results=results, label=label, gt_dir=gt_dir(root),
               predict_dir=predict_dir(label, root),
               rollout_dir=rollout_dir(label, root),
               report_dir=report_dir(label, results))
    cfg["ckpt"] = (a.ckpt or cfg["ckpt"]
                   or os.path.join(model_dir(end, root, variant, tag),
                                   f"{cfg['run_name']}_best.pt"))
    cfg.update(time_grids(cfg))
    return cfg


def quench_offs(cfg):
    """Quenches that need a schedule: every eval off plus the box quench."""
    offs = [float(o) for o in cfg["eval_offs"]]
    box = float(cfg["paper_size_off"])
    if cfg["paper_sizes"] and all(abs(o - box) > 1e-9 for o in offs):
        offs.append(box)
    return offs


def box_save_every(cfg):
    return float(cfg["paper_save_every"] or cfg["save_every"])


def run_cases(cfg):
    """Case table: the five compositions at l, then the box sizes."""
    cases: dict[str, dict[str, Any]] = {}
    for off in cfg["eval_offs"]:
        cases[comp_case(off)] = dict(
            kind="composition", l=int(cfg["l"]), off=float(off),
            schedule=f"{off:+.1f}", save_every=float(cfg["save_every"]),
            seeds=[int(s) for s in cfg["test_seeds"]])
    box = float(cfg["paper_size_off"])
    for n in cfg["paper_sizes"]:
        cases[f"l{int(n)}"] = dict(
            kind="box", l=int(n), off=box, schedule=f"{box:+.1f}",
            save_every=box_save_every(cfg),
            seeds=[int(s) for s in cfg["paper_seeds"]])
    return cases


def gt_plan(cfg):
    """(l, off, seeds, save_every) of every ground-truth set the run needs."""
    tune = [int(s) for s in cfg["tune_seeds"]]
    plan = [(int(cfg["l"]), off, [int(s) for s in cfg["test_seeds"]] + tune,
             float(cfg["save_every"])) for off in quench_offs(cfg)]
    for n in cfg["paper_sizes"]:
        plan.append((int(n), float(cfg["paper_size_off"]),
                     [int(s) for s in cfg["paper_seeds"]],
                     box_save_every(cfg)))
    return plan


def prewarm(cfg, device):
    """Solve, or find already cached, every ground-truth file of the run."""
    t0 = time.time()
    log(f"[gt] prewarm to t = {cfg['t_end']:g} in {cfg['gt_dir']}")
    total = solved = 0
    for l, off, seeds, ev in gt_plan(cfg):
        n = ground_truth_batch(cfg["gt_dir"], l, off, seeds, cfg["t_start"],
                               cfg["t_end"], ev, cfg["solver_dt"], device,
                               require_cuda=cfg["gt_require_cuda"])
        total += len(set(seeds))
        solved += n
        log(f"[gt]   l={l} off={off:+.1f} ev={ev:g}: {len(set(seeds))} "
            f"trajectories, {n} solved, {len(set(seeds)) - n} cached")
    if cfg["context"]:
        c0, c1 = context_span(cfg["t_start"], cfg["context"],
                              cfg["context_dt"])
        for l, off, seeds, _ in gt_plan(cfg):
            n = ground_truth_batch(cfg["gt_dir"], l, off, seeds, c0, c1,
                                   cfg["context_dt"], cfg["solver_dt"],
                                   device, require_cuda=cfg["gt_require_cuda"])
            log(f"[gt]   context l={l} off={off:+.1f} t = {c0:g} -> {c1:g} "
                f"every {cfg['context_dt']:g}: {n} solved")
    log(f"[gt] {total} trajectories ready ({solved} solved now) in "
        f"{time.time() - t0:.0f}s")


def frame_index(gt_t, times, what=""):
    """GT frame index of every rollout time; the times must match exactly."""
    gt_t = np.asarray(gt_t, dtype=np.float64)
    idx = np.clip(np.searchsorted(gt_t, np.asarray(times) - 1e-6), 0,
                  len(gt_t) - 1)
    bad = np.abs(gt_t[idx] - np.asarray(times)) >= 1e-6
    assert not bad.any(), (f"{what}: no ground-truth frame at t = "
                           f"{np.asarray(times)[bad][:5]}")
    return [int(i) for i in idx]


def run_context(cfg, l, off, seeds, device):
    """(k, len(seeds), l, l) context frames before t_start, or None when the
    run sees psi(t_start) only (cfg['context'] = 0, the default)."""
    if not cfg["context"]:
        return None
    return context_frames(cfg["gt_dir"], l, off, [int(s) for s in seeds],
                          cfg["t_start"], cfg["context"], cfg["context_dt"],
                          cfg["solver_dt"], device,
                          require_cuda=cfg["gt_require_cuda"])


def gt_frames(cfg, l, off, seed, ev, device):
    """(times, fields memmap, path) of one ground truth, cut to t_end."""
    path = ground_truth(cfg["gt_dir"], l, off, seed, cfg["t_start"],
                        cfg["t_end"], ev, cfg["solver_dt"], device,
                        quiet=True, require_cuda=cfg["gt_require_cuda"])
    t, f = load_gt(path, cfg["t_end"])
    return t, f, path


@torch.no_grad()
def rollout(model, psi0, times, device, dtype, truth=None, return_div=False,
            diverge_abs=None, warm=None, warm_dt=0.0):
    """Autoregressive rollout over all seeds at once.

    The RNO starts from psi0 with an empty memory (or one warmed on warm, a
    (k, b, l, l) array of ground-truth frames warm_dt apart that end
    warm_dt before psi0, in a --context run) and carries its hidden states
    from step to step.  A seed diverges when its field goes
    non-finite or max |psi| exceeds diverge_abs (default
    CONFIG['diverge_abs']).  It is frozen at its last good state and
    memory, so no NaN reaches the model, and its stored states and R^2 are
    NaN from that step on; the other seeds keep rolling, which is exact
    because the model acts on every sample separately.  The loop stops early
    only when no seed is alive, and the rows then end before that step.

    Returns (states, r2), plus div_at when return_div is True: the time
    each seed diverged at, NaN for a seed that never did.  Without a
    divergence states and r2 are exactly what the unmasked rollout gave.
    """
    limit = float(CONFIG["diverge_abs"] if diverge_abs is None
                  else diverge_abs)
    psi = torch.as_tensor(psi0, dtype=dtype, device=device)
    if psi.ndim == 3:
        psi = psi.unsqueeze(1)
    b = psi.shape[0]
    alive = torch.ones(b, dtype=torch.bool, device=device)
    dead = np.zeros(b, dtype=bool)
    div_at = np.full(b, np.nan)
    states = [psi[:, 0].float().cpu().numpy().copy()]
    r2 = [[1.0 if truth is None else r2_score(states[0][k], truth[0][k])
           for k in range(b)]]
    memory = None
    if warm is not None and len(warm):
        memory = warm_memory(model, [torch.as_tensor(w, dtype=dtype,
                                                     device=device)[:, None]
                                     for w in warm], warm_dt)
    for i in range(1, len(times)):
        dt = torch.full((b,), float(times[i] - times[i - 1]), device=device,
                        dtype=dtype)
        new, new_memory = model(psi, dt, memory)
        newly = alive & ~alive_mask(new, limit)
        if bool(newly.any()):
            gone = newly.cpu().numpy()
            div_at[gone] = float(times[i])
            dead |= gone
            alive = alive & ~newly
            none_left = not bool(alive.any())
            log(f"  WARNING: {int(gone.sum())} of {b} rollout(s) diverged "
                f"(non-finite or |psi| > {limit:g}) at t = {times[i]:g}; "
                + ("none left, stopping" if none_left
                   else "the others continue"))
            if none_left:
                break
        psi = torch.where(alive.view(-1, 1, 1, 1), new, psi)
        memory = mask_state(alive, new_memory, memory)
        states.append(psi[:, 0].float().cpu().numpy().copy())
        if dead.any():
            states[-1][dead] = np.nan
        if truth is not None:
            r2.append([float("nan") if dead[k]
                       else r2_score(states[-1][k], truth[i][k])
                       for k in range(b)])
    if return_div:
        return np.array(states), np.array(r2), div_at
    return np.array(states), np.array(r2)


def scored_r2(r2, times, div_at):
    """(len(times), b) rollout R^2 as the tuning score counts it.

    A diverged seed is R^2 = -1 at every time from its divergence time on
    (the convention of rno_train.eval_rollout), never its last value held
    flat; a rollout that stopped early is padded with -1 the same way.
    """
    r2 = np.asarray(r2, dtype=np.float64)
    n = len(times)
    if r2.shape[0] < n:
        r2 = np.concatenate([r2, np.full((n - r2.shape[0], r2.shape[1]),
                                         -1.0)])
    else:
        r2 = r2.copy()
    tv = np.asarray(times, dtype=np.float64)
    for k, t in enumerate(div_at):
        if np.isfinite(t):
            r2[tv >= t - 1e-6, k] = -1.0
    return r2


def tune(model, cfg, device, dtype, off):
    """Pick the stride schedule on the validation seeds, never a test seed."""
    seeds = [int(s) for s in cfg["tune_seeds"]]
    report = list(cfg["tune_report"])
    hits = tuple(sorted(set(report) | set(cfg["snapshots"])))
    grid = ", ".join(f"{t:g}" for t in report)
    log(f"[tuning] stride schedule for psi0 = {off:+.1f} on {len(seeds)} "
        f"validation seed(s) {seeds} (held out of the reported test set)")
    log(f"[tuning] score = mean R2 at t = {grid}; a fixed grid, so schedules "
        f"of different length stay comparable")
    if not seeds:
        return cfg["alpha"], cfg["dt_max"]
    ground_truth_batch(cfg["gt_dir"], cfg["l"], off, seeds, cfg["t_start"],
                       cfg["t_end"], cfg["save_every"], cfg["solver_dt"],
                       device, require_cuda=cfg["gt_require_cuda"])
    fields, gt_t = [], None
    for sd in seeds:
        gt_t, f, _ = gt_frames(cfg, cfg["l"], off, sd, cfg["save_every"],
                               device)
        fields.append(f)
    assert gt_t is not None
    q = float(gt_t[1] - gt_t[0])
    warm = run_context(cfg, cfg["l"], off, seeds, device)

    head = "".join(f"{'R2(' + format(t, 'g') + ')':>10}" for t in report)
    log(f"  {'alpha':>7} {'dt_max':>7} {'steps':>6} {'score':>9}{head}"
        f"{'div':>5}")

    scores = []
    for a in cfg["tune_alphas"]:
        for dm in cfg["tune_dt_maxs"]:
            times = make_schedule(cfg["t_start"], cfg["t_end"], a,
                                  cfg["dt_min"], dm, quantum=q,
                                  must_hit=hits)
            idx = frame_index(gt_t, times, "tune")
            truth = np.stack([np.asarray(f[idx]) for f in fields],
                             axis=1).astype(np.float64)
            _, r2b, div = rollout(model, truth[0], times, device, dtype,
                                  truth=truth, return_div=True,
                                  diverge_abs=cfg["diverge_abs"],
                                  warm=warm, warm_dt=cfg["context_dt"])
            r2 = scored_r2(r2b, times, div).mean(axis=1)
            tv = np.asarray(times, dtype=np.float64)
            at = [float(r2[int(np.argmin(np.abs(tv - tt)))])
                  for tt in report]
            sc = float(np.mean(at))
            scores.append((sc, float(a), float(dm), len(times) - 1))
            log(f"  {a:7.3f} {dm:7.1f} {len(times) - 1:6d} {sc:9.4f}"
                + "".join(f"{v:10.4f}" for v in at)
                + f"{int(np.isfinite(div).sum()):5d}")
            del truth

    best = max(scores, key=lambda x: x[0])
    log(f"  best: alpha={best[1]:g}, dt_max={best[2]:g} "
        f"({best[3]} steps) -> score {best[0]:.4f}")
    if best[2] >= max(cfg["tune_dt_maxs"]) - 1e-9:
        log("  [warn] dt_max sits at the top of the grid. Longer strides keep "
            "winning because they compound the per-step bias fewer times; do "
            "not widen the grid beyond the dt_max the model trained on or the "
            "step itself becomes an extrapolation.")
    if best[1] <= min(cfg["tune_alphas"]) + 1e-9:
        log("  [warn] alpha sits at the bottom of the grid")
    return best[1], best[2]


def arch_size_mismatch(ck, cfg):
    """Text naming the width/modes/layers/context/training seed of the
    checkpoint that differ from the requested ones, or '' when they
    agree."""
    arch = ck.get("arch", {})
    tc = ck.get("config") or {}
    got = (arch.get("width"), arch.get("modes"), arch.get("layers"),
           int(tc.get("context", 0)))
    want = (cfg["want_width"], cfg["want_modes"], cfg["want_layers"],
            cfg["context"])
    seed = int(tc.get("seed", 0))
    if got == want and seed == cfg["want_seed"]:
        return ""
    return (f"arch width = {got[0]}, modes = {got[1]}, layers = {got[2]}, "
            f"context = {got[3]}, training seed = {seed} but this run asks "
            f"for width = {want[0]}, modes = {want[1]}, layers = {want[2]}, "
            f"context = {want[3]}, seed = {cfg['want_seed']}")


def check_arch_size(ck, cfg, ckpt):
    bad = arch_size_mismatch(ck, cfg)
    if bad:
        raise SystemExit(
            f"{ckpt} has {bad} ({cfg['label']}); pass --modes "
            f"{ck['arch'].get('modes')} --width {ck['arch'].get('width')} "
            f"--layers {ck['arch'].get('layers')} "
            f"--context {int((ck.get('config') or {}).get('context', 0))} "
            f"--seed {int((ck.get('config') or {}).get('seed', 0))} "
            f"for this checkpoint")


def load_model(cfg, device, dtype):
    ckpt = cfg["ckpt"]
    if not os.path.exists(ckpt):
        var = f" --variant {cfg['variant']}"
        if cfg["size_tag"]:
            var += (f" --modes {cfg['want_modes']}"
                    f" --width {cfg['want_width']}"
                    f" --layers {cfg['want_layers']}"
                    f" --context {cfg['context']}"
                    f" --seed {cfg['want_seed']}")
        hint = (f"python3 rno_train.py --smoke --end {cfg['train_end']}{var}"
                if cfg["smoke"] else
                f"python rno_train.py --end {cfg['train_end']}{var}")
        raise SystemExit(f"checkpoint not found: {ckpt} -- run {hint}, or "
                         f"pass --ckpt")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    lap = bool(ck["arch"].get("laplacian_output", False))
    if lap != (cfg["variant"] == "lap"):
        raise SystemExit(
            f"{ckpt} has arch.laplacian_output = {lap}, but this run is "
            f"--variant {cfg['variant']} ({cfg['label']}); pass --variant "
            f"{variant_name(lap)} for this checkpoint")
    check_arch_size(ck, cfg, ckpt)
    cfg["context_dt"] = float((ck.get("config") or {}).get(
        "context_dt", cfg["context_dt"]))
    model = CHRNO(**ck["arch"]).to(device, dtype=dtype).eval()
    model.load_state_dict(ck["model"])
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[model] {ckpt}  {n_par:,} parameters  arch={ck['arch']}")
    log(f"[model] training metrics: {ck.get('metrics')}")
    return model, ck, n_par


def checkpoint_horizon(ck, cfg):
    """Training horizon of the checkpoint, and where it was read from.

    ck['train_t_end'] first; old checkpoints fall back to CONFIG
    ['train_t_end'] and then to the memmap in the checkpoint's own folder.
    """
    if ck.get("train_t_end"):
        return float(ck["train_t_end"]), "checkpoint"
    if cfg["train_t_end"]:
        return float(cfg["train_t_end"]), "CONFIG['train_t_end']"
    mm = os.path.join(os.path.dirname(cfg["ckpt"]), "cache", "train_X_f32.npy")
    if os.path.exists(mm):
        if os.path.getmtime(mm) > os.path.getmtime(cfg["ckpt"]):
            log(f"[model] WARNING: {mm} is newer than the checkpoint; the "
                f"horizon read from it may not be the one the model saw")
        return data_horizon(np.load(mm, mmap_mode="r").shape[1]), mm
    return None, "unknown: set CONFIG['train_t_end']"


def check_horizon(ck, cfg):
    t_train, src = checkpoint_horizon(ck, cfg)
    log(f"[model] training data ends at t = {t_train} ({src})")
    if t_train is not None and int(round(t_train)) != cfg["train_end"]:
        raise SystemExit(
            f"the checkpoint was trained to t = {t_train:g} ({src}), but "
            f"this run is labelled END = {cfg['train_end']} "
            f"({cfg['label']}); pass --end {int(round(t_train))}")
    return t_train, src


def check_seeds(cfg):
    leak = sorted(set(cfg["tune_seeds"]) & set(cfg["test_seeds"]))
    assert not leak, (
        f"tune seeds {leak} are also test seeds; the stride schedule would "
        f"be selected on data it is then scored on")
    low = [s for s in list(cfg["test_seeds"]) + list(cfg["paper_seeds"])
           if s < 20000]
    assert not low, (f"test seeds {low} overlap ch.py's training [0,10000) / "
                     f"validation [10000,20000) ranges")


def open_cache(cfg, fp):
    """The run's rollout meta.json, refusing another model or horizon."""
    mpath = os.path.join(cfg["rollout_dir"], "meta.json")
    if not os.path.exists(mpath):
        return {}
    with open(mpath) as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fp:
        raise SystemExit(
            f"{cfg['rollout_dir']} was built from checkpoint "
            f"{meta.get('fingerprint')}, but {cfg['ckpt']} is {fp}. Mixing "
            f"two models in one cache would corrupt every figure; move the "
            f"folder aside first.")
    for k in ("t_start", "t_end"):
        if abs(float(meta.get(k, cfg[k])) - float(cfg[k])) > 1e-6:
            raise SystemExit(f"{mpath} has {k} = {meta.get(k)}, this run "
                             f"{cfg[k]:g}")
    return meta


def archive_superseded(cfg):
    """Move a same-label run built from another checkpoint into the archive.

    Returns the log lines (the run log is not open yet when this runs, since
    its folder is one of the folders that move).  Nothing is deleted.  A
    checkpoint whose training horizon does not match the run's END, or whose
    architecture does not match the variant or the size, moves nothing, so
    the later check can stop the run with the old run intact.
    """
    mpath = os.path.join(cfg["rollout_dir"], "meta.json")
    if not (os.path.exists(mpath) and os.path.exists(cfg["ckpt"])):
        return []
    with open(mpath) as f:
        old = json.load(f).get("fingerprint")
    if old is None or old == fingerprint(cfg["ckpt"]):
        return []
    ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    is_lap = bool(ck.get("arch", {}).get("laplacian_output", False))
    if is_lap != (cfg["variant"] == "lap") or arch_size_mismatch(ck, cfg):
        return []
    t_train = ck.get("train_t_end") or cfg["train_t_end"]
    if t_train and int(round(float(t_train))) != cfg["train_end"]:
        return []
    base = os.path.join(cfg["root"] if cfg["smoke"] else ".", "Archive")
    dest = os.path.join(
        base, "superseded",
        f"{cfg['label']}_{old}_{datetime.now():%Y%m%d-%H%M%S}")
    lines = [f"[superseded] {cfg['rollout_dir']} was built from checkpoint "
             f"{old}, this run uses {fingerprint(cfg['ckpt'])}"]
    for name, src in (("rollout_cache", cfg["rollout_dir"]),
                      ("predict", cfg["predict_dir"]),
                      ("Results", cfg["report_dir"])):
        if os.path.exists(src):
            os.makedirs(dest, exist_ok=True)
            shutil.move(src, os.path.join(dest, name))
            lines.append(f"[superseded] moved {src} -> "
                         f"{os.path.join(dest, name)}")
    return lines


def schedule_source(cfg, fp, schedule_from):
    """Stride schedules from a results.json of the same checkpoint, or None.

    'auto' is Work/predict/<label>/results.json and is ignored (with a log
    line) when it is missing or was written for another checkpoint; an
    explicit path that does not match stops the run.  The match is on the
    sha256 fingerprint, since every model is called rno_ch_best.pt.
    """
    if not schedule_from:
        return None
    auto = schedule_from == "auto"
    path = (os.path.join(cfg["predict_dir"], "results.json") if auto
            else schedule_from)
    if not os.path.exists(path):
        if auto:
            log(f"[schedule] --schedule-from auto: no {path}")
            return None
        raise SystemExit(f"--schedule-from {path}: no such file")
    with open(path) as f:
        res = json.load(f)
    got = res.get("fingerprint")
    old = res.get("checkpoint") or ""
    if got is None and os.path.exists(old) and fingerprint(old) == fp:
        got = fp
    if got != fp:
        who = got or "unknown (no fingerprint recorded)"
        why = f"{path} was written for checkpoint {who}, not {fp}"
        if auto:
            log(f"[schedule] {why}; ignoring it")
            return None
        raise SystemExit(why)
    log(f"[schedule] reusing the schedules in {path}")
    return dict(path=path, schedule=res["schedule"])


def pick_schedule(model, cfg, meta, src, off, device, dtype):
    """--schedule-from, else the cached meta, else tune, else CONFIG."""
    key = f"{off:+.1f}"
    old = (meta.get("schedule") or {}).get(key)
    if src is not None and f"off{key}" in src["schedule"]:
        s = src["schedule"][f"off{key}"]
        out = dict(alpha=float(s["alpha"]), dt_max=float(s["dt_max"]),
                   source=src["path"])
    elif old is not None:
        out = dict(alpha=float(old["alpha"]), dt_max=float(old["dt_max"]),
                   source=old.get("source", "cached rollout meta"))
    elif cfg["do_tune"]:
        a, dm = tune(model, cfg, device, dtype, off)
        out = dict(alpha=float(a), dt_max=float(dm),
                   source=f"tuned on {len(cfg['tune_seeds'])} validation "
                          f"seeds")
    else:
        out = dict(alpha=float(cfg["alpha"]), dt_max=float(cfg["dt_max"]),
                   source="CONFIG alpha / dt_max")
    out["dt_min"] = float(cfg["dt_min"])
    log(f"[schedule] psi0 = {key}: alpha = {out['alpha']:g}, dt_max = "
        f"{out['dt_max']:g}  ({out['source']})")
    return out


def cached_ok(path, fp, label, s, t_end):
    """A seed file already rolled out by this checkpoint, run and schedule.

    A file with the diverged_at key is final, full length or cut at its own
    divergence.  One without it is accepted only when it reaches t_end: an
    older short file is the whole-chunk truncation (healthy seeds lost their
    trajectory with the one that blew up) and is rolled out again.
    """
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as z:
            f = z.files
            ok = (str(z["fingerprint"]) == fp
                  and abs(float(z["alpha"]) - s["alpha"]) < 1e-9
                  and abs(float(z["dt_max"]) - s["dt_max"]) < 1e-9)
            if "label" in f:
                ok = ok and str(z["label"]) == label
            if "t_end" in f:
                ok = ok and abs(float(z["t_end"]) - t_end) < 1e-6
            if ok and "diverged_at" not in f:
                ok = float(z["times"][-1]) >= t_end - 1e-6
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False
    return bool(ok)


def write_meta(cfg, meta):
    order: list[int] = []
    for c in meta["cases"].values():
        order += [int(x) for x in c.get("seeds") or [] if x not in order]
    meta["seeds"] = meta["seed_order"] = order
    meta["updated"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    os.makedirs(cfg["rollout_dir"], exist_ok=True)
    dump_json(os.path.join(cfg["rollout_dir"], "meta.json"), meta)


def roll_case(model, cfg, meta, case, seeds, fp, t_train, device, dtype):
    """Roll one case out for every seed not yet cached; write seed files."""
    c = meta["cases"][case]
    l, off, ev = c["l"], c["off"], c["save_every"]
    s = dict(alpha=c["alpha"], dt_min=c["dt_min"], dt_max=c["dt_max"])
    cdir, label = cfg["rollout_dir"], cfg["label"]
    snaps = [float(x) for x in cfg["snapshots"]]
    os.makedirs(os.path.join(cdir, case), exist_ok=True)
    todo = [sd for sd in seeds
            if not cached_ok(seed_file(cdir, case, sd), fp, label, s,
                             cfg["t_end"])]
    if todo:
        ground_truth_batch(cfg["gt_dir"], l, off, todo, cfg["t_start"],
                           cfg["t_end"], ev, cfg["solver_dt"], device,
                           require_cuda=cfg["gt_require_cuda"])
    devs = set()
    for sd in seeds:
        p = find_gt(cfg["gt_dir"], l, off, sd, cfg["t_start"], cfg["t_end"],
                    ev)
        if p is not None:
            devs.add(gt_provenance(p)["device"] or "unrecorded (older file)")
    c["gt_devices"] = sorted(devs)
    if not todo:
        log(f"[rollout] {case}: all {len(seeds)} seeds cached")
        return 0.0
    per = max(1, int(cfg["rollout_cells"] // (l * l)))
    times: Any = None
    wall = 0.0
    for i in range(0, len(todo), per):
        chunk = todo[i:i + per]
        fields = []
        for sd in chunk:
            gt_t, f, _ = gt_frames(cfg, l, off, sd, ev, device)
            if times is None:
                times = make_schedule(cfg["t_start"], cfg["t_end"],
                                      s["alpha"], s["dt_min"], s["dt_max"],
                                      quantum=float(gt_t[1] - gt_t[0]),
                                      must_hit=snaps)
            fields.append(np.asarray(f[frame_index(gt_t, times, case)]))
        truth = np.stack(fields, axis=1).astype(np.float64)
        del fields
        warm = run_context(cfg, l, off, chunk, device)
        t_w = time.time()
        states, r2, div = rollout(model, truth[0], times, device, dtype,
                                  truth=truth, return_div=True,
                                  diverge_abs=cfg["diverge_abs"],
                                  warm=warm, warm_dt=cfg["context_dt"])
        wall += time.time() - t_w
        tt = np.asarray(times, dtype=float)
        ks = [int(np.argmin(np.abs(tt - x))) for x in snaps]
        for b, sd in enumerate(chunk):
            nv = (len(tt) if np.isnan(div[b])
                  else int(np.searchsorted(tt, div[b] - 1e-6)))
            tr, pr = truth[:nv, b], states[:nv, b]
            blank = np.full((l, l), np.nan)
            savez_compressed_atomic(
                seed_file(cdir, case, sd),
                times=tt[:nv], r2=np.asarray(r2)[:nv, b],
                L_half_truth=np.array([domain_length(x, "half") for x in tr]),
                L_half_pred=np.array([domain_length(x, "half") for x in pr]),
                L_zero_truth=np.array([domain_length(x, "zero") for x in tr]),
                L_zero_pred=np.array([domain_length(x, "zero") for x in pr]),
                mean_psi_truth=tr.mean(axis=(-2, -1)),
                mean_psi_pred=pr.mean(axis=(-2, -1)),
                fe_truth=free_energy_density(tr),
                fe_pred=free_energy_density(pr.astype(np.float64)),
                snap_times=np.array(snaps),
                snap_truth=np.stack([truth[k, b] for k in ks]).astype(
                    np.float32),
                snap_pred=np.stack([states[k, b] if k < nv else blank
                                    for k in ks]).astype(np.float32),
                seed=sd, l=l, off=off, alpha=s["alpha"], dt_min=s["dt_min"],
                dt_max=s["dt_max"], fingerprint=fp, label=label,
                t_end=float(cfg["t_end"]),
                train_t_end=float(t_train or 0.0),
                diverged_at=float(div[b]))
        n_div = int(np.isfinite(div).sum())
        log(f"[rollout] {case}: {i + len(chunk)}/{len(todo)} seeds "
            f"({len(times) - 1} steps each)"
            + (f", {n_div} diverged" if n_div else ""))
        del truth, states
    return wall


def rollout_run(model, ck, n_par, cfg, device, dtype, prec, t_train,
                seeds=0, schedule_from=""):
    """The single rollout engine: schedules, then the per-seed cache.

    Every quench is tuned (or reused) and its composition case rolled out
    before the next quench; the box sizes then run with the critical
    quench's schedule.  Returns the rollout meta.
    """
    fp = fingerprint(cfg["ckpt"])
    meta = open_cache(cfg, fp)
    src = schedule_source(cfg, fp, schedule_from)
    cases = run_cases(cfg)
    tcfg = ck.get("config") or {}
    meta.update(
        label=cfg["label"], fingerprint=fp,
        checkpoint=os.path.abspath(cfg["ckpt"]), arch=ck["arch"],
        precision=prec, n_params=int(n_par), train_t_end=t_train,
        training_metrics=ck.get("metrics"),
        training_config={k: tcfg[k] for k in TRAIN_KEYS if k in tcfg},
        base_l=int(cfg["l"]), box_off=float(cfg["paper_size_off"]),
        t_start=float(cfg["t_start"]), t_end=float(cfg["t_end"]),
        save_every=float(cfg["save_every"]),
        box_save_every=box_save_every(cfg),
        snapshots=[float(x) for x in cfg["snapshots"]],
        main_snapshots=[float(x) for x in cfg["main_snaps"]],
        long_snapshots=[float(x) for x in cfg["long_snaps"]],
        long_snap_every=float(cfg["long_snap_every"]),
        threshold=float(cfg["threshold"]),
        diverge_abs=float(cfg["diverge_abs"]),
        context=int(cfg["context"]), context_dt=float(cfg["context_dt"]),
        created=meta.get("created") or f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    meta.setdefault("schedule", {})
    old_sched = {k: dict(v) for k, v in meta["schedule"].items()}
    old_cases = dict(meta.get("cases") or {})
    old_seeds = [int(x) for x in meta.get("seeds") or meta.get("seed_order")
                 or []]
    meta["cases"] = {}
    meta["planned_cases"] = list(cases)
    meta["complete"] = False
    log(f"[rollout] cache {cfg['rollout_dir']}  checkpoint {fp}")
    walls: dict[str, float] = {}
    todo = [(off, [k for k, c in cases.items() if c["kind"] == "composition"
                   and c["schedule"] == f"{off:+.1f}"])
            for off in quench_offs(cfg)]
    todo.append((None, [k for k, c in cases.items() if c["kind"] == "box"]))
    for off, names in todo:
        if off is not None:
            log("")
            key = f"{off:+.1f}"
            meta["schedule"][key] = pick_schedule(model, cfg, meta, src, off,
                                                  device, dtype)
        for case in names:
            c = cases[case]
            old = old_cases.get(case) or {}
            s = meta["schedule"][c["schedule"]]
            prev = dict(old_sched.get(old.get("schedule", ""), {}))
            prev.update({k: old[k] for k in ("alpha", "dt_max") if k in old})
            same = all(abs(float(prev.get(k, s[k])) - s[k]) < 1e-9
                       for k in ("alpha", "dt_max"))
            had = [int(x) for x in old.get("seeds") or old_seeds] if (
                old and same) else []
            rolled = c["seeds"][:seeds] if seeds else c["seeds"]
            want = set(had) | set(rolled)
            meta["cases"][case] = dict(
                c, seeds=[sd for sd in c["seeds"] if sd in want]
                + [sd for sd in had if sd not in c["seeds"]],
                alpha=s["alpha"], dt_min=s["dt_min"], dt_max=s["dt_max"],
                source=s["source"])
            write_meta(cfg, meta)
            log(f"[rollout] {case}: l={c['l']} psi0={c['off']:+.1f}, "
                f"{len(rolled)} seeds, schedule psi0 = {c['schedule']}")
            walls[case] = roll_case(model, cfg, meta, case, rolled, fp,
                                    t_train, device, dtype)
    meta["gt_devices"] = sorted({d for c in meta["cases"].values()
                                 for d in c.get("gt_devices") or []})
    meta["complete"] = True
    write_meta(cfg, meta)
    return meta, walls


def reference_gt(cfg, device):
    """First tune seed at the critical quench (or the first eval quench)."""
    offs = [float(o) for o in cfg["eval_offs"]]
    off = 0.0 if any(abs(o) < 1e-9 for o in offs) else offs[0]
    seed = int(cfg["tune_seeds"][0])
    t, f, path = gt_frames(cfg, cfg["l"], off, seed, cfg["save_every"], device)
    log(f"[diagnostic] reference trajectory {os.path.basename(path)}")
    return off, t, f


def diag_transfer(model, cfg, gt, device, dtype, t_train):
    log("")
    log("[diagnostic] 64 -> 128 TRANSFER")
    worst = max(tile_check(model, device, dtype, seed=i) for i in range(3))
    log(f"  tile invariance |model(tile) - tile(model)| over 3 carried steps "
        f"(state, increment and every hidden state) = {worst:.3e}   "
        f"{'PASS' if worst < 5e-3 else 'FAIL'}")

    t, f = gt
    seen = (t_train - 1.0) if t_train else float("inf")
    log(f"  one-step R^2 at l={cfg['l']} from an empty memory (training only "
        f"ever saw t <= {seen:g})")
    log(f"    {'t0':>7} {'dt':>6} {'RNO R2':>10} {'persistence':>12} "
        f"{'gain':>9}")
    rows = []
    for t0 in cfg["transfer_t0s"]:
        for want in (5.0, 20.0, 40.0):
            i0 = int(np.argmin(np.abs(t - t0)))
            i1 = int(np.argmin(np.abs(t - (t0 + want))))
            if i1 <= i0 or i1 >= len(t):
                continue
            dt = float(t[i1] - t[i0])
            x = torch.as_tensor(np.array(f[i0]), dtype=dtype,
                                device=device)[None, None]
            with torch.no_grad():
                p = model(x, torch.full((1,), dt, device=device,
                                        dtype=dtype))[0]
            a = r2_score(p[0, 0].float().cpu().numpy(), f[i1])
            b = r2_score(f[i0], f[i1])
            flag = "" if t0 <= seen else "   <- extrapolated in t"
            rows.append(dict(t0=float(t0), dt=dt, rno=a, persistence=b))
            log(f"    {t0:7.0f} {dt:6.1f} {a:10.5f} {b:12.5f} "
                f"{a-b:+9.5f}{flag}")
    return dict(tile_invariance=worst, one_step=rows)


def diag_predictability(cfg, gt, device):
    """How much rollout error is intrinsic rather than the model's.

    The perturbed solves run as one batch; the solver acts elementwise over
    the batch, so each equals its one-at-a-time solve.
    """
    log("")
    log("[diagnostic] INTRINSIC PREDICTABILITY (exact solver)")
    t, f = gt
    psi0 = np.asarray(f[0], dtype=np.float64)
    marks = [x for x in cfg["predict_marks"] if x <= t[-1] + 1e-9]
    if not marks:
        return {}
    idx = frame_index(t, marks, "predictability")
    rng = np.random.default_rng(0)
    eps_list = list(cfg["predictability_eps"])
    ics = []
    for eps in eps_list:
        n = rng.standard_normal(psi0.shape)
        n -= n.mean()
        n *= eps * psi0.std() / n.std()
        ics.append(psi0 + n)
    save = [float(t[i] - t[0]) for i in idx]
    _, got = integrate(np.stack(ics), save[-1], save, dt=cfg["solver_dt"],
                       device=device)
    log("  " + f"{'rel. perturb':>13}  "
        + "  ".join(f"R2(t={x:g})".rjust(11) for x in marks))
    out = {}
    for e, eps in enumerate(eps_list):
        vals = [r2_score(got[j, e], f[i]) for j, i in enumerate(idx)]
        out[f"eps_{eps:g}"] = dict(zip([f"t{int(x)}" for x in marks], vals))
        log(f"  {eps:13.1e}  " + "  ".join(f"{v:11.5f}" for v in vals))
    return out


def diag_drift(model, cfg, gt, device, dtype, t_train, src):
    """Per-step coarsening-rate bias against domain size L.

    t0 and t0 + dt snap to the nearest ground-truth frames, and the model
    steps by the gap between those two frames, so both sides cover one span.
    """
    log("")
    log("[diagnostic] PER-STEP COARSENING BIAS (continuous 'moment' estimator)")
    t, f = gt
    tt = t_train if t_train else float("inf")
    log(f"  training data ends at t = {tt:g} ({src})")
    log(f"  {'t':>7} {'dt':>5} {'L_truth':>9} {'dL_truth':>10} "
        f"{'dL_model':>10} {'bias %':>8}  in train range")
    rows = []
    for t0 in cfg["drift_times"]:
        ddt = min(50.0, max(float(cfg["drift_dt"]), 0.01 * float(t0)))
        i0 = int(np.argmin(np.abs(t - t0)))
        i1 = int(np.argmin(np.abs(t - (t0 + ddt))))
        if i1 <= i0 or i1 >= len(t):
            continue
        ddt = float(t[i1] - t[i0])
        x0 = np.asarray(f[i0], dtype=np.float64)
        with torch.no_grad():
            p = model(torch.as_tensor(x0, dtype=dtype,
                                      device=device)[None, None],
                      torch.full((1,), ddt, device=device, dtype=dtype))[0]
        L0 = domain_length(x0, "moment")
        dt_ = domain_length(np.asarray(f[i1], dtype=np.float64),
                            "moment") - L0
        dm_ = domain_length(p[0, 0].float().cpu().numpy().astype(np.float64),
                            "moment") - L0
        bias = 100.0 * (dm_ - dt_) / abs(dt_) if abs(dt_) > 1e-9 else float("nan")
        rows.append((float(t0), bias))
        log(f"  {t0:7.0f} {ddt:5.0f} {L0:9.2f} {dt_:10.4f} {dm_:10.4f} "
            f"{bias:8.1f}  {'yes' if t0 <= tt else 'NO -- extrapolated'}")
    ins = [b for x, b in rows if x <= tt and np.isfinite(b)]
    out = [b for x, b in rows if x > tt and np.isfinite(b)]
    res: dict[str, Any] = dict(train_t_end=t_train,
                               rows=[dict(t=x, bias_pct=b) for x, b in rows])
    if ins:
        res["mean_bias_inside_pct"] = float(np.mean(ins))
        log(f"  mean bias inside  training range ({len(ins)} pts): "
            f"{np.mean(ins):+.1f} %")
    if out:
        res["mean_bias_outside_pct"] = float(np.mean(out))
        log(f"  mean bias outside training range ({len(out)} pts): "
            f"{np.mean(out):+.1f} %")
    return res


def diag_lowk(model, cfg, off, device, dtype):
    """Teacher-forced increment gain per wavenumber shell on the tune seeds.

    g = sum Re(conj(dP) dT) / sum |dT|^2 over the modes of a shell, from the
    ground-truth frame at t0 to the one dt later (lowk_gains).  The l = 128
    box has 8 modes below the l = 64 training box's fundamental; a model that
    reproduces the dynamics has g = 1 in every shell.
    """
    log("")
    log("[diagnostic] LOW-k INCREMENT GAIN (teacher-forced, per shell)")
    seeds = [int(s) for s in cfg["tune_seeds"]]
    gts = [gt_frames(cfg, cfg["l"], off, sd, cfg["save_every"], device)[:2]
           for sd in seeds]
    dt, t0s = lowk_plan(gts[0][0], cfg["lowk_t0s"], LOWK_DT)
    gains = lowk_gains(model, gts, t0s, dt, device, dtype)
    log(f"  psi0 = {off:+.1f}, l = {cfg['l']}, {len(seeds)} seed(s), "
        f"dt = {dt:g}; below = the 8 modes |n| = 1, sqrt2 (k < the training "
        f"box's 2 pi/64), first = |n| in [2, 2.9], mid = [3, 8], high = rest")
    log(f"  {'t0':>7}" + "".join(f"{k:>9}" for k in LOWK_SHELLS))
    for t0, g in gains.items():
        log(f"  {t0:7.0f}" + "".join(f"{g[k]:9.3f}" for k in LOWK_SHELLS))
    return dict(off=float(off), l=int(cfg["l"]), seeds=seeds, dt=float(dt),
                shells=list(LOWK_SHELLS),
                gain={f"{t0:g}": g for t0, g in gains.items()})


def diag_speedup(cfg, device, dtype, model, gt, times):
    """Surrogate wall time against the explicit solver it replaces."""
    log("")
    log("[diagnostic] SPEEDUP vs the explicit solver")
    if device.type == "mps":
        log("  skipped: the surrogate runs on MPS but the float64 solver "
            "cannot, so it falls back to the CPU. Timing the two on "
            "different devices would report the MPS speedup as if it were "
            "the surrogate's, which is not a like-for-like comparison. "
            "Measure this on the GPU node instead.")
        return dict(skipped="mps: solver and surrogate on different devices")
    n_steps = len(times) - 1
    span = cfg["t_end"] - cfg["t_start"]
    solver_steps = int(round(span / cfg["solver_dt"]))
    ic = np.random.default_rng(0).standard_normal((cfg["l"], cfg["l"])) * 0.05
    n_probe = 2000
    t0 = time.time()
    integrate(ic, n_probe * cfg["solver_dt"], [n_probe * cfg["solver_dt"]],
              dt=cfg["solver_dt"], device=device)
    probe = time.time() - t0
    solver_est = probe / n_probe * solver_steps

    def timed(psi):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_w = time.time()
        rollout(model, psi, times, device, dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.time() - t_w

    psi0 = np.asarray(gt[1][0], dtype=np.float64)[None]
    rollout(model, psi0, times[:3], device, dtype)
    single = timed(psi0)
    per = max(1, int(cfg["rollout_cells"] // (cfg["l"] * cfg["l"])))
    batched = timed(np.repeat(psi0, per, axis=0)) / per

    log(f"  solver: {solver_steps:,} explicit steps of dt={cfg['solver_dt']} "
        f"-> {solver_est:.1f}s per trajectory (one at a time, "
        f"measured on {n_probe} steps)")
    lead = f"  RNO:    {n_steps} autoregressive steps -> "
    log(lead + f"{single:.3f}s per trajectory (one at a time)")
    log(" " * len(lead) + f"{batched:.3f}s per trajectory "
        f"(amortised over a batch of {per})")
    log(f"  speedup {solver_est / max(single, 1e-9):.0f}x like-for-like, "
        f"{solver_est / max(batched, 1e-9):.0f}x batched "
        f"({solver_steps / max(1, n_steps):.0f}x fewer steps)")
    log("  quote the like-for-like number: the solver probe is unbatched too. "
        "Both compare against the explicit dt=0.01 scheme, not a "
        "semi-implicit spectral solver, which would need far fewer steps.")
    return dict(solver_seconds_per_trajectory=float(solver_est),
                rno_seconds_per_trajectory=float(single),
                rno_seconds_per_trajectory_batched=float(batched),
                speedup=float(solver_est / max(single, 1e-9)),
                speedup_batched=float(solver_est / max(batched, 1e-9)),
                batch_size=int(per), solver_steps=solver_steps,
                rno_steps=n_steps)


def cache_summary(cfg, meta):
    """Per-case summaries computed from the rollout cache itself."""
    out = {}
    for case in meta["cases"]:
        d = ms.load_case(cfg["rollout_dir"], case, ms.case_seeds(meta, case),
                         meta, snaps=False)
        out[case] = ms.case_stats(d, float(cfg["threshold"]))
        out[case]["growth_zero_crossing"] = (
            ms.growth_record(d, meta, "zero") if d["seeds"]
            else dict(n_pred=None, n_truth=None))
    return out


def summary_table(summ, meta, cfg):
    """One table across every case of this run."""
    log("")
    log(f"[summary] {cfg['label']}: l = {cfg['l']} (+ boxes), t = "
        f"{cfg['t_start']:g} -> {cfg['t_end']:g}")
    log(f"  {'case':>10} {'alpha':>6} {'dt_max':>7} {'steps':>6} "
        f"{'mean R2':>18} {'final R2':>18} {'R2 >= thr to t':>18} "
        f"{'n never':>7} {'n_pred':>8} {'n div':>6}")
    for case, s in summ.items():
        c = meta["cases"][case]
        e, m, v = s["r2_at_end"], s["r2_trajectory_mean"], s["survival"]
        n_p = s["growth_zero_crossing"]["n_pred"]
        log(f"  {case:>10} {c['alpha']:6.2f} {c['dt_max']:7.1f} "
            f"{s['steps']:6d} {m['mean']:9.4f} +-{m['sd']:.4f} "
            f"{e['mean']:9.4f} +-{e['sd']:.4f} "
            f"{v['mean']:11.0f} +-{v['sd']:<4.0f} "
            f"{v['n_survive_to_end']:3d}/{s['n_seeds']:<3d} "
            f"{ms.fmt(n_p, '8.4f')} {s['n_diverged']:6d}")
    for case, s in summ.items():
        if s["n_truncated_legacy"]:
            log(f"  WARNING {case}: {s['n_truncated_legacy']} legacy seed "
                f"file(s) stop before t_end and carry no diverged_at; the "
                f"case was cut to the shortest (rerun the prediction: they "
                f"are rolled out again)")
    if any(s["n_diverged"] for s in summ.values()):
        log("  n div: seeds whose rollout went non-finite or past "
            f"max |psi| = {cfg['diverge_abs']:g}; R2 columns cover the other "
            "seeds, the survival column counts a diverged seed as failed at "
            "its last good time")
    comp = {k: v for k, v in summ.items()
            if meta["cases"][k]["kind"] == "composition"
            and math.isfinite(v["r2_trajectory_mean"]["mean"])}
    if comp:
        best = max(comp, key=lambda k: comp[k]["r2_trajectory_mean"]["mean"])
        worst = min(comp, key=lambda k: comp[k]["r2_trajectory_mean"]["mean"])
        log(f"  best {best} (mean R2 "
            f"{comp[best]['r2_trajectory_mean']['mean']:.4f}), worst {worst} "
            f"(mean R2 {comp[worst]['r2_trajectory_mean']['mean']:.4f})")


def header(cfg, device, prec, dtype, what):
    log("=" * 78)
    log(f" RNO / Cahn-Hilliard -- {what}{'  [SMOKE]' if cfg['smoke'] else ''}"
        f"   {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 78)
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        log(f"[setup] {p.name}, {p.total_memory / 1e9:.1f} GB, "
            f"torch {torch.__version__}")
    log(f"[setup] device={device} precision={prec} dtype={dtype}")
    log(f"[setup] label {cfg['label']}: END = {cfg['train_end']}, variant "
        f"{cfg['variant']}, width {cfg['want_width']}, modes "
        f"{cfg['want_modes']}, layers {cfg['want_layers']}, context "
        f"{cfg['context']}, seed {cfg['want_seed']}, t = "
        f"{cfg['t_start']:g} -> {cfg['t_end']:g}, save_every "
        f"{cfg['save_every']:g}")
    log(f"[setup] snapshots main {list(cfg['main_snaps'])}, long "
        f"{list(cfg['long_snaps'])}, tune scores at "
        f"{list(cfg['tune_report'])}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Roll the trained RNO out, diagnose it and draw its "
                    "report.")
    ap.add_argument("--end", type=float, default=None,
                    help="training horizon END of the model (default "
                         "CONFIG['train_end'] = 1000)")
    ap.add_argument("--t-end", type=float, default=0.0,
                    help="prediction horizon (default PAIRS[END])")
    ap.add_argument("--variant", choices=VARIANTS, default=None,
                    help="lap (default): the Laplacian-output model in "
                         "Work/models/rno_t<END>_lap/; base: the "
                         "direct-increment head in rno_t<END>/")
    ap.add_argument("--modes", type=int, default=None,
                    help="Fourier modes of the model (default 16): a "
                         "non-default value selects Work/models/rno_t<END>"
                         "_lap_m<M>/ and labels the run ..._m<M>_pred_...")
    ap.add_argument("--width", type=int, default=None,
                    help="hidden channels of the model (default 32); tagged "
                         "w<W> in the label")
    ap.add_argument("--layers", type=int, default=None,
                    help="RNO layers of the model (default 4); tagged d<L> "
                         "in the label")
    ap.add_argument("--context", type=int, default=0,
                    help="ablation: the model was trained with K context "
                         "frames (rno_train --context K); every rollout then "
                         "warms the memory on the K ground-truth frames "
                         "before t_start (default 0: psi(t_start) only)")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed of the model (default 0): a "
                         "non-zero value selects Work/models/rno_t<END>"
                         "_lap_..s<N>/ and labels the run ..s<N>_pred_...; "
                         "the checkpoint must have been trained with it")
    ap.add_argument("--gt-only", action="store_true",
                    help="solve or verify every ground-truth file the run "
                         "needs, then exit (no model is loaded)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run on the rno_train --smoke model; writes "
                         "only under Work/smoke/")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint (default Work/models/rno_t<END>_lap/"
                         "rno_ch_best.pt)")
    return ap.parse_args(argv)


def rollout_only(end=None, t_end=0.0, smoke=False, ckpt="", seeds=0,
                 schedule_from="", variant=None, modes=None, width=None,
                 seed=0, layers=None, context=0):
    """make_submission --rollout: fill one run's rollout cache, no diagnostics.

    Returns the run label.  Logs to Work/predict/<label>/rollout.log.
    """
    a = argparse.Namespace(end=end, t_end=t_end, smoke=smoke, ckpt=ckpt,
                           gt_only=False, variant=variant, modes=modes,
                           width=width, seed=seed, layers=layers,
                           context=context)
    to_project_dir()
    cfg = resolve_config(a)
    moved = archive_superseded(cfg)
    setup_log(cfg["predict_dir"], "rollout")
    for line in moved:
        log(line)
    device = torch.device(str(cfg["device"]))
    prec, dtype = set_precision()
    header(cfg, device, prec, dtype, "ROLLOUT CACHE")
    check_seeds(cfg)
    model, ck, n_par = load_model(cfg, device, dtype)
    t_train, _ = check_horizon(ck, cfg)
    rollout_run(model, ck, n_par, cfg, device, dtype, prec, t_train,
                seeds=seeds, schedule_from=schedule_from)
    return cfg["label"]


def main(argv=None):
    a = parse_args(argv)
    if a.ckpt:
        a.ckpt = os.path.abspath(a.ckpt)
    to_project_dir()
    cfg = resolve_config(a)
    device = torch.device(str(cfg["device"]))
    t_all = time.time()
    if a.gt_only:
        setup_log(predict_dir(f"gt_t{int(round(cfg['t_end'])):05d}",
                              cfg["root"]), "predict")
        header(cfg, device, "-", "-", "GROUND TRUTH ONLY")
        prewarm(cfg, device)
        log(f"[done] ground truth only, {time.time() - t_all:.0f}s")
        return

    moved = archive_superseded(cfg)
    setup_log(cfg["predict_dir"], "predict")
    for line in moved:
        log(line)
    prec, dtype = set_precision()
    header(cfg, device, prec, dtype, "PREDICTION")
    check_seeds(cfg)
    prewarm(cfg, device)

    model, ck, n_par = load_model(cfg, device, dtype)
    t_train, src = check_horizon(ck, cfg)
    log(f"[eval] offs={list(cfg['eval_offs'])}  "
        f"{len(cfg['test_seeds'])} test seeds={list(cfg['test_seeds'])}")
    meta, walls = rollout_run(model, ck, n_par, cfg, device, dtype, prec,
                              t_train)
    summ = cache_summary(cfg, meta)
    summary_table(summ, meta, cfg)

    off, gt_t, gt_f = reference_gt(cfg, device)
    gt = (gt_t, gt_f)
    out: dict[str, Any] = {}
    if cfg["do_transfer"]:
        out["transfer"] = diag_transfer(model, cfg, gt, device, dtype, t_train)
    if cfg["do_predictability"]:
        out["predictability"] = diag_predictability(cfg, gt, device)
    if cfg["do_drift"]:
        out["drift"] = diag_drift(model, cfg, gt, device, dtype, t_train, src)
    if cfg["do_lowk"]:
        out["lowk"] = diag_lowk(model, cfg, off, device, dtype)
    if cfg["do_speedup"]:
        s = meta["schedule"][f"{off:+.1f}"]
        times = make_schedule(cfg["t_start"], cfg["t_end"], s["alpha"],
                              s["dt_min"], s["dt_max"],
                              quantum=float(gt_t[1] - gt_t[0]),
                              must_hit=cfg["snapshots"])
        out["speedup"] = diag_speedup(cfg, device, dtype, model, gt, times)

    tcfg = ck.get("config") or {}
    res = dict(
        label=cfg["label"], variant=cfg["variant"],
        created=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        checkpoint=os.path.abspath(cfg["ckpt"]),
        fingerprint=meta["fingerprint"], arch=ck["arch"], precision=prec,
        n_params=int(n_par), device=str(device), train_end=cfg["train_end"],
        train_t_end=t_train, train_t_end_source=src,
        t_start=cfg["t_start"], t_end=cfg["t_end"], l=cfg["l"],
        threshold=cfg["threshold"], save_every=cfg["save_every"],
        context=int(cfg["context"]), context_dt=float(cfg["context_dt"]),
        solver_dt=cfg["solver_dt"], snapshots=list(cfg["snapshots"]),
        main_snapshots=list(cfg["main_snaps"]),
        long_snapshots=list(cfg["long_snaps"]),
        tune_report=list(cfg["tune_report"]),
        schedule={f"off{k}": v for k, v in meta["schedule"].items()},
        cases={k: {kk: v[kk] for kk in ("kind", "l", "off", "alpha",
                                        "dt_max", "seeds")}
               for k, v in meta["cases"].items()},
        summary=summ, rollout_seconds=walls, gt_devices=meta["gt_devices"],
        rollout_cache=cfg["rollout_dir"],
        training=dict(metrics=ck.get("metrics"),
                      config={k: tcfg[k] for k in TRAIN_KEYS if k in tcfg}),
        diagnostic_trajectory=dict(off=off, seed=int(cfg["tune_seeds"][0])),
        **out)
    rpath = os.path.join(cfg["predict_dir"], "results.json")
    dump_json(rpath, res)
    log("")
    log(f"[results] wrote {rpath}")
    if cfg["do_draw"]:
        log("")
        ms.draw([cfg["label"]], work=cfg["root"], results=cfg["results"])
    log("")
    log(f"[done] total {time.time() - t_all:.1f}s | {n_par:,} parameters | "
        f"report -> {cfg['report_dir']}")


if __name__ == "__main__":
    main()
