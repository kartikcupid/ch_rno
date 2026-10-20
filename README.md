# RNO for the Cahn–Hilliard equation

A Recurrent Neural Operator (RNO) surrogate for the dimensionless Cahn–Hilliard–Cook
equation

$$\partial_t\psi = \nabla^2\left(\psi^3 - \psi - \nabla^2\psi\right),$$

built as the second architecture next to the UNO project
([kartikcupid/uno_ch](https://github.com/kartikcupid/uno_ch)). The RNO is the one in
`neuralop.models.RNO` (Liu-Schiaffini et al. 2023,
[arXiv:2308.08794](https://arxiv.org/abs/2308.08794)), ported line by line and
checked against the library. The data, the test protocol, the seeds, the metrics,
the caches and the figures are the UNO project's, so the two models can be compared
row for row.

Decisions taken at the start (2026-10-20):

| question | decision |
|---|---|
| how to implement the RNO | faithful port of neuralop's RNO plus the UNO fixes: spectral weights on physical wavenumbers, Δt conditioning, Laplacian output, exact mass conservation. A test checks the port against the library |
| what the model sees at the start of a test rollout | ψ(t = 100) only, like UNO and the other architectures. The memory starts empty and builds up along the rollout. A history window is available as an ablation (`--context K`) |
| protocol | the same as UNO's current runs: END = 300 / 500 / 1000, rollouts from t = 100 to 10000 on l = 128 (five compositions) and l = 160 / 192 / 256, 50 test seeds, R² and snapshots. The 0.8 crossing is reported, not used to stop the rollout |

---

## Status (2026-10-20)

- **Code complete, checked on a CPU, not yet trained.** `python -m pytest tests/`
  passes all 15 tests, including the comparison with neuralop's RNO. A `--smoke`
  run goes end to end (dataset → training → stride tuning → rollouts →
  diagnostics → figures); see *Local checks*.
- **No results yet.** The first HPC steps are under *HPC*. Run the benchmark job
  first (`BENCH=1`): the RNO costs more per step than the UNO, and its training
  time on the A100 is not known yet.

---

## 1. The model

### 1.1 The RNO as neuralop defines it

The RNO is a GRU whose matrix–vector products are replaced by Fourier layers. With
x the lifted input at one time step and h the hidden state of a layer:

    z  = sigmoid(f1(x) + f2(h) + b1)            update gate
    r  = sigmoid(f3(x) + f4(h) + b2)            reset gate
    h~ = selu(f5(x) + f6(r * h) + b3)           candidate
    h' = (1 - z) * h + z * h~

Each f is an FNO block: `MLP(K x + W x) + g * x`. K is a spectral convolution,
W a bias-free 1×1 convolution, the MLP is pointwise (expansion 0.5, GELU) and g is a
per-channel soft gate. The b's are scalars. The network has a lifting MLP, `layers`
such RNO layers (each passes its output sequence to the next) and a projection MLP.
These are neuralop's defaults with `positional_embedding=None` and no domain padding.

`RNOCore`, `RNOBlock`, `RNOCell`, `FNOBlock`, `ChannelMLP` and `SpectralConv` in
`rno_train.py` implement exactly this. `tests/test_rno.py::test_port_matches_neuralop`
builds neuralop's `RNO` (v2.0.0, commit a6648ee of 2026-10-17) and the port side by
side and copies the weights across. It checks that both models have the same
parameter count, the same output for a 4-step input sequence, and the same output
and hidden states over two steps with carried states (what `RNO.predict` does). It
runs with and without `rno_skip`, and with even and odd mode counts. All of these
pass. neuralop is needed only for this test, not to train or predict.

### 1.2 What is changed, and why

The GRU equations above are untouched. Two things around them are changed.

**Spectral weights on physical wavenumbers.** neuralop's `SpectralConv` stores its
weights by mode number. Training happens on the 64-box and testing on the 128-box
**at the same dx = 1**, so mode m on the 128-box is half the wavenumber of mode m on
the 64-box. The library RNO would therefore apply each learned filter at the wrong
wavenumber, which is the defect in §1 of the UNO README. `SpectralConv` here keeps
neuralop's weight layout, Hermitian projection and bias on the reference grid
n_ref = 64. On any other grid n it resamples the weight linearly to the modes
j = (n / 64) · m that carry the same physical wavenumbers, with zero outside the
trained band. For n = 2·64, 3·64 and 4·64 the resampled weights at the multiples are
exactly the trained ones, so a tiled state sees exactly the 64-box operator. The
tests check `model(tile(ψ), tile(h)) = tile(model(ψ, h))` for the state, the
increment and every hidden state over three carried steps at 2×, 3× and 4× tiling
(error < 1e-10 in float64). `rno_predict.py` repeats the 2× check on every trained
model. No positional embedding is used: a [0, 1] grid embedding would also change
meaning with the box size.

**A Cahn–Hilliard step around the core (`CHRNO`).**

- The input channels are ψ, ψ³, the 5-point Laplacian of ψ, the mean of ψ, and
  `cond_ch = 8` constant channels holding a learned embedding of the step Δt. The
  model is conditioned on Δt, so one model serves every stride of the rollout
  schedule.
- The projection output M is scaled by gain(Δt) = exp(a + b log Δt). With
  `--variant lap` (the default) the increment is the periodic 5-point Laplacian of
  M. This is the form of every solver update over any Δt, it suppresses the modes
  below the 64-box fundamental by k², and it conserves mass exactly (UNO §6,
  *Laplacian output*). `--variant base` uses M directly. The spatial mean of the
  increment is always subtracted.
- The prediction is residual: ψ(t + Δt) = ψ(t) + increment. The last projection
  layer starts at zero, so the untrained model is the identity map.
- The hidden states of all layers are the **memory**. They are returned by every
  call and passed back on the next one, as `RNO.predict` does. A test rollout starts
  from ψ(100) with an empty memory (zeros plus the learned bias) and carries the
  memory to t = 10000. A seed that diverges is frozen with its memory, and the
  others continue (`mask_state`).

### 1.3 Size

Defaults: width (hidden channels) 32, modes 16, 4 RNO layers, **7.14M parameters**.
Modes 16 keeps |k| ≤ 2π·8/64 = 0.785 at dx = 1. That is the first-level cutoff of
UNO's w24 m8, which matched the 17.9M UNO default (UNO README, *Model size*).
Local, high-k structure still reaches the output through the 1×1 paths, the ψ³ and
∇²ψ input channels and the Laplacian head. Other sizes:

| width | modes | layers | parameters | tag |
|---|---|---|---|---|
| 32 | 24 | 4 | 15.39M | `m24` |
| 32 | 16 | 4 | 7.14M | (default) |
| 32 | 16 | 3 | 5.35M | `d3` |
| 24 | 16 | 4 | 4.02M | `w24` |
| 16 | 16 | 4 | 1.79M | `w16` |
| 24 | 12 | 3 | 1.77M | `w24m12d3` |

Every one of the 24 FNO blocks runs at full resolution (there is no U-net
coarsening). One step costs about 280 ms at l = 128 on two CPU threads. On the A100
the cost is unmeasured; `BENCH=1` measures it.

---

## 2. Training (`rno_train.py`)

Copied from the UNO training wherever the model does not matter: the same dataset
(`ch_gpu.py`, ch.py's 20 + 5 trajectories at l = 64), the D4 and ψ → −ψ augmentation,
the input noise, the loss (per-sample MSE/Var + 0.25 increment-normalised + 0.25
gradient term), EMA weights, AdamW with a cosine schedule and a learning-rate drop
per unroll stage, and the **survival-based checkpoint selection** on 10 validation
seeds rolled out at l = 128 from t = 100 to 10000. What is new for a recurrent model:

- **Chains start with an empty memory**, as a test rollout does, and the memory is
  carried along the chain.
- **Unroll curriculum 1 → 2 → 4 → 8** (from epochs 1, 5, 12, 28 of 60). The gradient
  flows through the last `grad_steps = 4` transitions, and the earlier steps of an
  8-step chain run without gradient. Their job is to give the memory some history
  before the loss sees it.
- **Schedule-shaped chains.** Half of every batch uses a constant stride
  (log-uniform in [1, 50]). The other half uses a stride that grows like α·t
  (α log-uniform in [0.05, 0.7], capped at 50), as the rollout schedule does. This
  way the memory also sees strides that change from step to step.
- The learning rate is 2e-3 (UNO: 3e-3). This is a first guess, not tuned.
- The float32 memmaps of a dataset are stored once per END in
  `Work/data/memmap_t<END>/` and shared by every model (UNO kept one copy per
  model). Two jobs that start together each write their own partial file and
  install identical bytes.

Checkpoints are `Work/models/rno_t<END>[_lap][_<tag>]/rno_ch_best.pt`. Training
resumes from `rno_ch_state.pt` when a job hits its wall clock.

### Context ablation (`--context K`)

`--context K` gives the memory K ground-truth frames, `context_dt = 4` apart, before
the start. In training these come before every chain; in validation, tuning and
every test rollout they are the K frames before t = 100. Each frame is fed with
teacher forcing, and only the memory is kept. This is the history-window ablation;
the default, `K = 0`, sees ψ(100) only. The frames before t = 100 are solved from
t = 0 like every ground truth and stored as small extra files
(`gt_..._t<100-4K>-100_ev4.npz`). The tag `c<K>` keeps the run apart
(`rno_t1000_lap_c5_pred_t10000`), and `rno_predict.py` refuses a checkpoint trained
with another K.

---

## 3. Prediction and figures (`rno_predict.py`, `make_submission.py`)

These are UNO's `uno_predict.py` and `make_submission.py` with the model calls
carrying the memory, the label prefix `rno_` and the model name RNO. Everything
else is unchanged:

- the stride schedule (Δt ∝ t, capped) is tuned per quench on the 10 validation
  seeds, scored at t = 2500, 5000, 7500 and 10000;
- 50 test seeds (all ≥ 20000) per case: ψ₀ = −0.4, −0.2, 0, +0.2, +0.4 at l = 128,
  and the critical quench at l = 160, 192 and 256 on the critical schedule;
- per-seed divergence handling (non-finite or max |ψ| > 10);
- the rollout cache `Work/rollout_cache/<label>/<case>/seed<N>.npz` and the same
  figures in `Results/<label>/`: snapshots (best seed), long snapshots, R² sweep,
  phase-ordering kinetics, domain growth, P(ψ), composition, `report.txt` and
  `manifest.json`;
- the diagnostics: the tiling test (now over carried steps and hidden states),
  one-step R² from an empty memory, intrinsic predictability, the coarsening drift,
  the low-k increment gain and the speedup against the solver.

Labels: `rno_t<END>[_lap][_<tag>]_pred_t<T>`. The tag is built from `w<W>`
(width), `m<M>` (modes), `d<L>` (layers), `c<K>` (context) and `s<N>` (training
seed), in that order, for every non-default value. Examples: `rno_t1000_lap_pred_t10000`,
`rno_t1000_lap_w24m12d3_pred_t10000`, `rno_t1000_lap_s1_pred_t10000`. The `rno_`
prefix keeps them apart from UNO's `train_t...` folders, so results of both can be
copied into one place.

---

## 4. Files

| file | role |
|---|---|
| `rno_train.py` | model (`SpectralConv` … `RNOCore`, `CHRNO`), training loop, ground-truth solver and cache (UNO's format), path helpers. `--end`, `--variant lap\|base`, `--width`, `--modes`, `--layers`, `--context`, `--seed`, `--npz`, `--smoke`, `--bench` |
| `rno_predict.py` | ground-truth prewarm, stride tuning, 50-seed rollouts, diagnostics, `results.json`, drawing. Same options plus `--t-end`, `--gt-only`, `--ckpt` |
| `make_submission.py` | `Results/` from the rollout caches, no torch (`--overview`, `--run LABEL`, `--rollout`) |
| `ch_gpu.py` | the training dataset, ch.py's dynamics frame for frame (UNO's file) |
| `tests/test_rno.py` | the neuralop comparison, tiling, mass, per-sample independence, sampler and label tests |
| `check_code.py` | compile, ≤ 88 columns, Python 3.11 f-string rules, no comments in `rno_train.py` / `rno_predict.py` |
| `rt.sh` `rp.sh` `rtp.sh` | Slurm: train, predict + draw, train + predict + draw. `END`, `VARIANT`, `WIDTH`, `MODES`, `LAYERS`, `SEED` (and `BENCH=1` for `rt.sh`) via `--export` |
| `gt.sh` `ch_gpu.sh` `sp.sh` | Slurm: ground truth only, dataset, full redraw |
| `hpc_setup.sh` | one-time links to the UNO project's venv, ground truth, datasets and memmaps |
| `hpc_storage.py`, `hpc_storage.sh` | storage and quota report (UNO's) |

Every setting is in a `CONFIG` dict at the top of `rno_train.py` and
`rno_predict.py`.

---

## 5. HPC (IIT Jodhpur, `dgx_fat`, A100 40 GB)

### 5.1 One-time setup

Home was at 184 of 200 GB on 2026-10-15, so run this project from scratch and link
everything large from the UNO project instead of copying it:

```bash
cd /scratch/data/m25ph1006
git clone https://github.com/kartikcupid/ch_rno.git && cd ch_rno
# (before the branch is merged: git checkout claude/blissful-davinci-laq0ah)
bash hpc_setup.sh                 # UNO_DIR=/iitjhome/m25ph1006/chc_uno_v1 by default
source env/bin/activate && python check_code.py
```

`hpc_setup.sh` links `env/` (the UNO venv: Python 3.11.7, torch 2.13; this project
needs nothing else), `Work/gt_cache/` (~112 GB of ground truth to t = 10000, the
same files and format), the datasets `Work/data/test_t<END>.npz` that still exist,
and the UNO float32 memmaps as `Work/data/memmap_t<END>/`. Nothing is copied or
deleted. Check the scratch purge policy before keeping anything expensive there
only. Set `UNO_DIR` if the UNO copy lives elsewhere (e.g. `chc_uno_1` on scratch).

The advisor's brief asks for Python 3.12, and the UNO env is 3.11.7. To follow the
brief, make a fresh venv instead of the link (`module avail python`, load a 3.12
module, `python -m venv env`, `pip install torch numpy matplotlib`). The code runs
on 3.9 to 3.14.

### 5.2 Order of jobs

```bash
sbatch --export=ALL,BENCH=1 rt.sh                 # (1) ~5 min: ms/step and peak memory
sbatch --export=ALL,END=1000 rtp.sh               # (2) the main run: train, predict, draw
sbatch --export=ALL,END=300 rtp.sh                # (3) the shorter training horizons
sbatch --export=ALL,END=500 rtp.sh
sbatch --export=ALL,END=1000,VARIANT=base rtp.sh  # (4) head ablation
sbatch --export=ALL,END=1000,SEED=1 rtp.sh        # (5) training-seed noise (UNO: SD 0.04-0.08)
```

- Ground truth: with the linked cache every file exists. Without it, run
  `g=$(sbatch --parsable gt.sh)` first and chain the others with
  `--dependency=afterok:$g`.
- Dataset: only needed when neither the `.npz` nor the memmaps were linked
  (`sbatch --export=ALL,END=1000 ch_gpu.sh`).
- Wall clock: `rt.sh` and `rtp.sh` ask for 12 h. Training resumes from its state
  file and prediction from its rollout cache, so a job that times out can simply be
  resubmitted. If the benchmark shows a step much above UNO's 136–256 ms at unroll
  4, lower `batch` or the size before the main run.
- Nodes are assigned at random (gpu1 is about 3× faster than gpu2 for UNO). Do not
  pin jobs, and compare times only within one node.

### 5.3 Copy results back

```bash
for r in rno_t1000_lap_pred_t10000; do
  rsync -a hpc:<project>/Results/$r Results/
  rsync -a hpc:<project>/Work/rollout_cache/$r Work/rollout_cache/
  rsync -a hpc:<project>/Work/predict/$r Work/predict/
done
python3 make_submission.py --overview
```

---

## 6. Risks and open questions

- **Memory length.** Training chains are at most 8 steps; a test rollout is about
  200–300 steps. The GRU gating keeps the hidden state bounded, but nothing in
  training shows it a memory that is hundreds of steps old. Validation rollouts to
  t = 10000 pick the checkpoint, so a drifting memory shows up there. If it does,
  the next steps are longer no-gradient prefixes (an unroll-16 stage) or carrying
  memories across training iterations.
- **Cost.** 24 full-resolution FNO blocks per step against UNO's U-net. Training
  time on the A100 is unknown until the benchmark runs.
- **Untuned.** The learning rate, the size, the schedule-chain fraction and
  `cond_ch` are first guesses. UNO's experience (cut modes before width; seed noise
  of 0.04–0.08 in pooled R²(10000)) applies to any size ranking here too.
- **Python version.** See 5.1.
- **Shared ground truth.** `Work/gt_cache` is a link into the UNO project, so a
  ground-truth file this project ever needs and UNO lacks (for example the
  `--context` files) is written there.
- neuralop's `rno_skip=True` stores input + output, not the hidden state, as the
  carried state of a lower layer. The port keeps that behaviour; the default is
  `rno_skip=False`, where the question does not arise.

---

## 7. Local checks

```bash
python3 check_code.py                       # static checks
python3 -m pytest tests/ -q                 # 15 tests (the neuralop one needs neuraloperator)
python3 ch_gpu.py --end 300                 # dataset (GPU, or --cpu: a few minutes)
python3 rno_train.py --smoke --end 300      # tiny model, writes only under Work/smoke/
python3 rno_predict.py --smoke --end 300    # tune, roll out, diagnose, draw
```
