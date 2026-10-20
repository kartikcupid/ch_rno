"""Checks of the RNO port and of the properties the project relies on.

    python -m pytest tests/ -q

The neuralop comparison needs `neuraloperator` installed (it is skipped
otherwise); every other test needs torch and numpy only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import make_submission as ms  # noqa: E402
import rno_train as rt  # noqa: E402

torch.set_num_threads(2)


def _conv1x1(src):
    """neuralop kernel-1 Conv1d weight (out, in, 1) -> Conv2d (out, in, 1, 1)."""
    return src.detach().reshape(src.shape[0], src.shape[1], 1, 1)


def _copy_mlp(dst, src):
    for d, s in zip(dst.fcs, src.fcs):
        d.weight.data.copy_(_conv1x1(s.weight))
        d.bias.data.copy_(s.bias.detach())


def _copy_fno_block(dst, src):
    w = src.convs[0].weight
    w = w.to_tensor() if hasattr(w, "to_tensor") else w
    dst.conv.weight.data.copy_(torch.view_as_real(w.detach().cfloat()))
    dst.conv.bias.data.copy_(src.convs[0].bias.detach())
    dst.skip.weight.data.copy_(_conv1x1(src.fno_skips[0].conv.weight))
    _copy_mlp(dst.mlp, src.channel_mlp[0])
    dst.gate.data.copy_(src.channel_mlp_skips[0].weight.detach())


def copy_from_neuralop(core, ref):
    """Load the weights of a neuralop RNO into an RNOCore of the same size."""
    _copy_mlp(core.lifting, ref.lifting)
    _copy_mlp(core.projection, ref.projection)
    for mine, theirs in zip(core.layers, ref.layers):
        mine.bias_h.data.copy_(theirs.bias_h.detach())
        for g in range(3):
            mine.cell.biases[g].data.copy_(theirs.cell.biases[g].detach())
            _copy_fno_block(mine.cell.input_gates[g],
                            theirs.cell.input_gates[g])
            _copy_fno_block(mine.cell.hidden_gates[g],
                            theirs.cell.hidden_gates[g])


@pytest.mark.parametrize("rno_skip", [False, True])
@pytest.mark.parametrize("modes", [8, 7])
def test_port_matches_neuralop(rno_skip, modes):
    nm = pytest.importorskip("neuralop.models")
    torch.manual_seed(0)
    n, width, layers, cin = 24, 6, 2, 5
    ref = nm.RNO(n_modes=(modes, modes), in_channels=cin, out_channels=1,
                 hidden_channels=width, n_layers=layers, rno_skip=rno_skip,
                 positional_embedding=None)
    core = rt.RNOCore(cin, 1, width, modes, layers, n_ref=n,
                      rno_skip=rno_skip)
    n_ref = sum(p.numel() * (2 if p.is_complex() else 1)
                for p in ref.parameters())
    n_mine = sum(p.numel() for p in core.parameters())
    assert n_mine == n_ref
    copy_from_neuralop(core, ref)
    ref.eval()
    core.eval()
    x = torch.randn(3, 4, cin, n, n)
    with torch.no_grad():
        want = ref(x)
        got, _ = core(x)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4), \
            float((got - want).abs().max())
        x1 = torch.randn(3, 1, cin, n, n)
        p_ref, s_ref = ref(x1, return_hidden_states=True,
                           keep_states_padded=True)
        p_got, s_got = core(x1)
        x2 = torch.randn(3, 1, cin, n, n)
        p_ref, s_ref = ref(x2, s_ref, return_hidden_states=True,
                           keep_states_padded=True)
        p_got, s_got = core(x2, s_got)
        assert torch.allclose(p_got, p_ref, atol=1e-5, rtol=1e-4)
        for a, b in zip(s_got, s_ref):
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def _model(dtype=torch.float64, std=0.3, **kw):
    """A small CHRNO whose zero-initialised last layer is randomised, so
    the increment is not trivially zero."""
    torch.manual_seed(1)
    arch = dict(width=6, modes=8, layers=2, n_ref=32, cond_dim=8, cond_ch=3)
    arch.update(kw)
    net = rt.CHRNO(**arch).to(dtype=dtype)
    torch.nn.init.normal_(net.core.projection.fcs[-1].weight, std=std)
    return net.eval()


@pytest.mark.parametrize("lap", [True, False])
def test_tiling_is_exact_with_memory(lap):
    net = _model(laplacian_output=lap)
    err = rt.tile_check(net, torch.device("cpu"), torch.float64, n=32,
                        steps=3)
    assert err < 1e-10, err


@pytest.mark.parametrize("k", [3, 4])
def test_larger_tilings_are_exact(k):
    net = _model()
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, 1, 32, 32, generator=g, dtype=torch.float64) * 0.6
    q = q - q.mean()
    dt = torch.tensor([7.0], dtype=torch.float64)
    with torch.no_grad():
        a, ha = net(q, dt)
        a, ha = net(a, dt, ha)
        b, hb = net(q.repeat(1, 1, k, k), dt)
        b, hb = net(b, dt, hb)
    assert torch.allclose(a.repeat(1, 1, k, k), b, atol=1e-10)
    for x, y in zip(ha, hb):
        assert torch.allclose(x.repeat(1, 1, k, k), y, atol=1e-10)


def test_untiled_box_change_is_smooth():
    """At s = 2.5 (l = 80 from 32) the weights are interpolated, not tiled:
    the model must still run and keep the mean."""
    net = _model()
    q = torch.randn(2, 1, 80, 80, dtype=torch.float64) * 0.5
    with torch.no_grad():
        p, h = net(q, torch.tensor([5.0, 9.0], dtype=torch.float64))
    assert p.shape == q.shape and len(h) == 2
    assert torch.allclose(p.mean(dim=(-2, -1)), q.mean(dim=(-2, -1)),
                          atol=1e-12)


def test_identity_at_init_and_mass():
    net = rt.CHRNO(width=6, modes=8, layers=2, n_ref=32, cond_dim=8,
                   cond_ch=3).double().eval()
    q = torch.randn(2, 1, 32, 32, dtype=torch.float64)
    dt = torch.tensor([3.0, 30.0], dtype=torch.float64)
    with torch.no_grad():
        p, _ = net(q, dt)
    assert torch.equal(p, q)
    net = _model(std=0.05)
    with torch.no_grad():
        p, h = q, None
        for _ in range(5):
            p, h = net(p, dt, h)
    assert float((p - q).abs().max()) > 1e-3
    assert torch.allclose(p.mean(dim=(-2, -1)), q.mean(dim=(-2, -1)),
                          atol=1e-12)


def test_samples_are_independent():
    """Divergence masking relies on the model acting per sample."""
    net = _model()
    q = torch.randn(3, 1, 32, 32, dtype=torch.float64)
    dt = torch.tensor([2.0, 8.0, 20.0], dtype=torch.float64)
    with torch.no_grad():
        pb, hb = net(q, dt)
        pb, hb = net(pb, dt, hb)
        for k in range(3):
            p1, h1 = net(q[k:k + 1], dt[k:k + 1])
            p1, h1 = net(p1, dt[k:k + 1], h1)
            assert torch.allclose(p1, pb[k:k + 1], atol=1e-12)
            for a, b in zip(h1, hb):
                assert torch.allclose(a, b[k:k + 1], atol=1e-12)


def test_mask_state():
    new = [torch.ones(3, 2, 4, 4)]
    old = [torch.zeros(3, 2, 4, 4)]
    alive = torch.tensor([True, False, True])
    got = rt.mask_state(alive, new, old)[0]
    assert got[0].eq(1).all() and got[1].eq(0).all() and got[2].eq(1).all()
    assert rt.mask_state(alive, new, None) is new


def test_spectral_conv_runtime_weight_hits_nodes():
    conv = rt.SpectralConv(2, 3, 8, n_ref=32).double()
    with torch.no_grad():
        w64, idx = conv._runtime_weight(64, torch.device("cpu"),
                                        torch.complex128)
    base = torch.view_as_complex(conv.weight.detach())
    rows = (idx.numpy() + 32) % 64 - 32
    assert rows.min() == -8 and rows.max() == 6
    even = np.nonzero(rows % 2 == 0)[0]
    assert torch.allclose(w64[:, :, even][:, :, :, ::2], base, atol=1e-12)


def test_sampler_chains_fit(tmp_path):
    n_traj, n_frames = 5, 1000
    arr = np.lib.format.open_memmap(str(tmp_path / "x.npy"), mode="w+",
                                    dtype=np.float32,
                                    shape=(n_traj, n_frames, 8, 8))
    arr[:] = np.arange(n_frames, dtype=np.float32)[None, :, None, None]
    arr.flush()
    cfg = dict(rt.CONFIG, train_off=None, data_on_gpu=False)
    s = rt.PairSampler(str(tmp_path / "x.npy"), cfg, torch.device("cpu"),
                       torch.float32, 0)
    for n_steps in (1, 4, 8):
        chain, dt, warm = s.sample(64, n_steps)
        assert warm == []
        frames = torch.stack([c[:, 0, 0, 0] for c in chain], dim=1)
        assert frames.max() <= n_frames - 1
        assert frames.min() >= s.f_min
        steps = torch.diff(frames, dim=1)
        assert torch.allclose(steps * s.fdt, dt, atol=1e-4)
        assert (dt >= cfg["dt_min"] - 1e-6).all()
        assert (dt <= cfg["dt_max"] + 1e-6).all()
    chain, dt, _ = s.sample(8, 60)
    assert dt.shape == (8, 60) and float(chain[-1].max()) <= n_frames - 1
    s.sched_frac = 1.0
    _, dt, _ = s.sample(256, 6)
    assert (torch.diff(dt, dim=1) >= -1e-6).float().mean() > 0.9
    cfg.update(context=3, context_dt=2.0)
    s = rt.PairSampler(str(tmp_path / "x.npy"), cfg, torch.device("cpu"),
                       torch.float32, 0)
    chain, dt, warm = s.sample(32, 4)
    f0 = chain[0][:, 0, 0, 0]
    assert len(warm) == 3
    for k, w in enumerate(warm):
        assert torch.equal(w[:, 0, 0, 0], f0 - (3 - k) * 20)
    assert (torch.stack([w[:, 0, 0, 0] for w in warm]) >= 0).all()


def test_labels_round_trip():
    for kw, tag in [(dict(), ""), (dict(width=24), "w24"),
                    (dict(modes=12, layers=3), "m12d3"),
                    (dict(width=24, modes=12, layers=3, seed=2),
                     "w24m12d3s2")]:
        assert rt.run_tag(**kw) == tag
        label = rt.run_label(1000, 10000, "lap", tag)
        assert ms.parse_label(label) == (1000, "lap", 10000)
        assert ms.label_tag(label) == tag
        w, m, d = ms.run_size(label)
        assert (w, m, d) == (kw.get("width", 32), kw.get("modes", 16),
                             kw.get("layers", 4))
        assert ms.label_seed(label) == kw.get("seed", 0)
    assert rt.run_label(300, 10000, "base") == "rno_t0300_pred_t10000"
    assert ms.parse_label("train_t1000_lap_pred_t10000") is None
