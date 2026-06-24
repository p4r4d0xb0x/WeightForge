"""Tests for grow_aster_1b_to_3b.py -- growing aster-1b into an aster-3b init ckpt.

Most tests are fast and dependency-light: they exercise the spec derivation, the
per-tensor growth transforms, fresh-init for new blocks, and a save/load round
trip on a *tiny synthetic* checkpoint (no real weights, no torch downloads).

One test is GATED on the real aster-1b checkpoint being present on disk; it is
skipped (not failed) when the sibling SLM_FROM_BEGIN repo or its checkpoint is
unavailable, so this suite stays runnable in isolation.

Run:   python -m pytest test_grow_aster_1b_to_3b.py -q
"""

from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

import grow_aster_1b_to_3b as fa


# --------------------------------------------------------------------------- #
# spec derivation (mirrors crates/slm-types/src/model.rs rules)
# --------------------------------------------------------------------------- #
def _spec_3b() -> fa.ModelSpec:
    return fa.ModelSpec(
        vocab_size=48000,
        d_model=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_mult=2.75,
        weight_tying=True,
    )


def test_derived_dims_match_3b_config():
    s = _spec_3b()
    assert s.head_dim == 128  # 3072 / 24
    assert s.kv_dim == 1024  # 8 * 128
    assert s.ffn_inner == 8448  # round(3072 * 2.75)


def test_tensor_shapes_full_key_set_tied():
    s = _spec_3b()
    shapes = s.tensor_shapes()
    # 1 embed + 28 blocks * 9 + 1 final_norm, no head (tied).
    assert len(shapes) == 1 + 28 * 9 + 1 == 254
    assert "head.weight" not in shapes
    assert shapes["embed.weight"] == (48000, 3072)
    assert shapes["blocks.0.q.weight"] == (3072, 3072)
    assert shapes["blocks.0.k.weight"] == (1024, 3072)
    assert shapes["blocks.0.v.weight"] == (1024, 3072)
    assert shapes["blocks.27.ffn_gate.weight"] == (8448, 3072)
    assert shapes["blocks.27.ffn_down.weight"] == (3072, 8448)
    assert shapes["final_norm.gamma"] == (3072,)


def test_tensor_shapes_untied_adds_head():
    s = fa.ModelSpec(48000, 3072, 28, 24, 8, 2.75, weight_tying=False)
    shapes = s.tensor_shapes()
    assert shapes["head.weight"] == (48000, 3072)
    assert len(shapes) == 255


def test_load_model_spec_parses_real_3b_toml():
    cfg = fa.DEFAULT_CONFIG
    if not cfg.exists():
        pytest.skip(f"3b config not present: {cfg}")
    s = fa.load_model_spec(cfg)
    assert (s.d_model, s.n_layers, s.n_heads, s.n_kv_heads) == (3072, 28, 24, 8)
    assert s.ffn_inner == 8448 and s.weight_tying is True


# --------------------------------------------------------------------------- #
# fresh-init (matches SLM init.rs distribution, not its exact RNG)
# --------------------------------------------------------------------------- #
def test_fresh_init_gamma_is_ones():
    gen = torch.Generator().manual_seed(0)
    g = fa._init_tensor("blocks.26.attn_norm.gamma", (3072,), 28, gen)
    assert torch.equal(g, torch.ones(3072))


def test_fresh_init_default_vs_residual_std():
    gen = torch.Generator().manual_seed(0)
    # large tensors so the empirical std is stable.
    q = fa._init_tensor("blocks.26.q.weight", (2048, 2048), 28, gen)
    o = fa._init_tensor("blocks.26.o.weight", (2048, 2048), 28, gen)
    down = fa._init_tensor("blocks.26.ffn_down.weight", (2048, 2048), 28, gen)

    # Truncating N(0,1) to |z| <= 2 lowers the empirical std below the nominal
    # 1.0 (the framework's SmokeRng truncation has the same property), so the
    # observed std is ~0.88 of the requested scale. Assert the *ratio* between
    # default and residual scales -- that invariant is what matters -- plus a
    # loose bound on the absolute level.
    trunc_factor = q.std().item() / 0.02  # empirical shrink from truncation
    assert 0.80 < trunc_factor < 1.0
    expected_resid = 0.02 / ((2.0 * 28) ** 0.5)
    # residual std should track the default std by the same scale ratio.
    assert abs(o.std().item() / q.std().item() - expected_resid / 0.02) < 0.05
    assert abs(down.std().item() / q.std().item() - expected_resid / 0.02) < 0.05
    # truncation: |z| <= 2 std  =>  |x| <= 2 * std
    assert q.abs().max().item() <= 2.0 * 0.02 + 1e-6


# --------------------------------------------------------------------------- #
# growth transfer on a *tiny synthetic* 1b-shaped state dict (no real weights)
# --------------------------------------------------------------------------- #
def _tiny_spec(d_model: int, n_layers: int) -> fa.ModelSpec:
    # n_heads chosen so head_dim divides cleanly; kv heads = 2.
    n_heads = d_model // 8  # head_dim = 8
    return fa.ModelSpec(
        vocab_size=40,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=2,
        ffn_mult=2.0,
        weight_tying=True,
    )


def _synthetic_src_sd(spec: fa.ModelSpec) -> dict[str, torch.Tensor]:
    """A ``v2.``-prefixed state dict matching ``spec`` (the 'teacher')."""
    return {
        fa.CKPT_PREFIX + name: torch.randn(shape)
        for name, shape in spec.tensor_shapes().items()
    }


def test_grow_produces_exact_target_shapes_and_full_keyset():
    small = _tiny_spec(d_model=16, n_layers=2)
    big = _tiny_spec(d_model=32, n_layers=4)
    src = _synthetic_src_sd(small)

    logical, results = fa.grow_checkpoint(src, big)

    target = big.tensor_shapes()
    assert set(logical.keys()) == set(target.keys())
    for name, shape in target.items():
        assert tuple(logical[name].shape) == shape
        assert logical[name].dtype == torch.float32


def test_grow_new_blocks_are_fresh_init():
    small = _tiny_spec(d_model=16, n_layers=2)
    big = _tiny_spec(d_model=32, n_layers=4)  # blocks 2,3 are new
    src = _synthetic_src_sd(small)

    _, results = fa.grow_checkpoint(src, big)
    by_name = {r.name: r for r in results}

    # block 0/1 grew from the teacher; block 2/3 are fresh.
    assert by_name["blocks.0.q.weight"].kind in ("SvdProject", "Copy", "CropPad")
    for i in (2, 3):
        assert by_name[f"blocks.{i}.q.weight"].kind == "FreshInit"
        assert by_name[f"blocks.{i}.q.weight"].src_shape is None
    rep = fa.grow_report(results)
    assert rep["fresh_init"] == 2 * 9  # 2 new blocks * 9 tensors


def test_grow_1d_norm_zero_pads_growth():
    small = _tiny_spec(d_model=16, n_layers=2)
    big = _tiny_spec(d_model=32, n_layers=2)
    src = _synthetic_src_sd(small)
    src_gamma = src[fa.CKPT_PREFIX + "blocks.0.attn_norm.gamma"].clone()

    logical, _ = fa.grow_checkpoint(src, big)
    grown = logical["blocks.0.attn_norm.gamma"]

    assert grown.shape == (32,)
    # original 16 values preserved (CropPad copies the overlap), tail zero-padded.
    assert torch.allclose(grown[:16], src_gamma)
    assert torch.count_nonzero(grown[16:]) == 0


def test_grow_report_counts():
    small = _tiny_spec(d_model=16, n_layers=2)
    big = _tiny_spec(d_model=32, n_layers=3)
    src = _synthetic_src_sd(small)
    _, results = fa.grow_checkpoint(src, big)
    rep = fa.grow_report(results)
    assert rep["target_tensors"] == 1 + 3 * 9 + 1
    assert rep["grown_from_1b"] + rep["fresh_init"] == rep["target_tensors"]
    assert 0.0 < rep["coverage"] <= 1.0


# --------------------------------------------------------------------------- #
# checkpoint writer contract (v2. prefix, F32, state.json, no head/optimizer)
# --------------------------------------------------------------------------- #
def _read_safetensors_header(path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def test_write_checkpoint_prefix_dtype_and_state(tmp_path):
    big = _tiny_spec(d_model=32, n_layers=3)
    logical = {name: torch.zeros(shape) for name, shape in big.tensor_shapes().items()}

    out = tmp_path / "aster-3b-init"
    # model-only mode (optimizer_sidecars=False): the legacy params+state output.
    params_path = fa.write_checkpoint(out, logical, seed=123, optimizer_sidecars=False)

    assert params_path.exists()
    hdr = _read_safetensors_header(params_path)
    keys = [k for k in hdr if k != "__metadata__"]
    # every key carries the v2. prefix and is F32.
    assert all(k.startswith("v2.") for k in keys)
    assert all(hdr[k]["dtype"] == "F32" for k in keys)
    assert "v2.head.weight" not in keys  # tied -> no head
    assert len(keys) == 1 + 3 * 9 + 1

    state = json.loads((out / "state.json").read_text())
    assert state == {"schema_version": 1, "global_step": 0, "lr": 0.0, "seed": 123}

    # with optimizer_sidecars=False, no optimizer/muon sidecars are written.
    assert not (out / "optimizer.safetensors").exists()
    assert not (out / "optimizer_state.json").exists()
    assert not (out / "muon_momentum.safetensors").exists()


# --------------------------------------------------------------------------- #
# optimizer-partition predicate (port of SLM route()/MUON_SUFFIXES)
# --------------------------------------------------------------------------- #
def test_route_matches_slm_partition_rule():
    # Muon: rank-2, blocks.*, matrix suffix.
    for suffix in ("q", "k", "v", "o", "ffn_gate", "ffn_up", "ffn_down"):
        assert fa.routes_to_muon(f"blocks.5.{suffix}.weight", (32, 32)) is True
    # AdamW (NOT muon): norms, embed, final_norm.
    assert fa.routes_to_muon("blocks.0.attn_norm.gamma", (32,)) is False
    assert fa.routes_to_muon("blocks.0.ffn_norm.gamma", (32,)) is False
    assert fa.routes_to_muon("embed.weight", (40, 32)) is False  # not blocks.*
    assert fa.routes_to_muon("final_norm.gamma", (32,)) is False
    # rank guard: a matrix name that is somehow rank-1 stays AdamW.
    assert fa.routes_to_muon("blocks.0.q.weight", (32,)) is False


def test_adamw_param_names_are_norms_embed_finalnorm():
    spec = _tiny_spec(d_model=32, n_layers=3)
    names = fa.adamw_param_names(spec.tensor_shapes())
    # 3 attn_norm + 3 ffn_norm + embed + final_norm = 8.
    assert len(names) == 3 + 3 + 1 + 1
    assert "embed.weight" in names
    assert "final_norm.gamma" in names
    assert all(
        n.endswith(".gamma") or n == "embed.weight" for n in names
    )
    # no matrix weight leaked into the AdamW set.
    assert not any(n.endswith("q.weight") or n.endswith("ffn_down.weight") for n in names)


# --------------------------------------------------------------------------- #
# optimizer sidecar writer (zero AdamW moments + optimizer_state.json)
# --------------------------------------------------------------------------- #
def test_write_optimizer_sidecars_zero_moments_and_pairs(tmp_path):
    big = _tiny_spec(d_model=32, n_layers=3)
    logical = {name: torch.randn(shape) for name, shape in big.tensor_shapes().items()}
    out = tmp_path / "ckpt"
    out.mkdir()

    opt_path, state_path, n_adamw = fa.write_optimizer_sidecars(out, logical)
    assert n_adamw == 3 + 3 + 1 + 1  # 8 AdamW params

    hdr = _read_safetensors_header(opt_path)
    keys = [k for k in hdr if k != "__metadata__"]
    # one m + one v per AdamW param, all F32, all adamw.* prefixed.
    assert len(keys) == 2 * n_adamw
    assert all(k.startswith("adamw.") for k in keys)
    assert all(k.endswith(".m") or k.endswith(".v") for k in keys)
    assert all(hdr[k]["dtype"] == "F32" for k in keys)

    # actual tensors are all zero, and shapes match the params.
    from safetensors.torch import load_file as _load
    moments = _load(str(opt_path))
    for k, t in moments.items():
        assert torch.count_nonzero(t) == 0
    assert moments["adamw.embed.weight.m"].shape == (40, 32)

    # no muon-routed param leaked into the AdamW sidecar.
    assert not any("q.weight" in k for k in keys)

    sidecar = json.loads(state_path.read_text())
    assert sidecar["schema_version"] == 1
    assert sidecar["beta1"] == 0.9 and sidecar["beta2"] == 0.95
    assert sidecar["eps"] == 1e-8 and sidecar["weight_decay"] == 0.1
    # steps: [name, 0] for each AdamW param.
    assert len(sidecar["steps"]) == n_adamw
    assert all(step == 0 for _, step in sidecar["steps"])
    step_names = {name for name, _ in sidecar["steps"]}
    assert step_names == set(fa.adamw_param_names(big.tensor_shapes()))


def test_write_checkpoint_default_emits_resumable_sidecars(tmp_path):
    big = _tiny_spec(d_model=32, n_layers=3)
    logical = {name: torch.zeros(shape) for name, shape in big.tensor_shapes().items()}
    out = tmp_path / "resumable"

    fa.write_checkpoint(out, logical, seed=7)  # default optimizer_sidecars=True

    # the 4-file resume contract is satisfied; muon sidecar omitted (optional).
    assert (out / "params.safetensors").exists()
    assert (out / "state.json").exists()
    assert (out / "optimizer.safetensors").exists()
    assert (out / "optimizer_state.json").exists()
    assert not (out / "muon_momentum.safetensors").exists()


def test_end_to_end_build_dry_run_on_synthetic(tmp_path):
    """build() with synthetic src+config, dry-run writes nothing but reports."""
    small = _tiny_spec(d_model=16, n_layers=2)
    src_path = tmp_path / "src.safetensors"
    save_file(_synthetic_src_sd(small), str(src_path))

    cfg_path = tmp_path / "pretrain-tiny.toml"
    cfg_path.write_text(
        "vocab_size = 40\n"
        "d_model = 32\n"
        "n_layers = 4\n"
        "n_heads = 4\n"
        "n_kv_heads = 2\n"
        "ffn_mult = 2.0\n"
        "max_seq_len = 64\n"
        "[embedding]\n"
        "weight_tying = true\n"
    )

    out = tmp_path / "out"
    rep = fa.build(src_path, cfg_path, out, dry_run=True)
    assert rep["target_tensors"] == 1 + 4 * 9 + 1
    assert not out.exists()  # dry-run writes nothing


def test_end_to_end_build_writes_loadable_shapes(tmp_path):
    small = _tiny_spec(d_model=16, n_layers=2)
    src_path = tmp_path / "src.safetensors"
    save_file(_synthetic_src_sd(small), str(src_path))

    cfg_path = tmp_path / "pretrain-tiny.toml"
    cfg_path.write_text(
        "vocab_size = 40\nd_model = 32\nn_layers = 4\nn_heads = 4\n"
        "n_kv_heads = 2\nffn_mult = 2.0\nmax_seq_len = 64\n"
        "[embedding]\nweight_tying = true\n"
    )

    out = tmp_path / "out"
    fa.build(src_path, cfg_path, out, dry_run=False)

    hdr = _read_safetensors_header(out / "params.safetensors")
    big = _tiny_spec(d_model=32, n_layers=4)
    for name, shape in big.tensor_shapes().items():
        assert hdr["v2." + name]["shape"] == list(shape)


# --------------------------------------------------------------------------- #
# GATED: the real aster-1b checkpoint (skips when the sibling repo is absent)
# --------------------------------------------------------------------------- #
def test_real_aster_1b_growth_plan():
    src = fa.DEFAULT_SRC
    cfg = fa.DEFAULT_CONFIG
    if not src.exists() or not cfg.exists():
        pytest.skip("real aster-1b checkpoint / 3b config not available")

    from safetensors.torch import load_file

    spec = fa.load_model_spec(cfg)
    src_sd = load_file(str(src))
    logical, results = fa.grow_checkpoint(src_sd, spec)

    # full 3b key set, exact shapes.
    target = spec.tensor_shapes()
    assert set(logical.keys()) == set(target.keys())
    for name, shape in target.items():
        assert tuple(logical[name].shape) == shape

    rep = fa.grow_report(results)
    # aster-1b has 26 blocks, 3b has 28 -> 2 new blocks * 9 = 18 fresh-init.
    assert rep["fresh_init"] == 18
    assert rep["grown_from_1b"] == 236
    assert rep["target_tensors"] == 254

    # optimizer partition: 28 attn_norm + 28 ffn_norm + embed + final_norm = 58
    # AdamW params; the rest (28*7 = 196 matrices) route to Muon.
    adamw = fa.adamw_param_names(target)
    assert len(adamw) == 58
    assert len(target) - len(adamw) == 196  # Muon-routed matrices
