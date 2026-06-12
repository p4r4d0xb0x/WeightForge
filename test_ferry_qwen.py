"""Tests for ferry_qwen.py -- distilling the real Qwen3-0.6B into a smaller ferry-?B.

These tests are GATED: they ``pytest.importorskip('transformers')`` and skip the
whole module if the dependency or the model weights are unavailable, so the core
``test_ferry.py`` suite (42 tests) stays fast and dependency-light. The heavy
tests load Qwen3-0.6B on CPU (one ~39s load, reused across tests via a fixture)
and run a short data-free distill, so they are slow by nature.

Run only these:   python -m pytest test_ferry_qwen.py -q
Skip these:       python -m pytest test_ferry.py -q   (unaffected)

Constraints asserted: CPU-only, data-free (synthetic token probes), and the
architecture-changed student is genuinely smaller than the teacher.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # enforce CPU before torch

import pytest
import torch

# Gate the entire module on transformers being importable.
pytest.importorskip("transformers", reason="transformers not installed")

import ferry
import ferry_qwen as fq


# --------------------------------------------------------------------------- #
# shared fixtures (load the real teacher once; it is the expensive part)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def teacher():
    try:
        return fq.load_teacher()
    except Exception as exc:  # network/model unavailable -> skip, do not fail
        pytest.skip(f"Qwen3-0.6B unavailable: {exc}")


@pytest.fixture(scope="module")
def small_preset():
    # A deliberately tiny student so the distill test is fast on CPU.
    return fq.StudentPreset("ferry-test", hidden_size=256, intermediate_size=768,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_no_gpu_is_used(teacher):
    """DEC-007: GPU is forbidden. All teacher params must live on CPU."""
    assert all(p.device.type == "cpu" for p in teacher.parameters())


def test_student_is_smaller_and_architecture_changed(teacher, small_preset):
    """The student must be a genuinely smaller, different-architecture model."""
    student = fq.build_student(small_preset)
    t_cfg = teacher.inner.config
    s_cfg = student.inner.config
    # architecture changed: fewer layers and narrower hidden than the teacher
    assert s_cfg.num_hidden_layers < t_cfg.num_hidden_layers
    assert s_cfg.hidden_size < t_cfg.hidden_size
    # genuinely smaller in parameter count
    assert fq.param_count(student) < fq.param_count(teacher)
    # same vocabulary (so no VocabMap needed)
    assert s_cfg.vocab_size == t_cfg.vocab_size


def test_student_is_uniform_float32(teacher, small_preset):
    """CPU autograd needs a uniform dtype; Qwen3 defaults to bfloat16, so the
    student loader must force float32 across ALL params (regression for the
    'Found dtype Float but expected BFloat16' backward error)."""
    student = fq.build_student(small_preset)
    assert all(p.dtype == torch.float32 for p in student.parameters())
    assert all(p.dtype == torch.float32 for p in teacher.parameters())


def test_logits_adapter_returns_tensor(teacher, small_preset):
    """LogitsModel.forward must return a raw (batch, seq, vocab) tensor so
    ferry.agreement / ferry.distill can consume it like a TinyLM."""
    student = fq.build_student(small_preset)
    ids = ferry.token_probe(2, 6, teacher.inner.config.vocab_size, seed=1)
    out = student(ids)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 6, student.inner.config.vocab_size)


def test_transfer_is_shape_safe_and_full_coverage(teacher, small_preset):
    """Stage 1: name-matched transfer must load cleanly into the student and
    cover every student tensor (Copy/CropPad/SvdProject; no Skip expected here)."""
    student = fq.build_student(small_preset)
    rep = fq.transfer_into_student(teacher, student)
    assert rep["coverage"] == 1.0
    assert rep["matched"] == rep["student_tensors"]
    # student still runs after the transfer
    ids = ferry.token_probe(1, 6, teacher.inner.config.vocab_size, seed=3)
    assert student(ids).shape[-1] == teacher.inner.config.vocab_size


def test_distill_is_data_free_and_improves_agreement(teacher, small_preset):
    """Stage 3: a short data-free distill must improve per-token agreement
    (lower MSE / higher cosine) over the post-transfer warm start.

    Data-free: ferry.distill only draws synthetic token probes internally; no
    dataset or disk path is touched here.
    """
    torch.manual_seed(0)
    student = fq.build_student(small_preset)
    fq.transfer_into_student(teacher, student)
    vocab = teacher.inner.config.vocab_size

    before = fq.evaluate(teacher, student, seq=12, seed=2)
    after = ferry.distill(
        student, teacher, vocab=vocab, seq=12, steps=30, batch=4, lr=2e-3, seed=0
    )
    # distill should reduce logit MSE and raise cosine (directional agreement).
    assert after["mse"] < before["mse"]
    assert after["cosine"] > before["cosine"]
