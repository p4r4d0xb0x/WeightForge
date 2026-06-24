"""Tests for Ferry, the data-free weight transfer PoC (ferry.py).

Run: pytest
"""

from __future__ import annotations

import torch

import ferry as clone


# --------------------------------------------------------------------------- #
# transform_tensor: the four deterministic cases
# --------------------------------------------------------------------------- #
def test_copy_same_shape() -> None:
    src = torch.randn(4, 3)
    out, kind = clone.transform_tensor(src, (4, 3))
    assert kind == "Copy"
    assert torch.equal(out, src)
    assert out is not src  # must be a clone, not an alias


def test_crop_pad_1d_crop() -> None:
    src = torch.arange(6, dtype=torch.float32)
    out, kind = clone.transform_tensor(src, (4,))
    assert kind == "CropPad"
    assert out.shape == (4,)
    assert torch.equal(out, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_crop_pad_1d_pad() -> None:
    src = torch.tensor([1.0, 2.0])
    out, kind = clone.transform_tensor(src, (4,))
    assert kind == "CropPad"
    assert out.shape == (4,)
    assert torch.equal(out, torch.tensor([1.0, 2.0, 0.0, 0.0]))


def test_svd_project_2d_dim_mismatch() -> None:
    src = torch.randn(128, 256)
    out, kind = clone.transform_tensor(src, (32, 64))
    assert kind == "SvdProject"
    assert out.shape == (32, 64)
    assert torch.isfinite(out).all()


def test_grow_2d_embeds_teacher_top_left_not_svd() -> None:
    """Pure 2D growth must EMBED the teacher (CropPad), not SVD-rotate it.

    Regression for the 1B->3B growth bug: when both dims grow, the old SVD
    path collapsed to ``diag(sigma)`` (rotated singular values), destroying the
    teacher's input/output basis and producing pathological activations that
    overflowed the forward pass at scale. The correct data-free growth is to
    place the teacher matrix in the top-left block and zero-pad the new
    rows/cols, preserving the teacher's forward map exactly on the original
    subspace.
    """
    src = torch.randn(4, 4)
    out, kind = clone.transform_tensor(src, (8, 8))
    assert kind == "CropPad"
    assert out.shape == (8, 8)
    # Teacher embedded verbatim in the top-left block; new rows/cols are zero.
    assert torch.equal(out[:4, :4], src)
    assert torch.equal(out[4:, :], torch.zeros(4, 8))
    assert torch.equal(out[:, 4:], torch.zeros(8, 4))
    # Forward map preserved on the original subspace: for an input whose new
    # dims are zero, the grown weight reproduces the teacher's output exactly.
    x = torch.cat([torch.randn(4), torch.zeros(4)])
    assert torch.allclose(out @ x, torch.cat([src @ x[:4], torch.zeros(4)]))


def test_mixed_grow_shrink_2d_uses_svd() -> None:
    """One side grows, one shrinks -> SVD restricts the shrinking side."""
    src = torch.randn(64, 128)
    out, kind = clone.transform_tensor(src, (96, 32))  # rows grow, cols shrink
    assert kind == "SvdProject"
    assert out.shape == (96, 32)
    assert torch.isfinite(out).all()


def test_skip_on_rank_mismatch() -> None:
    src = torch.randn(4, 4)
    out, kind = clone.transform_tensor(src, (4,))
    assert kind == "Skip"
    # Skip returns the source untouched (caller keeps student tensor).
    assert out.shape == src.shape


# --------------------------------------------------------------------------- #
# transfer + report: end-to-end on toy MLPs
# --------------------------------------------------------------------------- #
def test_transfer_produces_loadable_state_dict() -> None:
    teacher = clone.MLP([784, 256, 128, 10])
    student = clone.MLP([784, 64, 32, 10])

    new_sd, results = clone.transfer(teacher.state_dict(), student.state_dict())

    # The mapped dict must load cleanly into the student (shapes preserved).
    student.load_state_dict(new_sd)
    assert len(results) > 0


def test_transfer_preserves_student_shapes() -> None:
    teacher = clone.MLP([784, 256, 128, 10])
    student = clone.MLP([784, 64, 32, 10])
    student_spec = clone.extract_spec(student.state_dict())

    new_sd, _ = clone.transfer(teacher.state_dict(), student.state_dict())

    for name, shape in student_spec.items():
        assert tuple(new_sd[name].shape) == shape


def test_exact_same_model_full_coverage_zero_error() -> None:
    torch.manual_seed(1)
    teacher = clone.MLP([16, 8, 4])
    # Identical architecture: every tensor should Copy with ~0 error vs itself.
    new_sd, results = clone.transfer(teacher.state_dict(), teacher.state_dict())
    rep = clone.report(results, teacher.state_dict())

    assert rep["coverage"] == 1.0
    assert rep["skipped"] == 0
    assert all(r.kind == "Copy" for r in results)
    assert rep["mean_relative_error"] == 0.0


def test_report_structure() -> None:
    teacher = clone.MLP([784, 256, 10])
    student = clone.MLP([784, 64, 10])
    _, results = clone.transfer(teacher.state_dict(), student.state_dict())
    rep = clone.report(results, student.state_dict())

    for key in (
        "student_tensors",
        "matched",
        "transferred",
        "skipped",
        "coverage",
        "mean_relative_error",
        "by_kind",
    ):
        assert key in rep
    assert 0.0 <= rep["coverage"] <= 1.0


def test_no_match_leaves_student_untouched() -> None:
    teacher = clone.MLP([8, 4])
    student = clone.MLP([8, 4])
    # Rename teacher tensors so nothing matches by name.
    renamed = {f"other.{k}": v for k, v in teacher.state_dict().items()}

    original = {k: v.clone() for k, v in student.state_dict().items()}
    new_sd, results = clone.transfer(renamed, student.state_dict())

    assert results == []
    for k in original:
        assert torch.equal(new_sd[k], original[k])


# --------------------------------------------------------------------------- #
# stage 2: synthetic probe, agreement, output alignment ("same answer" goal)
# --------------------------------------------------------------------------- #
def test_synthetic_probe_shape_and_determinism() -> None:
    a = clone.synthetic_probe(16, 8, seed=42)
    b = clone.synthetic_probe(16, 8, seed=42)
    assert a.shape == (16, 8)
    assert torch.equal(a, b)  # same seed -> reproducible, no disk/dataset


def test_agreement_identical_models_is_perfect() -> None:
    torch.manual_seed(0)
    model = clone.MLP([8, 16, 10])
    probe = clone.synthetic_probe(64, 8, seed=1)
    agr = clone.agreement(model, model, probe)
    assert agr["top1_agree"] == 1.0
    assert agr["mse"] < 1e-12
    assert agr["cosine"] > 0.999


def test_align_output_guarantees_same_answer_when_capacity_suffices() -> None:
    # Student penultimate width (32) >= what is needed to reconstruct the
    # teacher's 10-dim output map -> alignment must make answers identical.
    torch.manual_seed(0)
    teacher = clone.MLP([8, 16, 10])
    student = clone.MLP([8, 32, 10])  # wide enough

    fit_probe = clone.synthetic_probe(256, 8, seed=1)
    clone.align_output(student, teacher, fit_probe)

    # Held-out probe: the match must generalize, not just fit the probe.
    held = clone.synthetic_probe(256, 8, seed=2)
    agr = clone.agreement(teacher, student, held)
    assert agr["top1_agree"] == 1.0
    assert agr["mse"] < 1e-8


def test_align_output_bottleneck_cannot_fully_match() -> None:
    # Student too narrow (width 2) -> exact agreement is impossible; Ferry must
    # surface the residual rather than fake a perfect match.
    torch.manual_seed(0)
    teacher = clone.MLP([8, 16, 10])
    student = clone.MLP([8, 2, 10])  # bottleneck

    probe = clone.synthetic_probe(256, 8, seed=1)
    agr = clone.align_output(student, teacher, probe)
    assert agr["top1_agree"] < 1.0
    assert agr["mse"] > 1e-6


def test_align_output_improves_agreement() -> None:
    torch.manual_seed(0)
    teacher = clone.MLP([8, 16, 10])
    student = clone.MLP([8, 32, 10])
    probe = clone.synthetic_probe(256, 8, seed=1)

    before = clone.agreement(teacher, student, probe)
    after = clone.align_output(student, teacher, probe)
    assert after["mse"] < before["mse"]


# --------------------------------------------------------------------------- #
# nonlinear model (ActMLP): the "more complex" demo path
# --------------------------------------------------------------------------- #
def test_actmlp_forward_shape() -> None:
    model = clone.ActMLP([8, 16, 4], act="relu")
    out = model(clone.synthetic_probe(5, 8, seed=0))
    assert out.shape == (5, 4)


def test_actmlp_is_nonlinear() -> None:
    # A nonlinear net must violate additivity f(a+b) != f(a)+f(b) in general.
    torch.manual_seed(0)
    model = clone.ActMLP([4, 32, 4], act="relu")
    a = clone.synthetic_probe(1, 4, seed=1)
    b = clone.synthetic_probe(1, 4, seed=2)
    with torch.no_grad():
        lhs = model(a + b)
        rhs = model(a) + model(b)
    assert not torch.allclose(lhs, rhs, atol=1e-4)


def test_actmlp_self_align_is_exact() -> None:
    # Re-fitting a model's last layer to match ITSELF must be exact: the model's
    # own head is a valid least-squares solution, even with a nonlinear body.
    torch.manual_seed(0)
    model = clone.ActMLP([8, 32, 10], act="gelu")
    probe = clone.synthetic_probe(256, 8, seed=1)
    agr = clone.align_output(model, model, probe)
    assert agr["top1_agree"] == 1.0
    assert agr["mse"] < 1e-10


def test_actmlp_nonlinear_teacher_leaves_held_out_residual() -> None:
    # With a nonlinear teacher and a different-bodied nonlinear student, matching
    # the probe does NOT make held-out inputs match exactly. Ferry must surface
    # that residual rather than report a fake perfect score.
    torch.manual_seed(0)
    teacher = clone.ActMLP([16, 48, 32, 10], act="relu")
    student = clone.ActMLP([16, 64, 10], act="relu")  # wide but different basis

    clone.align_output(student, teacher, clone.synthetic_probe(512, 16, seed=1))
    held = clone.agreement(teacher, student, clone.synthetic_probe(512, 16, seed=2))
    assert held["top1_agree"] < 1.0
    assert held["mse"] > 1e-6


# --------------------------------------------------------------------------- #
# align_hidden: closed-form hidden alignment that SUPPORTS nonlinear teachers
# --------------------------------------------------------------------------- #
def test_align_hidden_lifts_nonlinear_matched_depth() -> None:
    # A matched-depth nonlinear student: hidden alignment must beat head-only by a
    # wide margin and reach near-exact held-out agreement.
    in_dim, out_dim = 32, 10
    probe = clone.synthetic_probe(512, in_dim, seed=1)
    held = clone.synthetic_probe(512, in_dim, seed=2)
    for act in ("relu", "gelu", "tanh"):
        torch.manual_seed(0)
        teacher = clone.ActMLP([in_dim, 96, 64, out_dim], act=act)
        torch.manual_seed(1)
        s_head = clone.ActMLP([in_dim, 128, 72, out_dim], act=act)
        torch.manual_seed(1)
        s_hidden = clone.ActMLP([in_dim, 128, 72, out_dim], act=act)

        clone.align_output(s_head, teacher, probe)
        base = clone.agreement(teacher, s_head, held)["top1_agree"]
        clone.align_hidden(s_hidden, teacher, probe)
        lifted = clone.agreement(teacher, s_hidden, held)["top1_agree"]

        assert lifted > base + 0.1  # meaningful improvement
        assert lifted > 0.9  # near-exact for matched depth


def test_align_hidden_never_worse_on_depth_mismatch() -> None:
    # A shallower student cannot fully track a deeper teacher, but hidden
    # alignment must still be no worse than head-only (honest partial support).
    in_dim, out_dim = 32, 10
    probe = clone.synthetic_probe(512, in_dim, seed=1)
    held = clone.synthetic_probe(512, in_dim, seed=2)
    torch.manual_seed(0)
    teacher = clone.ActMLP([in_dim, 96, 64, out_dim], act="gelu")
    torch.manual_seed(1)
    s_head = clone.ActMLP([in_dim, 128, out_dim], act="gelu")
    torch.manual_seed(1)
    s_hidden = clone.ActMLP([in_dim, 128, out_dim], act="gelu")

    clone.align_output(s_head, teacher, probe)
    base = clone.agreement(teacher, s_head, held)["top1_agree"]
    clone.align_hidden(s_hidden, teacher, probe)
    lifted = clone.agreement(teacher, s_hidden, held)["top1_agree"]
    assert lifted >= base - 1e-6


def test_align_hidden_falls_back_to_head_only_for_tinylm() -> None:
    # TinyLM has no flat linear chain (_linear_chain -> []), so align_hidden must
    # behave exactly like align_output (head-only) and not raise.
    torch.manual_seed(0)
    vocab, seq = 40, 8
    teacher = clone.TinyLM(vocab, dim=32, heads=4, layers=2, seq=seq)
    student = clone.TinyLM(vocab, dim=24, heads=4, layers=2, seq=seq)
    assert clone._linear_chain(teacher) == []
    probe = clone.token_probe(128, seq, vocab, seed=1)
    out = clone.align_hidden(student, teacher, probe)
    assert set(out) == {"mse", "top1_agree", "cosine"}


def test_align_hidden_exact_when_self_aligned() -> None:
    # Aligning a model to itself (same arch) must be exact regardless of hidden
    # alignment touching intermediate layers.
    torch.manual_seed(0)
    teacher = clone.ActMLP([16, 40, 24, 10], act="relu")
    student = clone.ActMLP([16, 40, 24, 10], act="relu")
    student.load_state_dict(teacher.state_dict())
    probe = clone.synthetic_probe(256, 16, seed=5)
    out = clone.align_hidden(student, teacher, probe)
    assert out["top1_agree"] == 1.0
    assert out["mse"] < 1e-8


# --------------------------------------------------------------------------- #
# LLM-like model (TinyLM): a tiny self-contained transformer
# --------------------------------------------------------------------------- #
def test_token_probe_shape_and_determinism() -> None:
    a = clone.token_probe(8, 5, 32, seed=3)
    b = clone.token_probe(8, 5, 32, seed=3)
    assert a.shape == (8, 5)
    assert a.dtype == torch.long
    assert int(a.max()) < 32 and int(a.min()) >= 0
    assert torch.equal(a, b)


def test_tinylm_forward_shape() -> None:
    vocab, seq = 32, 6
    model = clone.TinyLM(vocab, dim=24, heads=4, layers=2, seq=seq)
    out = model(clone.token_probe(3, seq, vocab, seed=0))
    assert out.shape == (3, seq, vocab)  # (batch, seq, vocab) logits


def test_tinylm_transfer_is_loadable_and_shape_safe() -> None:
    torch.manual_seed(0)
    seq = 8
    teacher = clone.TinyLM(40, dim=32, heads=4, layers=3, seq=seq)
    student = clone.TinyLM(40, dim=24, heads=4, layers=2, seq=seq)
    student_spec = clone.extract_spec(student.state_dict())

    new_sd, results = clone.transfer(teacher.state_dict(), student.state_dict())
    # Every student tensor keeps its own shape -> the dict must load cleanly.
    loadable = {
        k: v for k, v in new_sd.items()
        if k in student_spec and tuple(v.shape) == student_spec[k]
    }
    student.load_state_dict({**student.state_dict(), **loadable})
    assert len(results) > 0
    rep = clone.report(results, student.state_dict())
    assert rep["coverage"] > 0.0


def test_tinylm_align_improves_per_token_agreement() -> None:
    # Re-fitting the LM head per token position must reduce held-out logit MSE,
    # even though the transformer is too nonlinear for an exact match.
    torch.manual_seed(0)
    vocab, seq = 48, 10
    teacher = clone.TinyLM(vocab, dim=48, heads=4, layers=2, seq=seq)
    student = clone.TinyLM(vocab, dim=32, heads=4, layers=2, seq=seq)

    held = clone.token_probe(128, seq, vocab, seed=2)
    before = clone.agreement(teacher, student, held)
    clone.align_output(student, teacher, clone.token_probe(256, seq, vocab, seed=1))
    after = clone.agreement(teacher, student, held)

    assert after["mse"] < before["mse"]
    assert after["top1_agree"] < 1.0  # honest: nonlinear depth blocks exact match


def test_tinylm_generation_residual_compounds() -> None:
    # Greedy autoregressive decode: teacher/student token agreement at a longer
    # horizon must not exceed the short-horizon agreement -- residuals compound.
    torch.manual_seed(0)
    vocab, seq = 48, 10
    teacher = clone.TinyLM(vocab, dim=48, heads=4, layers=2, seq=seq)
    student = clone.TinyLM(vocab, dim=32, heads=4, layers=2, seq=seq)
    clone.align_output(student, teacher, clone.token_probe(256, seq, vocab, seed=1))

    ctx = clone.token_probe(64, 4, vocab, seed=7)
    short = teacher.generate(ctx, 1)[:, 4:]
    short_s = student.generate(ctx, 1)[:, 4:]
    long = teacher.generate(ctx, 8)[:, 4:]
    long_s = student.generate(ctx, 8)[:, 4:]

    short_match = (short == short_s).float().mean().item()
    long_match = (long == long_s).float().mean().item()
    assert long_match <= short_match  # longer horizon never matches better


# --------------------------------------------------------------------------- #
# Stage 3: distill (gradient fine-tune on fresh synthetic probes, data-free)
# --------------------------------------------------------------------------- #
def test_distill_requires_exactly_one_input_mode() -> None:
    teacher = clone.ActMLP([8, 16, 4], act="relu")
    student = clone.ActMLP([8, 16, 4], act="relu")
    # neither mode
    try:
        clone.distill(student, teacher)
        raise AssertionError("expected ValueError when no input mode is given")
    except ValueError:
        pass
    # both modes
    try:
        clone.distill(student, teacher, in_dim=8, vocab=4)
        raise AssertionError("expected ValueError when both modes are given")
    except ValueError:
        pass
    # token mode without seq
    try:
        clone.distill(student, teacher, vocab=4)
        raise AssertionError("expected ValueError when seq missing in token mode")
    except ValueError:
        pass


def test_distill_closes_nonlinear_gap_when_capacity_adequate() -> None:
    # A depth-matched, adequately wide student should reach near-exact agreement
    # on a held-out probe after closed-form warm start + gradient distillation --
    # this is the nonlinear limit being CLOSED, not merely narrowed.
    in_dim, out_dim = 16, 6
    torch.manual_seed(0)
    teacher = clone.ActMLP([in_dim, 48, 32, out_dim], act="relu")
    torch.manual_seed(1)
    student = clone.ActMLP([in_dim, 64, 48, out_dim], act="relu")

    held = clone.synthetic_probe(512, in_dim, seed=2)
    base = clone.agreement(teacher, student, held)["top1_agree"]
    clone.align_hidden(student, teacher, clone.synthetic_probe(512, in_dim, seed=1))
    out = clone.distill(student, teacher, in_dim=in_dim, steps=400)

    assert out["top1_agree"] > base       # strictly better than the warm start base
    assert out["top1_agree"] > 0.95       # gap essentially closed
    assert out["mse"] < 1e-3


def test_distill_improves_tinylm_per_token_agreement() -> None:
    # Gradient distillation on fresh synthetic token probes must push per-token
    # agreement well past what closed-form last-layer alignment alone reaches.
    torch.manual_seed(0)
    vocab, seq = 48, 10
    teacher = clone.TinyLM(vocab, dim=48, heads=4, layers=2, seq=seq)
    student = clone.TinyLM(vocab, dim=32, heads=4, layers=2, seq=seq)

    held = clone.token_probe(128, seq, vocab, seed=2)
    clone.align_output(student, teacher, clone.token_probe(256, seq, vocab, seed=1))
    after_align = clone.agreement(teacher, student, held)["top1_agree"]
    out = clone.distill(student, teacher, vocab=vocab, seq=seq, steps=300, lr=3e-4)

    assert out["top1_agree"] > after_align  # distill beats closed-form alone
    assert out["top1_agree"] > 0.7          # large absolute gain on per-token match


def test_distill_is_data_free_runs_from_models_and_dims_only() -> None:
    # No dataset, no file, no tokenizer -- distill needs only the two models and
    # the probe dimensions. A successful run that lowers MSE proves data-freeness.
    in_dim = 12
    torch.manual_seed(0)
    teacher = clone.ActMLP([in_dim, 24, 5], act="gelu")
    torch.manual_seed(1)
    student = clone.ActMLP([in_dim, 24, 5], act="gelu")

    held = clone.synthetic_probe(256, in_dim, seed=2)
    before = clone.agreement(teacher, student, held)["mse"]
    out = clone.distill(student, teacher, in_dim=in_dim, steps=200)
    assert out["mse"] < before


# --------------------------------------------------------------------------- #
# Stage 0: vocabulary reconciliation (LLM teacher/student with different vocabs)
# --------------------------------------------------------------------------- #
def test_build_vocab_map_projection_is_a_selection_matrix() -> None:
    # t_for_s maps student tokens to teacher ids; -1 means student-only (no target).
    t_for_s = torch.tensor([2, 0, -1, 1])  # V_s = 4
    vmap = clone.build_vocab_map(t_for_s, size_t=3)
    assert vmap.size_t == 3 and vmap.size_s == 4
    assert vmap.projection.shape == (3, 4)  # (V_t, V_s)
    # Mapped columns are one-hot at the teacher id; the student-only column is zero.
    assert torch.equal(vmap.projection[:, 0], torch.tensor([0.0, 0.0, 1.0]))  # ->2
    assert torch.equal(vmap.projection[:, 1], torch.tensor([1.0, 0.0, 0.0]))  # ->0
    assert torch.equal(vmap.projection[:, 2], torch.tensor([0.0, 0.0, 0.0]))  # -1
    assert torch.equal(vmap.projection[:, 3], torch.tensor([0.0, 1.0, 0.0]))  # ->1


def test_vocab_map_project_selects_teacher_logit_columns() -> None:
    t_for_s = torch.tensor([2, 0, 1])
    vmap = clone.build_vocab_map(t_for_s, size_t=3)
    teacher_logits = torch.tensor([[10.0, 20.0, 30.0]])  # (1, V_t=3)
    projected = vmap.project(teacher_logits)  # (1, V_s=3)
    # student col j gets teacher column t_for_s[j]: [t2, t0, t1] = [30, 10, 20]
    assert torch.equal(projected, torch.tensor([[30.0, 10.0, 20.0]]))


def test_vocab_map_remap_ids_translates_student_to_teacher_space() -> None:
    t_for_s = torch.tensor([5, 7, -1, 3])
    vmap = clone.build_vocab_map(t_for_s, size_t=8)
    student_ids = torch.tensor([[0, 3, 1]])
    teacher_ids = vmap.remap_ids(student_ids)
    assert torch.equal(teacher_ids, torch.tensor([[5, 3, 7]]))
    # Student-only ids (-1) are clamped to a valid teacher token (0), not negative.
    assert int(vmap.remap_ids(torch.tensor([2])).item()) == 0


def test_reconcile_vocab_identity_when_sizes_match_is_a_noop() -> None:
    # Same vocab size -> shared-prefix map is the identity; agreement with the map
    # must equal agreement without it (the no-op guarantee that protects LM tests).
    torch.manual_seed(0)
    vocab, seq = 16, 6
    teacher = clone.TinyLM(vocab, dim=24, heads=4, layers=2, seq=seq)
    student = clone.TinyLM(vocab, dim=24, heads=4, layers=2, seq=seq)
    vmap = clone.reconcile_vocab(student, teacher)
    assert torch.equal(vmap.t_for_s, torch.arange(vocab))
    probe = clone.token_probe(32, seq, vocab, seed=1)
    a_plain = clone.agreement(teacher, student, probe)
    a_mapped = clone.agreement(teacher, student, probe, vmap)
    assert abs(a_plain["mse"] - a_mapped["mse"]) < 1e-6
    assert abs(a_plain["top1_agree"] - a_mapped["top1_agree"]) < 1e-6


def test_shared_token_probe_only_samples_mapped_tokens() -> None:
    t_for_s = torch.tensor([0, -1, 2, -1, 4])  # shared student ids: {0, 2, 4}
    vmap = clone.build_vocab_map(t_for_s, size_t=5)
    probe = clone.shared_token_probe(8, 5, vmap, seed=0)
    assert probe.shape == (8, 5)
    uniq = set(probe.flatten().tolist())
    assert uniq <= {0, 2, 4}  # never samples a student-only token


def test_align_output_with_vocab_map_handles_different_vocab_sizes() -> None:
    # Teacher V=24, student V=16: heads have different widths, so alignment is only
    # well-defined AFTER reconciliation. The map makes the target student-shaped
    # and lifts per-token agreement well above the unaligned baseline.
    torch.manual_seed(0)
    seq = 8
    vocab_t, vocab_s = 24, 16
    teacher = clone.TinyLM(vocab_t, dim=32, heads=4, layers=2, seq=seq)
    student = clone.TinyLM(vocab_s, dim=24, heads=4, layers=2, seq=seq)
    vmap = clone.reconcile_vocab(student, teacher)

    held = clone.shared_token_probe(128, seq, vmap, seed=2)
    base = clone.agreement(teacher, student, held, vmap)["top1_agree"]
    after = clone.align_output(
        student, teacher, clone.shared_token_probe(256, seq, vmap, seed=1), vmap
    )
    assert after["top1_agree"] > base  # reconciled alignment strictly improves


def test_distill_with_vocab_map_closes_cross_vocab_gap() -> None:
    # Stage 0 + stage 3 together push cross-vocab per-token agreement high, proving
    # the vocabulary mismatch is reconciled rather than worked around.
    torch.manual_seed(0)
    seq = 8
    vocab_t, vocab_s = 24, 16
    teacher = clone.TinyLM(vocab_t, dim=32, heads=4, layers=2, seq=seq)
    student = clone.TinyLM(vocab_s, dim=24, heads=4, layers=2, seq=seq)
    vmap = clone.reconcile_vocab(student, teacher)

    clone.align_output(
        student, teacher, clone.shared_token_probe(256, seq, vmap, seed=1), vmap
    )
    held = clone.shared_token_probe(128, seq, vmap, seed=2)
    after_align = clone.agreement(teacher, student, held, vmap)["top1_agree"]
    out = clone.distill(
        student, teacher, vocab=vocab_s, seq=seq, steps=300, lr=3e-4, vocab_map=vmap
    )
    assert out["top1_agree"] > after_align
    assert out["top1_agree"] > 0.7


def test_distill_rejects_vocab_map_in_continuous_mode() -> None:
    # vocab_map is meaningless without a token vocabulary; guard against misuse.
    t_for_s = torch.tensor([0, 1])
    vmap = clone.build_vocab_map(t_for_s, size_t=2)
    teacher = clone.ActMLP([8, 16, 4], act="relu")
    student = clone.ActMLP([8, 16, 4], act="relu")
    try:
        clone.distill(student, teacher, in_dim=8, vocab_map=vmap)
        raise AssertionError("expected ValueError for vocab_map in continuous mode")
    except ValueError:
        pass


def _scrambled_map(vocab_s: int, vocab_t: int, n_shared: int, seed: int):
    """Non-trivial student->teacher vocab map: random ids in random slots, partial.

    Unlike the shared-prefix default, token id ``j`` matches an arbitrary teacher
    token and ``vocab_s - n_shared`` student tokens have no teacher match (``-1``).
    This is the realistic worst-case correspondence.
    """
    g = torch.Generator().manual_seed(seed)
    t_for_s = torch.full((vocab_s,), -1, dtype=torch.long)
    teacher_ids = torch.randperm(vocab_t, generator=g)[:n_shared]
    student_slots = torch.randperm(vocab_s, generator=g)[:n_shared]
    t_for_s[student_slots] = teacher_ids
    return clone.build_vocab_map(t_for_s, vocab_t)


def _build_combined_mismatch():
    """Teacher/student differing in vocab AND depth AND width, scrambled vocab map.

    Returns (teacher, student, vmap, seq) after weight transfer. Shared across the
    two worst-case regression tests so they exercise the identical setup.
    """
    torch.manual_seed(0)
    seq = 12
    vocab_t, vocab_s = 72, 48
    teacher = clone.TinyLM(vocab_t, dim=80, heads=4, layers=4, seq=seq)
    student = clone.TinyLM(vocab_s, dim=40, heads=4, layers=2, seq=seq)
    new_sd, _ = clone.transfer(teacher.state_dict(), student.state_dict())
    student.load_state_dict({**student.state_dict(), **{
        k: v for k, v in new_sd.items() if k in student.state_dict()
        and v.shape == student.state_dict()[k].shape
    }})
    vmap = _scrambled_map(vocab_s, vocab_t, n_shared=40, seed=5)
    return teacher, student, vmap, seq


def test_combined_mismatch_baseline_is_near_zero() -> None:
    # With vocab + depth + width all different and a SCRAMBLED partial vocab map,
    # raw weight transfer alone produces essentially no agreement -- the honest
    # baseline for "two genuinely different LMs". (Contrast the clean shared-prefix
    # demo, which starts ~0.41.) This documents the worst-case starting point.
    teacher, student, vmap, seq = _build_combined_mismatch()
    held = clone.shared_token_probe(256, seq, vmap, seed=2)
    base = clone.agreement(teacher, student, held, vmap)["top1_agree"]
    assert base < 0.10  # scrambled map -> near-random before any alignment


def test_combined_mismatch_pipeline_recovers_all_three_axes() -> None:
    # The full staged pipeline (stage 0 reconcile -> stage 2 head align -> stage 3
    # distill) must monotonically lift agreement from near-zero and recover the
    # worst case to a high per-token match, proving the three mismatches (LM-head
    # vocab, middle-layer depth, hidden width) are handled together, not in
    # isolation.
    teacher, student, vmap, seq = _build_combined_mismatch()
    probe = clone.shared_token_probe(256, seq, vmap, seed=1)
    held = clone.shared_token_probe(256, seq, vmap, seed=2)

    base = clone.agreement(teacher, student, held, vmap)["top1_agree"]
    clone.align_output(student, teacher, probe, vmap)
    after_align = clone.agreement(teacher, student, held, vmap)["top1_agree"]
    # 1500 steps: the two-sided orthogonal SVD projection (U_m^T A V_n) warm-starts
    # this scrambled-vocab worst case slightly lower than the old slice form, so the
    # gradient loop needs a few hundred more steps to clear 0.8 -- but it converges
    # HIGHER (0.86+ at 2500). More steps, not a lower bar.
    out = clone.distill(
        student, teacher, vocab=48, seq=seq, steps=1500, lr=3e-4, vocab_map=vmap
    )

    assert after_align > base          # stage 2 lifts off the near-zero floor
    assert out["top1_agree"] > after_align  # stage 3 closes further
    assert out["top1_agree"] > 0.8     # worst case still recovers to a high match
