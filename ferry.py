"""Ferry -- make a student model give the SAME answers as a teacher.

Goal
----
Given a *teacher* and a *student* that may differ in layer count and hidden
dimension, make the student reproduce the teacher's outputs. Still no external
dataset is used: every input is a *synthetic probe* (random tensors / token ids
we generate ourselves). The teacher supplies the targets, so the pipeline is
data-free even though it now includes a gradient stage.

Stages
------
0. ``reconcile_vocab`` (LM only) -- when teacher and student have DIFFERENT
                        vocabularies (different tokenizers), build a ``VocabMap``
                        that maps student token ids to teacher token ids and
                        projects teacher logits into the student's vocab space.
                        Required before any LM merge/alignment, because a language
                        model emits a distribution over *its own* vocabulary.
1. ``transfer``      -- carry teacher weights into the student in weight space
                        (deterministic copy / crop-pad / SVD projection).
2. ``align_output`` / ``align_hidden``
                     -- closed-form least-squares alignment of the student's
                        final layer (and, for the MLP family, its hidden layers)
                        so student(x) == teacher(x) on a synthetic probe.
3. ``distill``       -- gradient fine-tune on *fresh* synthetic probes, warm-
                        started from stages 1-2. This is what actually CLOSES the
                        nonlinear / depth-mismatch / autoregressive limits that
                        closed-form alone cannot. Gradient training is now
                        allowed (the earlier "no gradient loop" constraint was
                        explicitly lifted); it stays data-free via synthetic
                        probes resampled every step (the key to generalization).

Guarantee and its condition
---------------------------
``align_output`` makes the student match the teacher *exactly* (up to float
error) for ANY input, **iff** the student's last hidden representation is rich
enough to linearly reconstruct the teacher's output. Concretely: if the student
penultimate width >= the rank of the teacher's output map, agreement -> 100%.
If the student is too narrow (a bottleneck), exact agreement is mathematically
impossible and Ferry reports the residual instead of hiding it.

Design (kept intentionally flat so it is easy to edit)
------------------------------------------------------
    extract_spec / match_tensors / transform_tensor / transfer / report
        -- weight-space transfer (stage 1)
    synthetic_probe(n, in_dim)      -> random probe inputs (no dataset)
    token_probe(n, seq, vocab)      -> random token-id sequences (LLM-like probe)
    reconcile_vocab(student, teacher) -> VocabMap (stage 0, vocab mismatch)
    shared_token_probe(n, seq, vmap)  -> token probe over the shared vocabulary
    agreement(teacher, student, x)  -> {mse, top1_agree, cosine}
    align_output(student, teacher, x) -> fit student's last layer (stage 2)
    align_hidden(student, teacher, x) -> closed-form hidden + head fit (stage 2b,
                                         lifts nonlinear teachers; MLP family)
    distill(student, teacher, ...)    -> gradient fine-tune on fresh synthetic
                                         probes (stage 3, closes the limits)

Demo models, increasing in complexity:
    MLP     -- linear stack (exact same-answer guarantee holds)
    ActMLP  -- nonlinear MLP (closed-form lifts it; distill closes it ~0.99)
    TinyLM  -- tiny GPT-style transformer (distill flattens autoregressive decay)

Transforms for shape mismatch (all deterministic, no learning):
    Copy / CropPad / SvdProject / Skip

Run the toy demo:
    python ferry.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #
@dataclass
class TensorMatch:
    """A teacher tensor paired with the student tensor of the same name."""

    name: str
    teacher_shape: tuple[int, ...]
    student_shape: tuple[int, ...]


@dataclass
class TransferResult:
    """Outcome of transferring one tensor."""

    name: str
    kind: str  # Copy | CropPad | SvdProject | Skip
    teacher_shape: tuple[int, ...]
    student_shape: tuple[int, ...]
    error: float  # relative drift vs original student tensor (0.0 == replaced cleanly)


# --------------------------------------------------------------------------- #
# core logic
# --------------------------------------------------------------------------- #
def extract_spec(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    """Return ``{tensor_name: shape}`` for every tensor in a state dict."""
    return {name: tuple(t.shape) for name, t in state_dict.items()}


def match_tensors(
    teacher_spec: dict[str, tuple[int, ...]],
    student_spec: dict[str, tuple[int, ...]],
) -> list[TensorMatch]:
    """Pair tensors that share a name in both models.

    Name-based matching keeps the PoC simple: both models are expected to use
    the same parameter names (e.g. two MLPs built the same way but with
    different widths). Tensors present in only one model are ignored here and
    surface later as skipped/uncovered.
    """
    matches: list[TensorMatch] = []
    for name, t_shape in teacher_spec.items():
        if name in student_spec:
            matches.append(TensorMatch(name, t_shape, student_spec[name]))
    return matches


def transform_tensor(
    src: torch.Tensor, dst_shape: tuple[int, ...]
) -> tuple[torch.Tensor, str]:
    """Map ``src`` onto ``dst_shape`` deterministically, no training.

    Returns the transformed tensor plus the name of the transform used.
    """
    src_shape = tuple(src.shape)

    if src_shape == dst_shape:
        return src.clone(), "Copy"

    # Different rank: we have no principled data-free reshape -> skip.
    if len(src_shape) != len(dst_shape):
        return src.clone(), "Skip"

    # 2D weight matrices with mismatched dims: use SVD low-rank projection so
    # the dominant directions of the teacher matrix survive into the student.
    if len(dst_shape) == 2 and src_shape != dst_shape:
        return _svd_project(src, dst_shape), "SvdProject"

    # Same rank, mismatched size, not a 2D matrix: crop or zero-pad per dim.
    return _crop_pad(src, dst_shape), "CropPad"


def _crop_pad(src: torch.Tensor, dst_shape: tuple[int, ...]) -> torch.Tensor:
    """Crop oversized dims and zero-pad undersized dims to reach ``dst_shape``."""
    # Crop first so every dim is <= dst.
    slices = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst_shape))
    cropped = src[slices]

    out = torch.zeros(dst_shape, dtype=src.dtype, device=src.device)
    out_slices = tuple(slice(0, s) for s in cropped.shape)
    out[out_slices] = cropped
    return out


def _svd_project(src: torch.Tensor, dst_shape: tuple[int, ...]) -> torch.Tensor:
    """Two-sided orthogonal SVD projection of a 2D ``src`` onto ``dst_shape``.

    A weight matrix ``A: (M, N)`` maps an ``N``-dim input space to an ``M``-dim
    output space. When the student is narrower (``m<M`` and/or ``n<N``) the
    principled data-free compression is to project ``A`` onto the **top singular
    subspaces** on both sides::

        A = U S Vh                       # full SVD (U:(M,r), Vh:(r,N))
        U_m = U[:, :m]                   # top-m left  (output) directions
        V_n = Vh[:n, :].T                # top-n right (input)  directions
        B   = U_m^T @ A @ V_n            # (m, n) student weight

    ``B = U_m^T A V_n`` is the Eckart-Young-optimal restriction of ``A`` to the
    student's principal output/input subspaces: the student input lives in the
    teacher's top-``n`` input directions and is mapped to the teacher's top-``m``
    output directions, with orthonormal ``U_m``/``V_n`` (``U_m^T U_m = I``,
    ``V_n^T V_n = I``). This supersedes the earlier "low-rank reconstruct then
    slice the top-left block" form, which broke orthogonality and biased toward
    the leading rows/columns. Purely algebraic: no data, no gradient steps.

    Dimension *growth* on a side (``m>M`` or ``n>N``) has no teacher signal to
    fill, so that side is zero-padded after projecting the available directions.
    """
    out_rows, out_cols = dst_shape
    a = src.float()

    u, _s, vh = torch.linalg.svd(a, full_matrices=False)
    rank = u.shape[1]  # = vh.shape[0] = min(M, N): usable singular directions

    # Top singular directions per side, capped by BOTH the student size and the
    # available rank. A wide/tall teacher has only ``rank`` directions, so we can
    # never take more than ``rank`` even if the student axis is larger.
    m = min(out_rows, rank)
    n = min(out_cols, rank)
    u_m = u[:, :m]          # (M, m) orthonormal output basis
    v_n = vh[:n, :].T       # (N, n) orthonormal input  basis

    # B = U_m^T A V_n : restrict A to the top output/input subspaces. (m, n)
    projected = u_m.T @ a @ v_n

    # Always normalize to the exact student shape: zero-pad any side that is
    # wider than the available rank (no teacher signal there). No-op when the
    # projection already matches ``dst_shape``.
    if tuple(projected.shape) != (out_rows, out_cols):
        projected = _crop_pad(projected, (out_rows, out_cols))
    return projected.to(src.dtype)


def _relative_error(new: torch.Tensor, old: torch.Tensor) -> float:
    """Relative L2 drift between the replaced tensor and the original student.

    1.0 means "completely changed"; small means the student barely moved. This
    is a weight-space diagnostic only -- it does NOT measure answer quality.
    """
    denom = old.float().norm().item()
    if denom == 0.0:
        return float(new.float().norm().item() > 0.0)
    return (new.float() - old.float()).norm().item() / denom


def transfer(
    teacher_sd: dict[str, torch.Tensor],
    student_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[TransferResult]]:
    """Build a new student state dict with teacher weights mapped in.

    Returns the new state dict (safe to ``load_state_dict``) and a per-tensor
    result log. Student tensors with no teacher match are left untouched.
    """
    matches = match_tensors(extract_spec(teacher_sd), extract_spec(student_sd))
    new_sd: dict[str, torch.Tensor] = {k: v.clone() for k, v in student_sd.items()}
    results: list[TransferResult] = []

    for m in matches:
        src = teacher_sd[m.name]
        old = student_sd[m.name]
        mapped, kind = transform_tensor(src, m.student_shape)

        if kind == "Skip":
            results.append(
                TransferResult(m.name, "Skip", m.teacher_shape, m.student_shape, 1.0)
            )
            continue

        new_sd[m.name] = mapped
        err = _relative_error(mapped, old)
        results.append(
            TransferResult(m.name, kind, m.teacher_shape, m.student_shape, err)
        )

    return new_sd, results


def report(
    results: list[TransferResult], student_sd: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Summarize a transfer into a small, printable feasibility report."""
    total_student = len(student_sd)
    transferred = [r for r in results if r.kind != "Skip"]
    by_kind: dict[str, int] = {}
    for r in results:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1

    coverage = len(transferred) / total_student if total_student else 0.0
    mean_err = (
        sum(r.error for r in transferred) / len(transferred) if transferred else 0.0
    )

    return {
        "student_tensors": total_student,
        "matched": len(results),
        "transferred": len(transferred),
        "skipped": len(results) - len(transferred),
        "coverage": round(coverage, 4),
        "mean_relative_error": round(mean_err, 4),
        "by_kind": by_kind,
    }


# --------------------------------------------------------------------------- #
# stage 2: synthetic probing + output alignment (make the answers match)
# --------------------------------------------------------------------------- #
def synthetic_probe(n: int, in_dim: int, seed: int | None = None) -> torch.Tensor:
    """Generate ``n`` random probe inputs of width ``in_dim``.

    These are self-generated tensors, NOT a dataset: no labels, no real-world
    samples, no disk I/O. They exist only to compare teacher/student behaviour
    and to drive the closed-form alignment below.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, in_dim, generator=gen)


def token_probe(n: int, seq: int, vocab: int, seed: int | None = None) -> torch.Tensor:
    """Generate ``n`` random token-id sequences of length ``seq`` over ``vocab``.

    The sequence-model analogue of ``synthetic_probe``: random integer token ids,
    NOT a corpus. No tokenizer, no text, no disk -- still strictly data-free.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n, seq), generator=gen)


# --------------------------------------------------------------------------- #
# stage 0: vocabulary reconciliation (LLM only)
# --------------------------------------------------------------------------- #
@dataclass
class VocabMap:
    """A correspondence between a student vocabulary and a teacher vocabulary.

    A language model emits a distribution over *its own* vocabulary, so when the
    teacher and student were built with different tokenizers (different vocab
    sizes / token ids), comparing or aligning their logits position-for-position
    is meaningless: index ``j`` denotes a different token on each side. Before any
    merge / alignment we must reconcile the two vocab spaces. ``VocabMap`` carries
    that correspondence and the two operations the rest of the pipeline needs:

      * :meth:`remap_ids` -- translate a probe expressed in *student* token ids
        into the *teacher* token ids, so the same probe can be fed to both models.
      * :meth:`project`   -- map teacher logits ``(.., V_t)`` into the student's
        logit space ``(.., V_s)``, so targets become dimensionally and
        semantically comparable to student outputs.

    The correspondence is ``t_for_s``: a LongTensor of shape ``(V_s,)`` whose
    entry ``j`` is the teacher token id that student token ``j`` corresponds to,
    or ``-1`` if the student token has no teacher counterpart (student-only
    token). In real LLMs this map comes from matching token *strings* across the
    two tokenizers; Ferry stays tokenizer-free and represents it abstractly.
    """

    t_for_s: torch.Tensor  # (V_s,) long; teacher id per student id, -1 = none
    size_t: int            # teacher vocabulary size (V_t)
    size_s: int            # student vocabulary size (V_s)
    projection: torch.Tensor  # (V_t, V_s) selection matrix, built at construction

    def remap_ids(self, student_ids: torch.Tensor) -> torch.Tensor:
        """Translate *student*-space token ids into *teacher*-space token ids.

        Student-only ids (``-1``) are clamped to ``0`` so the teacher still gets a
        valid token; callers that want clean targets should probe only over the
        shared region (see :func:`shared_token_probe`).
        """
        mapped = self.t_for_s[student_ids]
        return mapped.clamp_min(0)

    def project(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Map teacher logits ``(.., V_t)`` into student logit space ``(.., V_s)``.

        Column ``j`` of the result is the teacher logit of the token that student
        token ``j`` corresponds to; student-only columns are zero (no teacher
        signal -- an honest "we cannot supply a target for this token").
        """
        proj = self.projection.to(teacher_logits.dtype)
        return teacher_logits @ proj


def build_vocab_map(t_for_s: torch.Tensor, size_t: int) -> VocabMap:
    """Build a :class:`VocabMap` from a student->teacher id correspondence.

    ``t_for_s[j]`` is the teacher id for student token ``j`` (``-1`` = none).
    The projection is a ``(V_t, V_s)`` selection matrix: a one-hot column per
    mapped student token, a zero column per student-only token.
    """
    t_for_s = t_for_s.long()
    size_s = int(t_for_s.shape[0])
    if int(t_for_s.max().item() if size_s else -1) >= size_t:
        raise ValueError("build_vocab_map: a teacher id in t_for_s exceeds size_t")
    proj = torch.zeros(size_t, size_s)
    cols = torch.arange(size_s)
    mapped = t_for_s >= 0
    proj[t_for_s[mapped], cols[mapped]] = 1.0
    return VocabMap(t_for_s=t_for_s, size_t=size_t, size_s=size_s, projection=proj)


def reconcile_vocab(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    t_for_s: torch.Tensor | None = None,
) -> VocabMap:
    """Reconcile the student and teacher vocabularies (stage 0, run before merge).

    The vocab sizes are read from each model's output head (last ``nn.Linear``).
    If ``t_for_s`` is given it is used verbatim. Otherwise Ferry falls back to a
    *shared-prefix* correspondence -- student token ``j`` maps to teacher token
    ``j`` for ``j < min(V_t, V_s)`` and any student-only tail maps to ``-1``. That
    default models "the first tokens are shared subwords"; a real deployment would
    pass a ``t_for_s`` derived from matching tokenizer strings.
    """
    size_t = _last_linear(teacher).out_features
    size_s = _last_linear(student).out_features
    if t_for_s is None:
        shared = min(size_t, size_s)
        t_for_s = torch.full((size_s,), -1, dtype=torch.long)
        t_for_s[:shared] = torch.arange(shared)
    return build_vocab_map(t_for_s, size_t)


def shared_token_probe(
    n: int, seq: int, vocab_map: VocabMap, seed: int | None = None
) -> torch.Tensor:
    """Random *student*-space token probe restricted to the shared vocabulary.

    Only student tokens that have a teacher counterpart (``t_for_s >= 0``) are
    sampled, so every probe id maps to a meaningful teacher token. Returns ids in
    the student's vocab space (feed to the student directly; pass through
    :meth:`VocabMap.remap_ids` before feeding the teacher).
    """
    shared = torch.nonzero(vocab_map.t_for_s >= 0, as_tuple=False).flatten()
    if shared.numel() == 0:
        raise ValueError("shared_token_probe: no shared tokens between vocabularies")
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    pick = torch.randint(0, shared.numel(), (n, seq), generator=gen)
    return shared[pick]


@torch.no_grad()
def agreement(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    probe: torch.Tensor,
    vocab_map: VocabMap | None = None,
) -> dict[str, float]:
    """Measure how closely student outputs match teacher outputs on ``probe``.

    Returns mean-squared error, top-1 argmax agreement (classification view),
    and mean row cosine similarity. This is the metric the whole "same answer"
    goal is judged by.

    Leading dims are flattened so this works for both plain classifier outputs
    ``(n, out)`` and sequence-model logits ``(n, seq, vocab)`` -- in the latter
    case every (sample, position) pair is scored independently. For a 2D output
    the flatten is a no-op.

    ``vocab_map`` reconciles different vocabularies (stage 0): ``probe`` is given
    in *student* token ids, the teacher is fed the remapped teacher ids, and the
    teacher logits are projected into the student's vocab space before scoring.
    When ``None`` (same vocab, or a non-LM model) the behaviour is unchanged.
    """
    if vocab_map is None:
        yt = _flatten_logits(teacher(probe))
    else:
        yt = _flatten_logits(vocab_map.project(teacher(vocab_map.remap_ids(probe))))
    ys = _flatten_logits(student(probe))

    mse = (yt - ys).pow(2).mean().item()
    top1 = (yt.argmax(dim=-1) == ys.argmax(dim=-1)).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(yt, ys, dim=-1).mean().item()
    return {"mse": mse, "top1_agree": top1, "cosine": cos}


def _flatten_logits(out: torch.Tensor) -> torch.Tensor:
    """Collapse any leading dims into rows: ``(..., d) -> (prod(...), d)``."""
    return out.float().reshape(-1, out.shape[-1])


def _last_linear(model: torch.nn.Module) -> torch.nn.Linear:
    """Return the last ``nn.Linear`` in a module (the output head we re-fit)."""
    last: torch.nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last = module
    if last is None:
        raise ValueError("model has no nn.Linear layer to align")
    return last


@torch.no_grad()
def align_output(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    probe: torch.Tensor,
    vocab_map: VocabMap | None = None,
) -> dict[str, float]:
    """Re-fit the student's final linear layer so student(x) == teacher(x).

    Closed-form, no gradient loop:
      1. Run the probe through the teacher to get target outputs ``Y``.
      2. Run the probe through the student up to (but not including) its last
         linear layer to get features ``F``.
      3. Solve the least-squares system ``[F | 1] @ W = Y`` for weight+bias.
      4. Write ``W`` back into the student's last layer.

    If the student's penultimate width can linearly reconstruct ``Y`` (rank
    condition), this makes agreement exact for *all* inputs, not just the probe.
    Otherwise it returns the best linear fit and the residual is visible via
    ``agreement``.

    Works for sequence models too: the last layer's input features and the
    teacher target are flattened over any leading (batch/position) dims, so the
    LM head of a tiny transformer is fit per token position. For a plain MLP the
    flatten is a no-op.

    ``vocab_map`` (stage 0) reconciles a teacher/student vocab mismatch: the probe
    is in student token ids, the teacher is fed remapped ids, and its logits are
    projected into the student vocab space so the least-squares target matches the
    student head's output width. ``None`` keeps the original same-vocab behaviour.

    Returns the agreement on the probe AFTER alignment.
    """
    last = _last_linear(student)

    # Features feeding the last layer: capture the last layer's input via hook.
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["feat"] = _flatten_logits(inputs[0].detach())

    handle = last.register_forward_hook(hook)
    try:
        if vocab_map is None:
            target = _flatten_logits(teacher(probe))
        else:
            target = _flatten_logits(
                vocab_map.project(teacher(vocab_map.remap_ids(probe)))
            )
        student(probe)  # populates captured["feat"] via the hook
    finally:
        handle.remove()

    feat = captured["feat"]  # (n_rows, in_features)
    ones = torch.ones(feat.shape[0], 1, dtype=feat.dtype, device=feat.device)
    feat_aug = torch.cat([feat, ones], dim=1)  # (n, in_features + 1)

    solution = torch.linalg.lstsq(feat_aug, target).solution  # (in_features+1, out)
    weight = solution[:-1].T.contiguous()  # (out, in_features)
    bias = solution[-1].contiguous()  # (out,)

    last.weight.copy_(weight.to(last.weight.dtype))
    last.bias.copy_(bias.to(last.bias.dtype))

    return agreement(teacher, student, probe, vocab_map)


def _linear_chain(model: torch.nn.Module) -> list[torch.nn.Linear]:
    """Ordered list of ``nn.Linear`` layers inside a flat ``model.net`` Sequential.

    Used by :func:`align_hidden` for the MLP family (``MLP`` / ``ActMLP``), whose
    forward is a single linear/activation chain. Returns ``[]`` for models without
    a ``net`` Sequential (e.g. ``TinyLM``), which signals "no hidden alignment".
    """
    net = getattr(model, "net", None)
    if not isinstance(net, torch.nn.Sequential):
        return []
    return [m for m in net if isinstance(m, torch.nn.Linear)]


def _preactivations(
    chain: list[torch.nn.Linear],
    net: torch.nn.Sequential,
    probe: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For each Linear in ``chain``, capture ``(input_to_linear, pre_activation)``.

    Runs ``probe`` through ``net`` once, recording the tensor fed into each Linear
    and the Linear's raw output (before any following activation). Both flattened
    over leading dims so sequence inputs would still work.
    """
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    h = probe
    for module in net:
        if isinstance(module, torch.nn.Linear):
            z = module(h)
            captured.append((_flatten_logits(h.detach()), _flatten_logits(z.detach())))
            h = z
        else:
            h = module(h)
    return captured


@torch.no_grad()
def align_hidden(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    probe: torch.Tensor,
) -> dict[str, float]:
    """Closed-form *hidden-layer* alignment for nonlinear teachers (stage 2b).

    Head-only :func:`align_output` cannot match a nonlinear teacher exactly,
    because the student's nonlinear features are a *different basis*. This pass
    reshapes that basis: with a forward sweep it re-fits each student hidden
    linear layer so its **pre-activation** regresses (least squares) onto the
    teacher's matched pre-activation on the synthetic probe, then fits the head
    via :func:`align_output`.

    Still fully within Ferry's constraints: no data (synthetic probe only), no
    gradient loop (only ``torch.linalg.lstsq`` solves), deterministic algebra.

    Scope: the flat MLP family (``MLP`` / ``ActMLP``). For matched depth and the
    same activation it lifts held-out agreement dramatically; for depth mismatch
    (fewer student layers than teacher) it improves but cannot fully close the
    gap -- the student has too few layers to track every teacher layer. Models
    without a flat linear chain (e.g. ``TinyLM``) fall back to head-only.

    Returns the agreement on the probe AFTER alignment.
    """
    s_chain = _linear_chain(student)
    t_chain = _linear_chain(teacher)
    # Need a flat chain on both sides with at least one hidden layer to align.
    if len(s_chain) >= 2 and len(t_chain) >= 2:
        n_hidden = min(len(s_chain), len(t_chain)) - 1  # exclude the head
        s_net = student.net  # type: ignore[union-attr]
        for i in range(n_hidden):
            teacher_pre = _preactivations(t_chain, teacher.net, probe)[i][1]  # type: ignore[union-attr]
            student_in = _preactivations(s_chain, s_net, probe)[i][0]
            lin = s_chain[i]
            out_dim = lin.out_features
            # Fit into the student layer's own width: crop/zero-pad the teacher
            # pre-activation target to match (width mismatch is expected).
            target = teacher_pre
            if target.shape[1] >= out_dim:
                target = target[:, :out_dim]
            else:
                pad = torch.zeros(
                    target.shape[0], out_dim - target.shape[1], dtype=target.dtype
                )
                target = torch.cat([target, pad], dim=1)

            ones = torch.ones(student_in.shape[0], 1, dtype=student_in.dtype)
            feat_aug = torch.cat([student_in, ones], dim=1)
            sol = torch.linalg.lstsq(feat_aug, target).solution
            lin.weight.copy_(sol[:-1].T.contiguous().to(lin.weight.dtype))
            lin.bias.copy_(sol[-1].contiguous().to(lin.bias.dtype))

    # Always finish by fitting the head onto the teacher output.
    return align_output(student, teacher, probe)


def distill(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    *,
    in_dim: int | None = None,
    vocab: int | None = None,
    seq: int | None = None,
    steps: int = 400,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    vocab_map: VocabMap | None = None,
) -> dict[str, float]:
    """Stage 3: gradient fine-tune the student to match the teacher (data-free).

    This is what *closes* the limits that closed-form alignment can only narrow:
    nonlinear teachers, depth mismatch, and (for ``TinyLM``) the autoregressive
    compounding of per-token error. Run it AFTER the closed-form stages so the
    student starts from a good warm init and converges fast.

    Data-free by construction: every optimization step draws a **fresh** synthetic
    probe and uses the teacher's own output as the target. Resampling each step
    (instead of reusing one fixed probe) is what makes the fit *generalize* to
    held-out inputs rather than memorize the probe.

    Two input modes (exactly one must be selected):
      * continuous (MLP / ActMLP): pass ``in_dim`` -> probes are ``randn`` vectors.
      * token (TinyLM):            pass ``vocab`` and ``seq`` -> probes are random
                                   token-id sequences.

    ``vocab_map`` (token mode only) reconciles a teacher/student vocab mismatch
    (stage 0): probes are drawn over the *shared* vocabulary in student ids, the
    teacher is fed remapped teacher ids, and its logits are projected into the
    student vocab space so the MSE target matches the student head width. When
    ``None`` the teacher and student are assumed to share a vocabulary.

    Loss is MSE on the raw outputs/logits, which directly drives the same quantity
    ``agreement`` reports (and pulls top-1 along with it).

    Returns the agreement on a fresh held-out probe AFTER distillation.
    """
    token_mode = vocab is not None
    cont_mode = in_dim is not None
    if token_mode == cont_mode:
        raise ValueError(
            "distill: choose exactly one input mode -- either in_dim (continuous) "
            "or vocab+seq (token), not both/neither"
        )
    if token_mode and seq is None:
        raise ValueError("distill: token mode requires both vocab and seq")
    if vocab_map is not None and not token_mode:
        raise ValueError("distill: vocab_map only applies to token (vocab+seq) mode")

    def make_probe(step: int) -> torch.Tensor:
        if token_mode:
            if vocab_map is not None:
                return shared_token_probe(batch, seq, vocab_map, seed=seed + 1 + step)  # type: ignore[arg-type]
            return token_probe(batch, seq, vocab, seed=seed + 1 + step)  # type: ignore[arg-type]
        return synthetic_probe(batch, in_dim, seed=seed + 1 + step)  # type: ignore[arg-type]

    def make_target(probe: torch.Tensor) -> torch.Tensor:
        if vocab_map is not None:
            return vocab_map.project(teacher(vocab_map.remap_ids(probe)))
        return teacher(probe)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    student.train()
    for step in range(steps):
        probe = make_probe(step)
        with torch.no_grad():
            target = make_target(probe)
        out = student(probe)
        loss = torch.nn.functional.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    student.eval()

    # Held-out evaluation on a probe unseen during the loop (negative seed offset).
    if token_mode and vocab_map is not None:
        eval_probe = shared_token_probe(batch, seq, vocab_map, seed=seed - 99)  # type: ignore[arg-type]
    elif token_mode:
        eval_probe = token_probe(batch, seq, vocab, seed=seed - 99)  # type: ignore[arg-type]
    else:
        eval_probe = synthetic_probe(batch, in_dim, seed=seed - 99)  # type: ignore[arg-type]
    return agreement(teacher, student, eval_probe, vocab_map)


# --------------------------------------------------------------------------- #
# toy demo: two MLPs with different widths/depths, no data involved
# --------------------------------------------------------------------------- #
class MLP(torch.nn.Module):
    """Minimal *linear* MLP (no activations) so the demo has tensors to transfer.

    Because every layer is linear, the whole network is an affine map. That is
    what lets ``align_output`` reach an *exact* same-answer guarantee for any
    input once the student is wide enough (see the rank condition in the module
    docstring): an affine teacher can be linearly reconstructed from an affine
    student's penultimate features.
    """

    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(a, b))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActMLP(torch.nn.Module):
    """A genuinely *nonlinear* MLP: Linear layers with activations between them.

    This is the "more complex" model. The output head stays linear (so
    ``align_output`` still has a final ``nn.Linear`` to re-fit), but the body is
    nonlinear. That matters: with a nonlinear teacher, matching on a finite
    probe no longer implies matching everywhere, because the student's nonlinear
    features are a *different* basis than the teacher's. The exact same-answer
    guarantee therefore weakens to a best linear fit -- and Ferry reports the
    held-out residual honestly instead of pretending it is zero.
    """

    _ACTS = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
    }

    def __init__(self, sizes: list[int], act: str = "relu") -> None:
        super().__init__()
        if act not in self._ACTS:
            raise ValueError(f"unknown activation {act!r}; choose from {list(self._ACTS)}")
        make_act = self._ACTS[act]
        layers: list[torch.nn.Module] = []
        n_linear = len(sizes) - 1
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(torch.nn.Linear(a, b))
            if i < n_linear - 1:  # activation between hidden layers, not on the head
                layers.append(make_act())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyBlock(torch.nn.Module):
    """One pre-norm transformer block: self-attention + GELU MLP, both residual."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(torch.nn.Module):
    """A tiny, self-contained GPT-style language model -- the "LLM-like" demo.

    Pure ``torch`` (no ``transformers``, no downloads, no data): token + position
    embeddings -> stacked attention/MLP blocks -> final LayerNorm -> linear LM
    head over a small vocabulary. ``forward`` returns logits ``(batch, seq, vocab)``.

    Ferry applies unchanged: stage 1 transfers every named tensor (attention
    projections, MLP, embeddings, head) by name; stage 2 re-fits the LM head
    (the last ``nn.Linear``) per token position. Because the model is deeply
    nonlinear, the exact same-answer guarantee does not hold -- and, worse than a
    one-shot classifier, per-token logit residuals *compound across autoregressive
    generation steps*. Ferry shows that decay honestly (see ``_demo_llm_like``).
    """

    def __init__(
        self, vocab: int, dim: int, heads: int, layers: int, seq: int
    ) -> None:
        super().__init__()
        self.seq = seq
        self.tok = torch.nn.Embedding(vocab, dim)
        self.pos = torch.nn.Embedding(seq, dim)
        self.blocks = torch.nn.Sequential(*[_TinyBlock(dim, heads) for _ in range(layers)])
        self.lnf = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        x = self.lnf(self.blocks(x))
        return self.head(x)  # (batch, seq, vocab)

    @torch.no_grad()
    def generate(self, ctx: torch.Tensor, steps: int) -> torch.Tensor:
        """Greedy autoregressive decode: append ``steps`` argmax tokens to ``ctx``."""
        ids = ctx.clone()
        for _ in range(steps):
            logits = self(ids[:, -self.seq :])[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def _demo_linear_transfer(in_dim: int, out_dim: int) -> None:
    """Part A: a deeper, structurally different *linear* teacher/student pair.

    The teacher is a 4-layer network; the student is shallower and much
    narrower. Every tensor needs a non-trivial transform (SVD projection or
    crop/pad), yet the exact same-answer guarantee still holds after alignment
    because both networks are affine.
    """
    torch.manual_seed(0)
    teacher = MLP([in_dim, 128, 96, 64, out_dim])  # 4 linear layers
    student = MLP([in_dim, 80, 48, out_dim])  # shallower + narrower

    new_sd, results = transfer(teacher.state_dict(), student.state_dict())
    student.load_state_dict({**student.state_dict(), **{
        k: v for k, v in new_sd.items() if k in student.state_dict()
        and v.shape == student.state_dict()[k].shape
    }})

    print("== Ferry stage 1: weight transfer (deeper linear teacher) ==")
    print(f"teacher: MLP[{in_dim},128,96,64,{out_dim}]  student: MLP[{in_dim},80,48,{out_dim}]")
    for k, v in report(results, student.state_dict()).items():
        print(f"{k:20s}: {v}")

    probe = synthetic_probe(512, in_dim, seed=1)
    before = agreement(teacher, student, probe)
    after = align_output(student, teacher, probe)
    held = agreement(teacher, student, synthetic_probe(512, in_dim, seed=2))

    print("\n== Ferry stage 2: output alignment (synthetic probe) ==")
    print(f"{'before align':16s}: {before}")
    print(f"{'after  align':16s}: {after}")
    print(f"{'held-out probe':16s}: {held}")


def _demo_capacity_sweep(in_dim: int, out_dim: int) -> None:
    """Part B: same answer is guaranteed only if the student is wide enough."""
    print("\n== capacity condition (penultimate width vs out_dim=10) ==")
    for width in (4, 8, 10, 16, 48):
        torch.manual_seed(0)
        t = MLP([in_dim, 128, 96, 64, out_dim])
        s = MLP([in_dim, width, out_dim])
        align_output(s, t, synthetic_probe(512, in_dim, seed=1))
        a = agreement(t, s, synthetic_probe(512, in_dim, seed=2))
        verdict = "match" if a["top1_agree"] > 0.999 else "bottleneck"
        print(
            f"width={width:3d}  top1_agree={a['top1_agree']:.3f}  "
            f"mse={a['mse']:.2e}  -> {verdict}"
        )


def _demo_nonlinear_limit(in_dim: int, out_dim: int) -> None:
    """Part C: a *nonlinear* teacher -- closed-form lifts it, distill CLOSES it.

    Three columns make the progression explicit on a genuinely nonlinear teacher
    (ActMLP), with a depth-matched student:
      * head-only  : closed-form last-layer fit -- only a best linear fit;
      * hidden+head: stage 2b closed-form hidden alignment -- jumps to ~0.97;
      * +distill   : stage 3 gradient fine-tune on fresh synthetic probes --
                     closes the gap to ~0.99 (the limit is no longer a wall).
    """
    probe = synthetic_probe(512, in_dim, seed=1)
    held = synthetic_probe(512, in_dim, seed=2)

    print("\n== nonlinear teacher: closed-form lifts, distill closes (held-out top1) ==")
    print(f"teacher: ActMLP[{in_dim},96,64,{out_dim}]  student: ActMLP[{in_dim},128,96,{out_dim}]")
    student_sizes = [in_dim, 128, 96, out_dim]  # depth-matched, adequate capacity
    for act in ("relu", "gelu", "tanh"):
        torch.manual_seed(0)
        t = ActMLP([in_dim, 96, 64, out_dim], act=act)
        # Same student init for all three methods (re-seed before each build).
        torch.manual_seed(1)
        s_head = ActMLP(student_sizes, act=act)
        torch.manual_seed(1)
        s_hidden = ActMLP(student_sizes, act=act)
        torch.manual_seed(1)
        s_distill = ActMLP(student_sizes, act=act)

        align_output(s_head, t, probe)
        base = agreement(t, s_head, held)["top1_agree"]
        align_hidden(s_hidden, t, probe)
        lifted = agreement(t, s_hidden, held)["top1_agree"]
        align_hidden(s_distill, t, probe)  # warm start
        closed = distill(s_distill, t, in_dim=in_dim, steps=500)["top1_agree"]
        print(
            f"act={act:4s}  head-only={base:.3f} -> hidden+head={lifted:.3f} "
            f"-> +distill={closed:.3f}"
        )


def _demo_llm_like() -> None:
    """Part D: an "LLM-like" tiny transformer -- and the autoregressive limit.

    Ferry's two stages apply unchanged to a small GPT-style model:
      * stage 1 transfers every named tensor (attention q/k/v/o, MLP, embeddings,
        LM head) into a shallower/narrower student;
      * stage 2 re-fits the student's LM head per token position on a synthetic
        token probe.

    Closed-form last-layer alignment alone leaves a large per-token residual that
    COMPOUNDS under greedy autoregressive generation. Stage 3 ``distill`` (gradient
    fine-tune on fresh synthetic token probes) drives the per-token match far up
    and, because per-token error shrinks, the generation decay flattens markedly.
    """
    torch.manual_seed(0)
    vocab, seq = 64, 12
    teacher = TinyLM(vocab, dim=64, heads=4, layers=3, seq=seq)
    student = TinyLM(vocab, dim=48, heads=4, layers=2, seq=seq)

    new_sd, results = transfer(teacher.state_dict(), student.state_dict())
    student.load_state_dict({**student.state_dict(), **{
        k: v for k, v in new_sd.items() if k in student.state_dict()
        and v.shape == student.state_dict()[k].shape
    }})

    print("\n== LLM-like: tiny transformer (TinyLM) ==")
    print(f"teacher: TinyLM(vocab={vocab},dim=64,heads=4,layers=3)  "
          f"student: dim=48,layers=2")
    print(f"stage-1 transfer: {report(results, student.state_dict())}")

    probe = token_probe(256, seq, vocab, seed=1)
    held = token_probe(256, seq, vocab, seed=2)
    before = agreement(teacher, student, held)
    align_output(student, teacher, probe)  # stage 2: re-fit LM head per token pos
    after = agreement(teacher, student, held)

    def gen_curve(tag: str) -> None:
        ctx = token_probe(64, 4, vocab, seed=7)
        cells = []
        for steps in (1, 2, 4, 6, 8):
            gen_t = teacher.generate(ctx, steps)[:, ctx.shape[1]:]
            gen_s = student.generate(ctx, steps)[:, ctx.shape[1]:]
            cells.append(f"s{steps}={(gen_t == gen_s).float().mean().item():.3f}")
        print(f"  generation token-match {tag}: " + "  ".join(cells))

    print(f"{'per-token before     ':22s}: {before}")
    print(f"{'per-token stage-2    ':22s}: {after}")
    gen_curve("(stage-2, decays)   ")

    # Stage 3: gradient distillation on fresh synthetic token probes.
    distilled = distill(student, teacher, vocab=vocab, seq=seq, steps=800, lr=3e-4)
    print(f"{'per-token +distill   ':22s}: {distilled}")
    gen_curve("(+distill, flatter) ")


def _demo_vocab_mismatch() -> None:
    """Part E: teacher and student with DIFFERENT vocabularies (stage 0).

    A language model emits a distribution over its own vocabulary, so a teacher
    with a 64-token vocab and a student with a 48-token vocab cannot be aligned
    head-to-head: the LM-head output widths differ and token id ``j`` denotes a
    different token on each side. Ferry first reconciles the two vocab spaces
    (``reconcile_vocab`` -> ``VocabMap``), then runs the usual stages with that
    map: probes are drawn over the shared vocabulary, the teacher is fed remapped
    ids, and its logits are projected into the student's vocab space.
    """
    torch.manual_seed(0)
    seq = 12
    vocab_t, vocab_s = 64, 48
    teacher = TinyLM(vocab_t, dim=64, heads=4, layers=3, seq=seq)
    student = TinyLM(vocab_s, dim=48, heads=4, layers=2, seq=seq)

    new_sd, results = transfer(teacher.state_dict(), student.state_dict())
    student.load_state_dict({**student.state_dict(), **{
        k: v for k, v in new_sd.items() if k in student.state_dict()
        and v.shape == student.state_dict()[k].shape
    }})

    print("\n== LLM vocab mismatch: stage 0 vocabulary reconciliation ==")
    print(f"teacher: TinyLM(vocab={vocab_t},dim=64,layers=3)  "
          f"student: vocab={vocab_s},dim=48,layers=2")

    # Without stage 0 the heads have different widths -> alignment is undefined.
    vt = _last_linear(teacher).out_features
    vs = _last_linear(student).out_features
    print(f"naive: teacher head emits {vt} logits, student emits {vs} -- "
          "not comparable (no shared token axis)")

    vmap = reconcile_vocab(student, teacher)  # shared-prefix correspondence
    shared = int((vmap.t_for_s >= 0).sum().item())
    print(f"stage-0 VocabMap: {shared}/{vs} student tokens mapped to teacher ids; "
          f"projection {tuple(vmap.projection.shape)} (V_t x V_s)")
    print(f"stage-1 transfer: {report(results, student.state_dict())}")

    probe = shared_token_probe(256, seq, vmap, seed=1)
    held = shared_token_probe(256, seq, vmap, seed=2)
    before = agreement(teacher, student, held, vmap)
    align_output(student, teacher, probe, vmap)  # stage 2 in reconciled space
    after = agreement(teacher, student, held, vmap)
    distilled = distill(
        student, teacher, vocab=vocab_s, seq=seq, steps=800, lr=3e-4, vocab_map=vmap
    )

    print(f"{'per-token before     ':22s}: {before}")
    print(f"{'per-token stage-2    ':22s}: {after}")
    print(f"{'per-token +distill   ':22s}: {distilled}")


def _scrambled_vocab_map(vocab_s: int, vocab_t: int, n_shared: int, seed: int):
    """A *non-trivial* student->teacher vocab map: random ids in random slots.

    The built-in default (`reconcile_vocab`) is the shared-prefix map
    (`arange(min)`), which is unrealistically clean -- token id ``j`` already
    means roughly the same thing on both sides. A real tokenizer pairing is a
    *scramble*: student token ``j`` matches an arbitrary teacher token, and some
    student tokens (here ``vocab_s - n_shared`` of them) have no teacher match at
    all (``-1``). This builder produces that worst-case correspondence so the
    demo/tests exercise the genuinely hard case.
    """
    g = torch.Generator().manual_seed(seed)
    t_for_s = torch.full((vocab_s,), -1, dtype=torch.long)
    teacher_ids = torch.randperm(vocab_t, generator=g)[:n_shared]
    student_slots = torch.randperm(vocab_s, generator=g)[:n_shared]
    t_for_s[student_slots] = teacher_ids
    return build_vocab_map(t_for_s, vocab_t)


def _demo_combined_mismatch() -> None:
    """Part F: the realistic worst case -- vocab, depth, AND width all differ.

    Earlier parts isolate one axis at a time (Part C varies depth/width on a
    nonlinear MLP; Part E varies vocab with a *clean* shared-prefix map). This
    part stacks every transformer-level mismatch at once on ``TinyLM``:
      * LM-head / vocabulary differ (V 72 -> 48),
      * middle-layer depth differs (4 -> 2 blocks),
      * hidden width differs (dim 80 -> 40),
    and, crucially, the vocab correspondence is a *scrambled, partial* map (only
    40 of 48 student tokens have a teacher match, in arbitrary positions) rather
    than the clean shared prefix. That makes the pre-alignment agreement collapse
    to near zero -- the honest baseline for "two genuinely different LMs" -- and
    shows how far the staged pipeline recovers it.
    """
    torch.manual_seed(0)
    seq = 12
    vocab_t, vocab_s = 72, 48
    teacher = TinyLM(vocab_t, dim=80, heads=4, layers=4, seq=seq)
    student = TinyLM(vocab_s, dim=40, heads=4, layers=2, seq=seq)

    new_sd, results = transfer(teacher.state_dict(), student.state_dict())
    student.load_state_dict({**student.state_dict(), **{
        k: v for k, v in new_sd.items() if k in student.state_dict()
        and v.shape == student.state_dict()[k].shape
    }})

    print("\n== combined mismatch: vocab + depth + width all differ (scrambled map) ==")
    print(f"teacher: TinyLM(vocab={vocab_t},dim=80,layers=4)  "
          f"student: vocab={vocab_s},dim=40,layers=2")

    n_shared = 40  # only 40/48 student tokens map to a teacher token; 8 are -1
    vmap = _scrambled_vocab_map(vocab_s, vocab_t, n_shared, seed=5)
    print(f"stage-0 VocabMap: {n_shared}/{vocab_s} student tokens mapped "
          f"(scrambled, partial); projection {tuple(vmap.projection.shape)} (V_t x V_s)")
    print(f"stage-1 transfer: {report(results, student.state_dict())}")

    probe = shared_token_probe(256, seq, vmap, seed=1)
    held = shared_token_probe(256, seq, vmap, seed=2)
    before = agreement(teacher, student, held, vmap)
    align_output(student, teacher, probe, vmap)  # stage 2 in reconciled space
    after = agreement(teacher, student, held, vmap)
    distilled = distill(
        student, teacher, vocab=vocab_s, seq=seq, steps=800, lr=3e-4, vocab_map=vmap
    )

    print(f"{'per-token before     ':22s}: {before}")
    print(f"{'per-token stage-2    ':22s}: {after}")
    print(f"{'per-token +distill   ':22s}: {distilled}")


def _demo() -> None:
    in_dim, out_dim = 32, 10
    _demo_linear_transfer(in_dim, out_dim)
    _demo_capacity_sweep(in_dim, out_dim)
    _demo_nonlinear_limit(in_dim, out_dim)
    _demo_llm_like()
    _demo_vocab_mismatch()
    _demo_combined_mismatch()


if __name__ == "__main__":
    _demo()
