"""Closed-form embed *basis alignment* for the byte-composed Aster student (b').

Why this file exists
--------------------
``transfer_gemma_to_aster.py --embed-byte-compose`` makes every Korean embed row
**non-zero** (reachability): the Korean next-token probability mass jumps from ~2%
to ~91%. But that mass is *diffuse* -- under greedy / top-k decoding a structural
token (space, punctuation) still beats every individual Korean token, because the
seeded rows live in **Gemma's embedding singular basis**, which is NOT the basis
Aster's hidden state ``h_final`` actually rotates into. ``logit[j] = h_final .
embed[j]`` is therefore (near-)random for the seeded rows: present but unaligned.

This is the *alignment* half of "reachability vs alignment". We fix it the only
way the data-free constraint allows -- a **closed-form, gradient-free rotation**
in the spirit of Ferry Stage 2b:

  1. The ~19880 *whole-token* matched students (``t_for_s >= 0``) are anchors: for
     them we DO know where their embed row should point in Aster's output space --
     namely wherever best reproduces the real Gemma teacher's logit for that token.
  2. Run synthetic **shared-token probes** through the student -> collect the tied-
     head input features ``F = h_final`` (``AsterForCausalLM.final_hidden``). Run the
     same probes (remapped) through the real Gemma teacher -> gather the teacher
     logit columns of the anchor tokens. Least-squares solve ``F @ E_fit^T = T``
     gives ``E_fit`` = the anchor embed rows Aster *should* have (in its own basis).
  3. Orthogonal **Procrustes**: find the rotation ``R`` (d x d, ``R^T R = I``)
     minimizing ``|| E_seed_anchor @ R - E_fit_anchor ||_F``. ``R`` maps the whole
     Gemma-projected embed basis onto Aster's output basis.
  4. Apply the SAME ``R`` to ALL rows: ``E_aligned = E_seed @ R``. This works for
     the byte-composed Korean rows too because they were built from the SAME
     projected-Gemma matrix (``transfer_embed``: each byte row is a mean of
     ``proj`` rows), so they share the anchors' coordinate system. An *orthogonal*
     R preserves every row norm exactly -> the byte-composition norm-equalization
     (what lets Korean compete in greedy) is kept intact.

Honest scope (what this can and cannot do)
------------------------------------------
* Fixes failure point (1) hidden-basis mismatch -- a real, closed-form partial
  alignment. It can de-diffuse the Korean mass and improve teacher-logit
  agreement on the anchors.
* Does NOT fix (2) byte-ORDER loss (the byte-mean seed is orderless) or (3) the
  absent sequential/grammar signal (that needs training/KD). So this is a
  *partial* alignment: expect better ranking, NOT fluency. Measured, not claimed.
* Tied-embed circularity: rotating ``embed`` also rotates the *input* embeddings,
  so ``F`` shifts slightly after one shot. ``--iters`` re-fits a few times
  (each R orthogonal, product orthogonal). Default 1 (one honest shot).

Hard constraints (inherited): CPU-only (GPU forbidden, DEC-007), data-free
(synthetic probes + tokenizer vocab tables only -- no dataset, no gradient loop;
only ``lstsq`` / ``svd``), the live aster-1b checkpoint is never touched (output
is a NEW file; only ``v2.embed.weight`` differs from the input, the other 235
tensors are copied byte-for-byte).

Usage:
    python align_aster_embed.py            # bc -> bc-aligned, default knobs
    python align_aster_embed.py --iters 3 --n-batches 24
"""

from __future__ import annotations

import argparse
import json
import os

# CPU-only, GPU forbidden (DEC-007). Must precede torch device selection.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from ferry import agreement, shared_token_probe  # noqa: E402
from ferry_aster import (  # noqa: E402
    DEFAULT_TOKENIZER,
    AsterConfig,
    AsterForCausalLM,
    SparseVocabMap,
    load_aster_weights,
    load_gemma_teacher,
)
from transfer_gemma_to_aster import (  # noqa: E402
    _aster_decode,
    _resolve_snapshot,
    build_vocab_map,
    load_safetensors,
    save_safetensors,
)

BC_PARAMS = "./test_output/aster-1b-from-gemma-2-2b-embedmap-bc/params.safetensors"


# --------------------------------------------------------------------------- #
# Korean-token mask (for the headline reachability/ranking measurement)
# --------------------------------------------------------------------------- #
def korean_id_mask(tok_path: str, vocab_size: int) -> torch.Tensor:
    """Boolean ``(vocab_size,)`` mask: True where the student token decodes to
    a string containing a Hangul syllable (U+AC00..U+D7A3)."""
    with open(tok_path) as f:
        vocab = json.load(f)["model"]["vocab"]
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tok, sid in vocab.items():
        if not (0 <= sid < vocab_size):
            continue
        text = _aster_decode(tok)
        if text and any("\uac00" <= ch <= "\ud7a3" for ch in text):
            mask[sid] = True
    return mask


@torch.no_grad()
def measure_korean(model: AsterForCausalLM, ids: list[int], kr_mask: torch.Tensor) -> dict:
    """Next-token Korean reachability for a prompt: Korean prob-mass, greedy top1
    Korean-ness, and the rank of the best Korean token among all logits."""
    logits = model(torch.tensor([ids], dtype=torch.long))[0, -1]  # (V,)
    probs = torch.softmax(logits, dim=-1)
    kr_mass = float(probs[kr_mask].sum())
    top1 = int(logits.argmax())
    order = torch.argsort(logits, descending=True)
    kr_positions = kr_mask[order]  # bool in rank order
    best_rank = int(torch.nonzero(kr_positions, as_tuple=False)[0, 0]) if kr_positions.any() else -1
    return {"kr_mass": kr_mass, "top1_is_kr": bool(kr_mask[top1]), "best_kr_rank": best_rank}


# --------------------------------------------------------------------------- #
# closed-form alignment
# --------------------------------------------------------------------------- #
@torch.no_grad()
def accumulate_normal_eqs(
    student: AsterForCausalLM,
    teacher,
    vmap: SparseVocabMap,
    teacher_ids_for_whole: torch.Tensor,
    cfg: AsterConfig,
    batch: int,
    seq: int,
    n_batches: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate ``A = F^T F`` (d,d) and ``B = F^T T_whole`` (d, M) over fresh
    synthetic shared-token probes.

    ``F`` = student ``final_hidden`` (the tied-head input features); ``T_whole`` =
    the teacher's projected logit columns for the whole-token anchors (gathered at
    ``teacher_ids_for_whole``). Float64 normal equations for a well-conditioned
    closed-form solve. Data-free: probes are random shared-vocab token ids.
    """
    d = cfg.d_model
    m = int(teacher_ids_for_whole.numel())
    a_mat = torch.zeros(d, d, dtype=torch.float64)
    b_mat = torch.zeros(d, m, dtype=torch.float64)
    n_pos = 0
    for b in range(n_batches):
        probe = shared_token_probe(batch, seq, vmap, seed=1000 + b)
        feat = student.final_hidden(probe).reshape(-1, d).double()  # (rows, d)
        tlog = teacher(vmap.remap_ids(probe))  # (batch, seq, V_t)
        tlog = tlog.reshape(-1, tlog.shape[-1])  # (rows, V_t)
        t_whole = tlog.index_select(1, teacher_ids_for_whole).double()  # (rows, M)
        a_mat += feat.t() @ feat
        b_mat += feat.t() @ t_whole
        n_pos += feat.shape[0]
    return a_mat, b_mat, n_pos


def procrustes_rotation(
    x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Orthogonal Procrustes: ``R`` (d,d, ``R^T R = I``) minimizing the (optionally
    weighted) ``sum_i w_i ||x_i R - y_i||^2``.

    ``x``, ``y`` are ``(M, d)``. The optimum maximises ``Tr(R^T M_cross)`` with
    ``M_cross = X^T diag(w) Y = sum_i w_i x_i^T y_i = U S V^T`` -> ``R = U V^T``.
    ``weights=None`` reduces to the uniform ``M_cross = x^T y`` (DEC-013 behaviour).
    Per-anchor weights let Korean anchors (a 9% minority) steer the rotation instead
    of being drowned by the non-Korean majority. Scale-invariant in ``y``.
    """
    xd, yd = x.double(), y.double()
    if weights is not None:
        xd = xd * weights.double().unsqueeze(1)  # row-scale x by w_i
    m_cross = xd.t() @ yd  # (d, d) == sum_i w_i x_i^T y_i
    u, _s, vh = torch.linalg.svd(m_cross)
    return (u @ vh).float()


@torch.no_grad()
def align_once(
    student: AsterForCausalLM,
    teacher,
    vmap: SparseVocabMap,
    whole_ids: torch.Tensor,
    teacher_ids_for_whole: torch.Tensor,
    cfg: AsterConfig,
    args: argparse.Namespace,
    anchor_weights: torch.Tensor | None = None,
) -> dict:
    """One closed-form alignment step. Mutates ``student.embed`` in place by an
    orthogonal rotation derived from the whole-token anchors. ``anchor_weights``
    (``(M,)``, optional) upweights Korean anchors in the Procrustes fit. Returns
    diagnostics."""
    d = cfg.d_model
    e_seed = student.embed.detach().clone()  # (V, d) current basis

    a_mat, b_mat, n_pos = accumulate_normal_eqs(
        student, teacher, vmap, teacher_ids_for_whole, cfg,
        args.batch, args.seq, args.n_batches,
    )
    lam = args.ridge * float(a_mat.diagonal().mean())
    a_reg = a_mat + lam * torch.eye(d, dtype=torch.float64)
    e_fit_whole = torch.linalg.solve(a_reg, b_mat).t().float()  # (M, d) anchor target

    x = e_seed.index_select(0, whole_ids)  # (M, d) seed anchor rows
    rot = procrustes_rotation(x, e_fit_whole, anchor_weights)  # (d, d) orthogonal

    e_aligned = e_seed @ rot  # (V, d) -- rotate ALL rows by the same R
    student.embed.copy_(e_aligned)

    xr = x @ rot
    resid = float((xr - e_fit_whole).norm() / (e_fit_whole.norm() + 1e-9))
    row_cos = float(torch.nn.functional.cosine_similarity(xr, e_fit_whole, dim=1).mean())
    change = float((e_aligned - e_seed).norm() / (e_seed.norm() + 1e-9))
    return {
        "n_pos": n_pos, "ridge": lam,
        "procrustes_residual": resid, "anchor_row_cosine": row_cos,
        "embed_rel_change": change,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _teacher_tok(repo: str) -> str:
    return os.path.join(_resolve_snapshot(repo), "tokenizer.json")


def run_align(args: argparse.Namespace) -> None:
    cfg = AsterConfig()
    print(f"[1/6] loading byte-composed student: {args.student_params}")
    student = AsterForCausalLM(cfg).eval()
    load_aster_weights(student, args.student_params)

    print(f"[2/6] loading real teacher: {args.teacher} (CPU f32)")
    teacher = load_gemma_teacher(args.teacher)

    print(f"[3/6] building vocab map: {args.student_tokenizer} <-> {args.teacher}")
    t_for_s_list, stats = build_vocab_map(args.student_tokenizer, _teacher_tok(args.teacher))
    t_for_s = torch.tensor(t_for_s_list, dtype=torch.long)
    vmap = SparseVocabMap(t_for_s, size_t=stats["teacher_vocab"])
    whole_ids = torch.nonzero(t_for_s >= 0, as_tuple=False).squeeze(1)
    teacher_ids_for_whole = t_for_s[whole_ids]
    print(f"      anchors (whole-token matched): {whole_ids.numel()} "
          f"(korean {stats['korean_matched']})")

    kr_mask = korean_id_mask(args.student_tokenizer, cfg.vocab_size)
    # Korean-weighted Procrustes: upweight the Korean whole-token anchors so the
    # (single, global, orthogonal) rotation respects Korean geometry instead of
    # being dominated by the ~91% non-Korean anchor majority. kr_weight=1.0 is the
    # exact DEC-013 uniform behaviour.
    anchor_is_kr = kr_mask.index_select(0, whole_ids)  # (M,) bool
    anchor_weights = torch.where(
        anchor_is_kr, torch.tensor(float(args.kr_weight)), torch.tensor(1.0)
    )
    print(f"      korean anchors upweighted x{args.kr_weight} "
          f"({int(anchor_is_kr.sum())}/{whole_ids.numel()} anchors)")
    from tokenizers import Tokenizer
    prompt_ids = Tokenizer.from_file(args.student_tokenizer).encode(args.prompt).ids

    held = shared_token_probe(args.batch, args.seq, vmap, seed=-99)
    before_agree = agreement(teacher, student, held, vmap)
    before_kr = measure_korean(student, prompt_ids, kr_mask)
    print(f"[4/6] BEFORE: anchor agree top1={before_agree['top1_agree']:.4f} "
          f"mse={before_agree['mse']:.3f} cos={before_agree['cosine']:.4f} | "
          f"kr_mass={before_kr['kr_mass']:.4f} top1_kr={before_kr['top1_is_kr']} "
          f"best_kr_rank={before_kr['best_kr_rank']}")

    print(f"[5/6] closed-form Procrustes alignment: iters={args.iters} "
          f"probes={args.batch}x{args.seq}x{args.n_batches}")
    for it in range(args.iters):
        diag = align_once(student, teacher, vmap, whole_ids, teacher_ids_for_whole,
                          cfg, args, anchor_weights)
        ag = agreement(teacher, student, held, vmap)
        kr = measure_korean(student, prompt_ids, kr_mask)
        print(f"      iter {it+1}: resid={diag['procrustes_residual']:.4f} "
              f"anchor_cos={diag['anchor_row_cosine']:.4f} "
              f"embed_change={diag['embed_rel_change']:.4f} || "
              f"agree top1={ag['top1_agree']:.4f} cos={ag['cosine']:.4f} | "
              f"kr_mass={kr['kr_mass']:.4f} top1_kr={kr['top1_is_kr']} "
              f"best_kr_rank={kr['best_kr_rank']}")

    after_agree = agreement(teacher, student, held, vmap)
    after_kr = measure_korean(student, prompt_ids, kr_mask)

    # Write a NEW checkpoint: copy the input params, replace ONLY the embed row.
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "params.safetensors")
    print(f"[6/6] saving aligned weights (embed-only delta): {out_path}")
    sd = load_safetensors(args.student_params)
    sd["v2.embed.weight"] = student.embed.detach().float().contiguous()
    save_safetensors(sd, out_path)

    report = {
        "input": args.student_params, "teacher": args.teacher,
        "method": "orthogonal_procrustes_embed_basis_alignment",
        "anchors": int(whole_ids.numel()), "korean_anchors": stats["korean_matched"],
        "probe": {"batch": args.batch, "seq": args.seq, "n_batches": args.n_batches,
                  "iters": args.iters, "ridge": args.ridge, "kr_weight": args.kr_weight},
        "before": {"anchor_agreement": before_agree, "korean": before_kr},
        "after": {"anchor_agreement": after_agree, "korean": after_kr},
        "prompt": args.prompt,
    }
    with open(os.path.join(args.out, "align_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("-" * 72)
    print("HONEST READING:")
    print(f"  anchor teacher-agreement: top1 {before_agree['top1_agree']:.4f} -> "
          f"{after_agree['top1_agree']:.4f}, cosine {before_agree['cosine']:.4f} -> "
          f"{after_agree['cosine']:.4f}")
    print(f"  korean next-token (prompt {args.prompt!r}): mass "
          f"{before_kr['kr_mass']:.4f} -> {after_kr['kr_mass']:.4f}, "
          f"best_kr_rank {before_kr['best_kr_rank']} -> {after_kr['best_kr_rank']}, "
          f"greedy_top1_kr {before_kr['top1_is_kr']} -> {after_kr['top1_is_kr']}")
    print("  R is ORTHOGONAL -> every row norm preserved (byte-comp norm balance kept).")
    print("  This is PARTIAL alignment: byte-order loss + absent grammar signal remain")
    print("  (need training/KD). Improved ranking is NOT a fluency claim.")
    print(f"  done. compare slm-cli chat: {args.student_params}  vs  {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-params", default=BC_PARAMS,
                    help="byte-composed transferred params to align (input)")
    ap.add_argument("--teacher", default="google/gemma-2-2b")
    ap.add_argument("--student-tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out",
                    default="./test_output/aster-1b-from-gemma-2-2b-embedmap-bc-aligned")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=16,
                    help="number of fresh synthetic probe batches to accumulate")
    ap.add_argument("--iters", type=int, default=1,
                    help="re-fit the rotation N times (tied-embed circularity); each R "
                         "orthogonal. KEEP 1: measured, iters>=2 over-rotates toward the "
                         "teacher's (anti-Korean) prior on random probes and collapses the "
                         "Korean mass (iter1 best_kr_rank ~9 -> iter3 ~762)")
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="relative Tikhonov ridge on F^T F for the lstsq solve")
    ap.add_argument("--kr-weight", type=float, default=1.0,
                    help="per-anchor weight for Korean whole-token anchors in the "
                         "Procrustes fit (1.0 = uniform DEC-013; >1 lets the 9%% Korean "
                         "anchor minority steer the global rotation)")
    ap.add_argument("--prompt", default="옛날 옛적에 한 마을에")
    args = ap.parse_args()
    run_align(args)


if __name__ == "__main__":
    main()
