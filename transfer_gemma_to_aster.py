"""Pure weight-space initial transfer: Gemma-2 -> Aster aster-1b.

This is a *data-free, training-free* PoC. It takes a real pretrained Gemma-2
checkpoint (teacher) and reshapes its weight tensors onto Aster's aster-1b
student architecture (different depth was avoided by picking gemma-2-2b, which
happens to also have 26 layers; hidden / head_dim / ffn / vocab still differ),
using ONLY the deterministic linear algebra from ``ferry.transform_tensor``
(Copy / CropPad / SvdProject / Skip). No KD, no forward pass, no probes, no
gradients, no dataset, no disk I/O for data. The teacher's own weights are the
only signal.

It is honest about the limits:
  * ``embed`` is Skipped by default (vocab 256000 vs 48000, different tokenizers
    -> the token axes are not comparable; SVD-projecting them would be
    meaningless). With ``--embed-vocab-map`` it is instead seeded for the ~41% of
    student tokens whose *normalized* string matches a teacher token (byte-level
    'Ġ' vs sentencepiece '▁' reconciled). Honest caveat: the seeded rows use the
    embed's own right-singular basis for the hidden squeeze, which is NOT aligned
    with the FFN/attention hidden rotation, so (under weight tying) the logits
    geometry stays inconsistent -- it breaks the embed=0 collapse but is not
    fluent transfer.
  * ``attention`` (q/k/v/o) is SvdProject'd for completeness, but head_dim
    differs (256 vs 96) and the two models use different RoPE bases, so the
    transferred attention weights carry NO usable geometry -- reported, not hidden.
  * ``ffn`` (gate/up/down) is the one block with a matching activation family
    (both GeGLU with gelu-tanh), so its SVD projection is the only part with
    partial semantic meaning.
  * ``norm`` gammas are CropPad'd (RMSNorm scale per channel) -- partial meaning
    on the surviving channels.

Output is written as NEW files under ``./test_output`` -- the live aster-1b
checkpoint at ``artifacts/checkpoints/aster-1b`` (step 3600, still training) is
never touched.

Usage:
    python transfer_gemma_to_aster.py
    python transfer_gemma_to_aster.py --teacher google/gemma-2-2b --out ./test_output
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from dataclasses import dataclass

# Pure-algebra, CPU-only. No CUDA needed for SVD/crop-pad on these sizes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

# Reuse Ferry's deterministic, data-free transforms verbatim -- single source of
# truth for the weight algebra (Copy / CropPad / SvdProject / Skip).
from ferry import _svd_project, transform_tensor  # noqa: E402


# --------------------------------------------------------------------------- #
# Aster aster-1b target spec (from configs/model/pretrain-1b.toml, measured).
# v2 namespace, per-tensor shapes the student checkpoint expects.
# --------------------------------------------------------------------------- #
ASTER_1B = {
    "d_model": 1536,
    "n_layers": 26,
    "n_heads": 16,
    "n_kv_heads": 8,
    "head_dim": 96,
    "ffn_inner": 6144,
    "vocab": 48000,
}


def aster_target_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    """Build the full {tensor_name -> shape} map aster-1b's params expect."""
    d = cfg["d_model"]
    q_dim = cfg["n_heads"] * cfg["head_dim"]
    kv_dim = cfg["n_kv_heads"] * cfg["head_dim"]
    f = cfg["ffn_inner"]
    shapes: dict[str, tuple[int, ...]] = {
        "v2.embed.weight": (cfg["vocab"], d),
        "v2.final_norm.gamma": (d,),
    }
    for i in range(cfg["n_layers"]):
        p = f"v2.blocks.{i}."
        shapes[p + "q.weight"] = (q_dim, d)
        shapes[p + "k.weight"] = (kv_dim, d)
        shapes[p + "v.weight"] = (kv_dim, d)
        shapes[p + "o.weight"] = (d, q_dim)
        shapes[p + "ffn_gate.weight"] = (f, d)
        shapes[p + "ffn_up.weight"] = (f, d)
        shapes[p + "ffn_down.weight"] = (d, f)
        shapes[p + "attn_norm.gamma"] = (d,)
        shapes[p + "ffn_norm.gamma"] = (d,)
    return shapes


# --------------------------------------------------------------------------- #
# Vocabulary reconciliation for embed transfer (Stage 0, opt-in).
#
# The default transfer force-Skips ``v2.embed.weight`` because the two token
# axes are not directly comparable: token id ``j`` means different things on each
# side, the vocab sizes differ (256000 vs 48000), AND the two tokenizers encode
# whitespace differently (Aster is GPT-2 *byte-level* BPE -> 'Ġ' = space, Hangul
# stored as raw-byte chars; Gemma is *SentencePiece* BPE -> '▁' = space, Hangul
# stored literally). So a naive string match only overlaps ~12.8%.
#
# After normalizing BOTH sides to real UTF-8 text (byte-level decode for Aster,
# '▁'->' ' for Gemma) the true overlap is ~41% of Aster's 48000 tokens (incl.
# ~1800 Korean tokens). For those matched tokens we CAN transfer the teacher's
# embedding row -- reshaped on the hidden axis only (the vocab axis is now an
# id->id permutation, not an SVD target). Unmatched tokens stay zero-init (the
# honest 'no teacher signal' state). This is still data-free (no probes, no
# forward): we only read the two tokenizer.json vocab tables off disk.
# --------------------------------------------------------------------------- #
def _byte_level_decoder() -> dict[str, int]:
    """GPT-2 byte-level BPE char->byte table (inverse of bytes_to_unicode)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_C2B = _byte_level_decoder()


def _aster_decode(tok: str) -> str | None:
    """Decode an Aster (byte-level BPE) token string to real UTF-8 text.

    Returns ``None`` if the token's byte sequence is not valid UTF-8 (a partial
    multibyte fragment) -- such tokens have no whole-character meaning to match.
    """
    try:
        return bytes(_C2B[ch] for ch in tok).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def _load_vocab(tok_path: str) -> dict[str, int]:
    with open(tok_path) as f:
        return json.load(f)["model"]["vocab"]


def build_vocab_map(
    student_tok_path: str, teacher_tok_path: str
) -> tuple[list[int], dict]:
    """Build ``t_for_s`` (teacher id per student id, -1 = unmatched) + stats.

    Both vocabularies are normalized to real UTF-8 text so the differing
    whitespace conventions (byte-level 'Ġ' vs sentencepiece '▁') don't block
    matches. The result is a permutation-style map usable to seed Aster's embed
    rows from Gemma's, for the subset of tokens that genuinely coincide.
    """
    s_vocab = _load_vocab(student_tok_path)
    t_vocab = _load_vocab(teacher_tok_path)

    # teacher: normalized-text -> teacher id (first id wins on collision)
    t_norm: dict[str, int] = {}
    for tok, tid in t_vocab.items():
        text = tok.replace("\u2581", " ")
        t_norm.setdefault(text, tid)

    v_s = max(s_vocab.values()) + 1
    t_for_s = [-1] * v_s
    matched = 0
    korean = 0
    for tok, sid in s_vocab.items():
        text = _aster_decode(tok)
        if text is None:
            continue
        tid = t_norm.get(text)
        if tid is None:
            continue
        t_for_s[sid] = tid
        matched += 1
        if any("\uac00" <= ch <= "\ud7a3" for ch in text):
            korean += 1

    stats = {
        "student_vocab": v_s,
        "teacher_vocab": max(t_vocab.values()) + 1,
        "matched": matched,
        "match_frac": round(matched / v_s, 4),
        "korean_matched": korean,
    }
    return t_for_s, stats


def _byte_fallback_table(t_vocab: dict[str, int]) -> dict[int, int]:
    """Map UTF-8 byte value 0..255 -> teacher id, from SentencePiece ``<0xXX>``
    byte-fallback tokens (e.g. ``<0xEC>``)."""
    out: dict[int, int] = {}
    for tok, tid in t_vocab.items():
        if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
            try:
                out[int(tok[3:5], 16)] = tid
            except ValueError:
                pass
    return out


def build_byte_composition(
    student_tok_path: str, teacher_tok_path: str, t_for_s: list[int]
) -> tuple[dict[int, list[int]], dict]:
    """Compose a teacher embed seed for student tokens that have NO whole-token
    match, using the teacher's byte-fallback ``<0xXX>`` tokens.

    Motivation (diagnosed, not assumed): gemma-2-2b tokenizes Korean almost
    entirely via byte fallback -- it holds only ~2300 whole Korean tokens vs
    Aster's ~27000. So whole-token matching (:func:`build_vocab_map`) caps Korean
    embed coverage at ~1800 rows; the other ~25500 Korean rows stay zero and are
    therefore invisible in the tied-embed logits (a zero row can never win an
    argmax). Every UTF-8 byte 0..255 (except 0x09 TAB, absent here) exists as a
    ``<0xXX>`` teacher token with a learned embedding, so ANY decodable Aster
    token can be seeded by the MEAN of its UTF-8 bytes' teacher embeddings.

    This stays data-free (reads the two vocab tables + UTF-8 byte structure only;
    no probe, no forward, no dataset). It is a crude non-zero INIT: the mean
    discards byte ORDER, so it carries Gemma's byte-level signal but NOT
    whole-token semantics. It makes every Korean row distinct and non-zero (so
    Korean can surface in decoding) without claiming fluency.

    Only student ids whose ``t_for_s`` entry is ``-1`` (unmatched) are composed;
    whole-token matches are higher quality and kept as-is. A token is composed
    only if EVERY one of its bytes has a fallback token (else skipped -- honest:
    a partial seed would misrepresent the token). Returns
    ``(byte_comp: {student_id: [teacher_byte_ids]}, stats)``.
    """
    s_vocab = _load_vocab(student_tok_path)
    t_vocab = _load_vocab(teacher_tok_path)
    byte_tok = _byte_fallback_table(t_vocab)

    byte_comp: dict[int, list[int]] = {}
    composed = composed_kr = 0
    for tok, sid in s_vocab.items():
        if 0 <= sid < len(t_for_s) and t_for_s[sid] >= 0:
            continue  # already whole-token matched -- keep the higher-quality row
        text = _aster_decode(tok)
        if text is None:
            continue
        bs = text.encode("utf-8")
        if not all(b in byte_tok for b in bs) or not bs:
            continue  # missing a byte fallback -> cannot fully compose (skip)
        byte_comp[sid] = [byte_tok[b] for b in bs]
        composed += 1
        if any("\uac00" <= ch <= "\ud7a3" for ch in text):
            composed_kr += 1

    stats = {
        "byte_fallback_tokens": len(byte_tok),
        "byte_composed": composed,
        "byte_composed_korean": composed_kr,
    }
    return byte_comp, stats


def transfer_embed(
    teacher_embed: torch.Tensor,
    t_for_s: list[int],
    dst_shape: tuple[int, int],
    byte_comp: dict[int, list[int]] | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Seed the student embed from the teacher, vocab-matched + hidden-projected.

    ``teacher_embed`` is ``(V_t, H_t)``; ``dst_shape`` is ``(V_s, H_s)``. We reduce
    the hidden axis ``H_t -> H_s`` by RIGHT-projecting onto the teacher embedding's
    top-``H_s`` right-singular subspace (``A V_n``, ``V_n`` orthonormal). Crucially
    the vocab axis ``V_t`` is preserved in full -- it is a row index to permute, NOT
    an SVD target. (The generic two-sided ``_svd_project`` would wrongly compress the
    256000-row vocab axis down to ``rank`` and zero-pad the rest, dropping almost all
    tokens.) We then scatter the matched rows into the student matrix by the
    ``t_for_s`` permutation.

    When ``byte_comp`` (``{student_id: [teacher_byte_token_ids]}`` from
    :func:`build_byte_composition`) is supplied, every still-unmatched student row
    listed there is additionally seeded with the MEAN of its UTF-8 bytes' projected
    teacher embeddings, then RESCALED to the mean L2 norm of the whole-token rows.
    The rescale matters: with weight tying the logit is ``h . embed_row``, so a row
    whose norm is far smaller than its peers can never win an argmax -- equalizing
    norm lets the byte-composed (mostly Korean) rows actually compete in greedy
    decoding. Rows neither matched nor composed stay zero (no teacher signal).
    Returns ``(student_embed, n_whole_written, n_byte_written)``.
    """
    v_s, h_s = dst_shape
    v_t, h_t = teacher_embed.shape
    a = teacher_embed.float()
    # Right-only projection: keep all V_t rows, shrink hidden H_t -> H_s via the
    # top-H_s right singular vectors. _, _, vh = svd(A); V_n = vh[:H_s].T; proj=A@V_n.
    if h_t > h_s:
        _u, _s, vh = torch.linalg.svd(a, full_matrices=False)
        v_n = vh[:h_s, :].T  # (H_t, H_s), orthonormal columns
        proj = a @ v_n  # (V_t, H_s)
    elif h_t < h_s:
        proj = torch.zeros((v_t, h_s), dtype=torch.float32)
        proj[:, :h_t] = a
    else:
        proj = a

    # Vectorized scatter: student rows whose teacher id is valid get the matched
    # (projected) teacher row; all others stay zero. No Python per-row loop.
    out = torch.zeros((v_s, h_s), dtype=torch.float32)
    tfs = torch.tensor(t_for_s[:v_s], dtype=torch.long)
    valid = (tfs >= 0) & (tfs < v_t)  # (V_s,) bool mask of matched student rows
    sids = torch.nonzero(valid, as_tuple=False).squeeze(1)  # matched student ids
    tids = tfs[sids]  # corresponding teacher ids
    out[sids] = proj.index_select(0, tids)
    whole_written = int(sids.numel())

    byte_written = 0
    if byte_comp:
        # Target magnitude = mean row-norm of the whole-token seeds (fall back to
        # the projected teacher mean if there were no whole matches at all).
        if whole_written > 0:
            target_norm = float(out[sids].norm(dim=1).mean())
        else:
            target_norm = float(proj.norm(dim=1).mean())
        for sid, ids in byte_comp.items():
            if not (0 <= sid < v_s) or not ids:
                continue
            idx = torch.tensor(ids, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < v_t)]
            if idx.numel() == 0:
                continue
            seed = proj.index_select(0, idx).mean(dim=0)  # (H_s,) byte-mean
            n = float(seed.norm())
            if n > 0 and target_norm > 0:
                seed = seed * (target_norm / n)  # equalize scale vs whole rows
            out[sid] = seed
            byte_written += 1
    return out, whole_written, byte_written


# --------------------------------------------------------------------------- #
# Name mapping: Gemma-2 (HF) -> Aster v2. Per-layer 1:1 because gemma-2-2b and
# aster-1b both have 26 layers. ``kind`` is the semantic honesty tag, NOT the
# transform (the transform is chosen by shape via ferry.transform_tensor).
#   meaningful  : matching activation family / role, projection carries signal
#   partial     : role matches but only surviving channels keep meaning (norms)
#   meaningless : axes are not comparable (vocab / head_dim+RoPE) -> noise
# --------------------------------------------------------------------------- #
@dataclass
class MapRule:
    aster_suffix: str  # e.g. "q.weight" (per-layer) or full name (global)
    gemma_template: str  # {i} filled per layer; no {i} for globals
    semantic: str  # meaningful | partial | meaningless
    per_layer: bool
    force_skip: bool = False  # leave student zero-init regardless of shape


MAP_RULES: list[MapRule] = [
    # --- global ---
    # embed: vocab 256000 vs 48000 with DIFFERENT tokenizers -> token id j means
    # a different token on each side. Both are 2D so transform_tensor would SVD
    # them, but SVD-projecting non-comparable axes only fills slots with noise.
    # Force Skip (zero-init) -- the honest representation of "no signal here".
    MapRule("v2.embed.weight", "model.embed_tokens.weight", "meaningless", False,
            force_skip=True),
    MapRule("v2.final_norm.gamma", "model.norm.weight", "partial", False),
    # --- per-layer attention (head_dim + RoPE differ -> meaningless) ---
    MapRule("q.weight", "model.layers.{i}.self_attn.q_proj.weight", "meaningless", True),
    MapRule("k.weight", "model.layers.{i}.self_attn.k_proj.weight", "meaningless", True),
    MapRule("v.weight", "model.layers.{i}.self_attn.v_proj.weight", "meaningless", True),
    MapRule("o.weight", "model.layers.{i}.self_attn.o_proj.weight", "meaningless", True),
    # --- per-layer FFN (GeGLU both sides -> meaningful) ---
    MapRule("ffn_gate.weight", "model.layers.{i}.mlp.gate_proj.weight", "meaningful", True),
    MapRule("ffn_up.weight", "model.layers.{i}.mlp.up_proj.weight", "meaningful", True),
    MapRule("ffn_down.weight", "model.layers.{i}.mlp.down_proj.weight", "meaningful", True),
    # --- per-layer norms (RMSNorm scale, partial) ---
    # Gemma-2 has 4 norms/layer (sandwich); Aster has 2. Map the role-aligned
    # pair: input_layernorm -> attn_norm, pre_feedforward_layernorm -> ffn_norm.
    MapRule("attn_norm.gamma", "model.layers.{i}.input_layernorm.weight", "partial", True),
    MapRule("ffn_norm.gamma", "model.layers.{i}.pre_feedforward_layernorm.weight", "partial", True),
]


def select_teacher_layers(
    n_student: int, n_teacher: int, strategy: str
) -> list[int]:
    """Map each student layer index 0..n_student-1 to a teacher layer index.

    When the teacher is deeper than the student (e.g. gemma-2-9b has 42 layers,
    aster-1b has 26) we must pick which teacher layers seed the student. Two
    data-free, deterministic strategies:

      * ``uniform``: evenly spaced stride across the full teacher depth
        ``round(i * (n_teacher-1) / (n_student-1))`` -- the standard depth-
        distillation choice; covers the whole network (early + late blocks).
      * ``front``: the first ``n_student`` teacher layers (0..n_student-1);
        simplest, but discards the teacher's upper blocks.

    When ``n_teacher == n_student`` both reduce to identity (the gemma-2-2b
    case). ``n_teacher < n_student`` would need upsampling -- not a case we hit
    (all our teachers are deeper), so we clamp to the last teacher layer.
    """
    if strategy not in ("uniform", "front"):
        raise ValueError(f"unknown layer_select strategy: {strategy!r}")
    if n_student <= 0:
        return []
    if strategy == "front":
        return [min(i, n_teacher - 1) for i in range(n_student)]
    # uniform stride
    if n_student == 1:
        return [0]
    return [
        min(round(i * (n_teacher - 1) / (n_student - 1)), n_teacher - 1)
        for i in range(n_student)
    ]


def build_name_map(
    n_layers: int, n_teacher_layers: int, layer_select: str = "uniform"
) -> list[tuple[str, str, str, bool]]:
    """Expand MAP_RULES into concrete (aster, gemma, semantic, force_skip).

    ``n_layers`` is the student depth (aster-1b = 26). ``n_teacher_layers`` is
    the teacher depth (gemma-2-2b = 26, gemma-2-9b = 42). ``layer_select`` picks
    which teacher layer seeds each student layer when the depths differ.
    """
    teacher_of = select_teacher_layers(n_layers, n_teacher_layers, layer_select)
    out: list[tuple[str, str, str, bool]] = []
    for r in MAP_RULES:
        if not r.per_layer:
            out.append((r.aster_suffix, r.gemma_template, r.semantic, r.force_skip))
            continue
        for i in range(n_layers):
            t = teacher_of[i]  # teacher layer feeding student layer i
            out.append(
                (f"v2.blocks.{i}.{r.aster_suffix}",
                 r.gemma_template.format(i=t), r.semantic, r.force_skip)
            )
    return out


# --------------------------------------------------------------------------- #
# safetensors I/O (header-only read + tensor read; no torch.load, no pickle).
# --------------------------------------------------------------------------- #
_ST_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _read_st_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load a single .safetensors file into a {name -> tensor} dict (CPU)."""
    hdr, base = _read_st_header(path)
    out: dict[str, torch.Tensor] = {}
    with open(path, "rb") as f:
        blob = f.read()
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        dt = _ST_DTYPE[meta["dtype"]]
        s, e = meta["data_offsets"]
        raw = blob[base + s : base + e]
        t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
        out[name] = t
    return out


def load_teacher(repo_or_dir: str) -> dict[str, torch.Tensor]:
    """Load a sharded Gemma-2 checkpoint (all *.safetensors) into one dict."""
    snap = _resolve_snapshot(repo_or_dir)
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {snap}")
    sd: dict[str, torch.Tensor] = {}
    for sh in shards:
        sd.update(load_safetensors(sh))
    return sd


def teacher_num_layers(repo_or_dir: str) -> int:
    """Read ``num_hidden_layers`` from the teacher's config.json (cached snapshot).

    Lets the transfer adapt to teacher depth automatically: gemma-2-2b -> 26
    (1:1 with aster-1b), gemma-2-9b -> 42 (needs layer selection).
    """
    snap = _resolve_snapshot(repo_or_dir)
    cfg_path = os.path.join(snap, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # Gemma-2 is flat; Gemma-3/4 nest under "text_config".
    n = cfg.get("num_hidden_layers")
    if n is None:
        n = cfg.get("text_config", {}).get("num_hidden_layers")
    if n is None:
        raise ValueError(f"could not find num_hidden_layers in {cfg_path}")
    return int(n)


def _resolve_snapshot(repo_or_dir: str) -> str:
    """Accept a local dir, or an HF repo id whose snapshot is already cached."""
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    safe = "models--" + repo_or_dir.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(cache, safe, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(
            f"{repo_or_dir} not cached under {cache}; download it first"
        )
    return snaps[-1]


def save_safetensors(sd: dict[str, torch.Tensor], path: str) -> None:
    """Write a {name -> tensor} dict to a single .safetensors file."""
    inv = {v: k for k, v in _ST_DTYPE.items()}
    header: dict[str, dict] = {}
    blob = bytearray()
    for name, t in sd.items():
        t = t.contiguous()
        raw = t.view(torch.uint8).reshape(-1).numpy().tobytes() if t.numel() else b""
        s = len(blob)
        blob += raw
        header[name] = {
            "dtype": inv[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [s, len(blob)],
        }
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hdr_bytes)) % 8
    hdr_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(blob)


# --------------------------------------------------------------------------- #
# Transfer
# --------------------------------------------------------------------------- #
@dataclass
class TransferRow:
    aster_name: str
    gemma_name: str
    semantic: str  # meaningful | partial | meaningless
    kind: str  # Copy | CropPad | SvdProject | Skip | MissingTeacher
    teacher_shape: tuple[int, ...] | None
    aster_shape: tuple[int, ...]
    energy: float  # fraction of teacher spectral energy kept (1.0 if exact/N-A)
    embed_whole: int = 0  # VocabEmbed only: whole-token matched rows
    embed_byte: int = 0  # VocabEmbed only: byte-composition seeded rows


def _svd_energy_kept(src: torch.Tensor, dst_shape: tuple[int, ...]) -> float:
    """Fraction of the teacher matrix's squared Frobenius energy that survives the
    two-sided orthogonal SVD projection ``B = U_m^T A V_n`` onto ``dst_shape``.

    ``_svd_project`` now restricts ``A:(M,N)`` to its top-``m`` output and top-``n``
    input singular subspaces (``m=min(dst_rows,M)``, ``n=min(dst_cols,N)``). Because
    ``U_m``/``V_n`` are orthonormal, the kept energy is exactly the projected
    matrix's energy::

        energy_kept = ||U_m^T A V_n||_F^2 / ||A||_F^2

    measured directly (not the loose top-k singular-value bound, which overstates
    retention when *both* axes shrink at once). 1.0 means the projection loses
    nothing (student >= teacher rank on both sides); lower means the squeeze threw
    spectral mass away. This is the honest 'how much of the teacher weight matrix
    actually fits' number -- unlike weight drift vs a zero-init student (which is
    trivially 1.0 and uninformative).
    """
    if src.ndim != 2:
        return 1.0
    a = src.float()
    total = float((a * a).sum())
    if total == 0.0:
        return 1.0
    projected = _svd_project(a, dst_shape).float()
    # Energy of the projection itself (orthonormal bases => this IS the retained
    # Frobenius mass). Crop the zero-pad region back out so growth adds no energy.
    m = min(dst_shape[0], a.shape[0])
    n = min(dst_shape[1], a.shape[1])
    core = projected[:m, :n]
    kept = float((core * core).sum())
    return kept / total


def run_transfer(
    teacher_sd: dict[str, torch.Tensor],
    target_shapes: dict[str, tuple[int, ...]],
    name_map: list[tuple[str, str, str, bool]],
    vocab_map: list[int] | None = None,
    byte_comp: dict[int, list[int]] | None = None,
) -> tuple[dict[str, torch.Tensor], list[TransferRow]]:
    """Produce a new aster-1b state dict from teacher weights + a name map.

    Student tensors are initialized to zeros (we have no student checkpoint to
    warm from -- this is an *initial* transfer). Each mapped tensor is reshaped
    via ferry.transform_tensor, UNLESS the rule forces a Skip (e.g. embed, whose
    vocab axes are not comparable). Unmapped / skipped student tensors stay zero.

    When ``vocab_map`` (a ``t_for_s`` list from :func:`build_vocab_map`) is given,
    the otherwise-force-Skipped ``v2.embed.weight`` is instead seeded from the
    teacher embedding for the subset of tokens whose normalized strings coincide
    (Stage 0 vocab reconciliation). Unmatched rows stay zero -- unless ``byte_comp``
    (from :func:`build_byte_composition`) is also given, in which case unmatched
    decodable rows are seeded by the byte-fallback mean (extends coverage to most
    Korean tokens). See :func:`transfer_embed`.
    """
    new_sd: dict[str, torch.Tensor] = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in target_shapes.items()
    }
    rows: list[TransferRow] = []

    for aster_name, gemma_name, semantic, force_skip in name_map:
        dst_shape = target_shapes[aster_name]
        src = teacher_sd.get(gemma_name)
        if src is None:
            rows.append(
                TransferRow(aster_name, gemma_name, semantic, "MissingTeacher",
                            None, dst_shape, 0.0)
            )
            continue
        src = src.float()
        # Stage 0: vocab-matched embed transfer (only when a map is supplied and
        # this is the embed tensor). Overrides the default force-Skip.
        if aster_name == "v2.embed.weight" and vocab_map is not None:
            emb, whole_w, byte_w = transfer_embed(src, vocab_map, dst_shape, byte_comp)
            new_sd[aster_name] = emb
            written = whole_w + byte_w
            frac = written / dst_shape[0] if dst_shape[0] else 0.0
            rows.append(
                TransferRow(aster_name, gemma_name, "partial", "VocabEmbed",
                            tuple(src.shape), dst_shape, frac,
                            embed_whole=whole_w, embed_byte=byte_w)
            )
            continue
        if force_skip:
            # Intentionally leave zero-init: axes not comparable, SVD would lie.
            rows.append(
                TransferRow(aster_name, gemma_name, semantic, "Skip",
                            tuple(src.shape), dst_shape, 0.0)
            )
            continue
        mapped, kind = transform_tensor(src, dst_shape)
        if kind == "Skip":
            rows.append(
                TransferRow(aster_name, gemma_name, semantic, "Skip",
                            tuple(src.shape), dst_shape, 0.0)
            )
            continue
        new_sd[aster_name] = mapped.float()
        energy = _svd_energy_kept(src, dst_shape) if kind == "SvdProject" else 1.0
        rows.append(
            TransferRow(aster_name, gemma_name, semantic, kind,
                        tuple(src.shape), dst_shape, energy)
        )
    return new_sd, rows


# --------------------------------------------------------------------------- #
# Honest reporting
# --------------------------------------------------------------------------- #
def build_report(rows: list[TransferRow], target_shapes: dict) -> dict:
    """Coverage / by_kind / by_semantic summary. Honest about meaninglessness."""
    total = len(target_shapes)
    mapped = [r for r in rows if r.kind not in ("Skip", "MissingTeacher")]

    by_kind: dict[str, int] = {}
    by_semantic: dict[str, int] = {}
    sem_energy: dict[str, list[float]] = {}
    for r in rows:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
        by_semantic[r.semantic] = by_semantic.get(r.semantic, 0) + 1
        if r.kind == "SvdProject":
            sem_energy.setdefault(r.semantic, []).append(r.energy)

    meaningful_written = sum(
        1 for r in rows
        if r.semantic == "meaningful" and r.kind not in ("Skip", "MissingTeacher")
    )
    # The headline honesty number: spectral energy kept across the FFN stack
    # (the only meaningful block). Low here means even the FFN transfer is lossy.
    ffn_energy = [r.energy for r in rows
                  if r.semantic == "meaningful" and r.kind == "SvdProject"]
    ffn_energy_mean = round(sum(ffn_energy) / len(ffn_energy), 4) if ffn_energy else 0.0

    return {
        "target_tensors": total,
        "name_mapped": len(rows),
        "written": len(mapped),
        "skipped_or_missing": len(rows) - len(mapped),
        "coverage": round(len(mapped) / total, 4) if total else 0.0,
        "by_kind": by_kind,
        "by_semantic": by_semantic,
        "svd_energy_kept_by_semantic": {
            k: round(sum(v) / len(v), 4) for k, v in sem_energy.items()
        },
        "meaningful_tensors_written": meaningful_written,
        "ffn_svd_energy_kept": ffn_energy_mean,
    }


def print_report(rep: dict, rows: list[TransferRow], teacher: str, out_dir: str) -> None:
    print("=" * 72)
    print(f"Gemma-2 -> Aster aster-1b  pure weight transfer  (teacher={teacher})")
    print("=" * 72)
    print(f"  target tensors      : {rep['target_tensors']}")
    print(f"  name-mapped         : {rep['name_mapped']}")
    print(f"  written (non-skip)  : {rep['written']}")
    print(f"  skipped / missing   : {rep['skipped_or_missing']}")
    print(f"  coverage            : {rep['coverage']}")
    print(f"  by_kind             : {rep['by_kind']}")
    print(f"  by_semantic         : {rep['by_semantic']}")
    print(f"  svd_energy_kept     : {rep['svd_energy_kept_by_semantic']}")
    print(f"  ffn_svd_energy_kept : {rep['ffn_svd_energy_kept']}  <- the one honest signal")
    print("-" * 72)
    print("  HONEST READING:")
    # Embed reading depends on the actual mode used (force-Skip vs vocab map).
    embed_row = next((r for r in rows if r.aster_name == "v2.embed.weight"), None)
    if embed_row is not None and embed_row.kind == "VocabEmbed":
        v_s = embed_row.aster_shape[0]
        print(f"    * embed: Stage-0 vocab map seeded {embed_row.energy:.1%} of rows"
              f" ({embed_row.embed_whole + embed_row.embed_byte} / {v_s}).")
        print(f"      - whole-token matched : {embed_row.embed_whole} rows (normalized"
              " string == a teacher token; the higher-quality seed).")
        if embed_row.embed_byte:
            print(f"      - byte-composed       : {embed_row.embed_byte} rows (no whole match;"
                  " seeded by the MEAN of the")
            print("        token's UTF-8 bytes' <0xXX> teacher embeddings, rescaled to the")
            print("        whole-row norm). This is a CRUDE non-zero init: the mean discards")
            print("        byte ORDER, so it carries Gemma's byte-level signal but NOT")
            print("        whole-token semantics -- it makes Korean rows distinct & able to")
            print("        compete in greedy decoding, it is NOT a fluency claim.")
        print("      The hidden axis was right-projected (A V_n) to fit, but it lives in the")
        print("      EMBED's own singular basis -- NOT aligned with the FFN/attn hidden")
        print("      rotation, so with weight tying the logits geometry is still")
        print("      inconsistent. This breaks the embed=0 collapse but is NOT fluent transfer.")
    else:
        print("    * embed (vocab 256000!=48000, different tokenizers) -> forced Skip,")
        print("      stays zero-init. SVD here would only invent noise, so we don't.")
        print("      (Use --embed-vocab-map to seed matched tokens from the teacher.)")
    print("    * attention q/k/v/o: head_dim 256!=96 + different RoPE base -> the")
    print("      projected numbers fill the slots but carry NO usable attention")
    print("      geometry. Counted, never claimed as functional.")
    print("    * ffn gate/up/down: both GeGLU(gelu-tanh) -> SVD keeps the dominant")
    print("      linear directions; the ONLY block with partial semantic meaning.")
    print(f"      It retains {rep['ffn_svd_energy_kept']:.1%} of the teacher's spectral energy.")
    print("    * norms: RMSNorm scale, CropPad of first 1536/2304 channels -> partial.")
    print("    => This is an INITIAL skeleton, not a working model. Without KD or")
    print("       further training it will NOT produce Gemma-quality (or any fluent)")
    print("       output. The number that matters is 'meaningful_tensors_written' =")
    print(f"       {rep['meaningful_tensors_written']} (the FFN stack), retaining")
    print(f"       {rep['ffn_svd_energy_kept']:.1%} spectral energy after the SVD squeeze.")
    print("-" * 72)
    # one example row per (semantic, kind) bucket
    seen: set = set()
    for r in rows:
        key = (r.semantic, r.kind)
        if key in seen:
            continue
        seen.add(key)
        print(f"    [{r.semantic:11s} {r.kind:13s}] {r.aster_name}")
        print(f"        <- {r.gemma_name}  {r.teacher_shape} -> {r.aster_shape}  energy_kept={r.energy:.4f}")
    print("-" * 72)
    print(f"  written to: {out_dir}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="google/gemma-2-2b",
                    help="HF repo id (must be cached) or local dir")
    ap.add_argument("--out", default="./test_output",
                    help="output root (NEW files only; live checkpoint untouched)")
    ap.add_argument("--layer-select", default="uniform", choices=["uniform", "front"],
                    help="when teacher is deeper than student (e.g. 9b=42 vs 26), "
                         "pick teacher layers by even stride (uniform) or first-N (front)")
    ap.add_argument("--out-name", default=None,
                    help="override output subdir name (default derived from teacher+select)")
    ap.add_argument("--embed-vocab-map", action="store_true",
                    help="Stage 0: seed v2.embed from the teacher for tokens whose "
                         "normalized strings match (instead of force-Skip zero-init). "
                         "Requires --student-tokenizer and a cached teacher tokenizer.json")
    ap.add_argument("--student-tokenizer",
                    default="/data/0A_DATASET/L0_LLM/V3/TOKENIZER/tokenizer.json",
                    help="Aster tokenizer.json (vocab 48000); used only with --embed-vocab-map")
    ap.add_argument("--embed-byte-compose", action="store_true",
                    help="extend --embed-vocab-map: seed UNMATCHED decodable tokens "
                         "(esp. Korean) from the MEAN of their UTF-8 bytes' teacher "
                         "<0xXX> byte-fallback embeddings (data-free; crude non-zero "
                         "init, not whole-token semantics). Requires --embed-vocab-map")
    args = ap.parse_args(argv)
    if args.embed_byte_compose and not args.embed_vocab_map:
        ap.error("--embed-byte-compose requires --embed-vocab-map")

    target_shapes = aster_target_shapes(ASTER_1B)

    n_teacher = teacher_num_layers(args.teacher)
    n_student = ASTER_1B["n_layers"]
    teacher_of = select_teacher_layers(n_student, n_teacher, args.layer_select)
    name_map = build_name_map(n_student, n_teacher, args.layer_select)

    vocab_map = None
    vocab_stats = None
    byte_comp = None
    byte_stats = None
    if args.embed_vocab_map:
        teacher_tok = os.path.join(_resolve_snapshot(args.teacher), "tokenizer.json")
        vocab_map, vocab_stats = build_vocab_map(args.student_tokenizer, teacher_tok)
        print(
            f"[0b/4] vocab map: {vocab_stats['matched']}/{vocab_stats['student_vocab']} "
            f"student tokens matched ({vocab_stats['match_frac']:.1%}), "
            f"{vocab_stats['korean_matched']} Korean -> embed seeded for those rows",
            flush=True,
        )
        if args.embed_byte_compose:
            byte_comp, byte_stats = build_byte_composition(
                args.student_tokenizer, teacher_tok, vocab_map)
            print(
                f"[0c/4] byte-compose: +{byte_stats['byte_composed']} unmatched rows "
                f"seeded from {byte_stats['byte_fallback_tokens']} byte-fallback tokens "
                f"({byte_stats['byte_composed_korean']} Korean) -> total embed coverage "
                f"{vocab_stats['matched'] + byte_stats['byte_composed']}/"
                f"{vocab_stats['student_vocab']}",
                flush=True,
            )
    depth_note = (
        f"teacher depth {n_teacher} == student {n_student} (1:1, layer-select N/A)"
        if n_teacher == n_student
        else f"teacher depth {n_teacher} -> student {n_student} via "
             f"'{args.layer_select}': student i<-teacher {teacher_of}"
    )
    print(f"[0/4] {depth_note}", flush=True)

    print(f"[1/4] loading teacher {args.teacher} ...", flush=True)
    teacher_sd = load_teacher(args.teacher)
    print(f"      {len(teacher_sd)} teacher tensors loaded.", flush=True)

    print("[2/4] transferring (SVD / crop-pad, data-free) ...", flush=True)
    new_sd, rows = run_transfer(teacher_sd, target_shapes, name_map, vocab_map, byte_comp)

    print("[3/4] writing student checkpoint ...", flush=True)
    if args.out_name:
        sub = args.out_name
    else:
        # e.g. aster-1b-from-gemma-2-9b-uniform ; keep the 2b default name stable
        # (1:1 depth) so prior runs/paths are unaffected.
        tag = args.teacher.split("/")[-1]
        if n_teacher == n_student:
            sub = f"aster-1b-from-{tag}"
        else:
            sub = f"aster-1b-from-{tag}-{args.layer_select}"
        if args.embed_vocab_map:
            sub += "-embedmap"
        if args.embed_byte_compose:
            sub += "-bc"
    out_dir = os.path.join(args.out, sub)
    os.makedirs(out_dir, exist_ok=True)
    params_path = os.path.join(out_dir, "params.safetensors")
    save_safetensors(new_sd, params_path)

    rep = build_report(rows, target_shapes)
    embed_map_stats = dict(vocab_stats) if vocab_stats else None
    if embed_map_stats is not None and byte_stats is not None:
        embed_map_stats["byte_composition"] = byte_stats
        embed_map_stats["total_rows_seeded"] = (
            vocab_stats["matched"] + byte_stats["byte_composed"])
        embed_map_stats["total_korean_seeded"] = (
            vocab_stats["korean_matched"] + byte_stats["byte_composed_korean"])
    report_path = os.path.join(out_dir, "transfer_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "teacher": args.teacher,
                "target": "aster-1b",
                "teacher_layers": n_teacher,
                "student_layers": n_student,
                "layer_select": args.layer_select if n_teacher != n_student else "identity",
                "teacher_layer_for_student": teacher_of,
                "embed_vocab_map": embed_map_stats,
                "summary": rep,
                "rows": [
                    {
                        "aster": r.aster_name, "gemma": r.gemma_name,
                        "semantic": r.semantic, "kind": r.kind,
                        "teacher_shape": r.teacher_shape, "aster_shape": list(r.aster_shape),
                        "svd_energy_kept": round(r.energy, 6),
                    }
                    for r in rows
                ],
            },
            f, indent=2,
        )

    print("[4/4] done.\n", flush=True)
    print_report(rep, rows, args.teacher, out_dir)
    print(f"  report json: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
