"""Grow SLM_FROM_BEGIN's trained aster-1b into an aster-3b *initial* checkpoint.

What this is
------------
``ferry.py`` proves the same-answer / weight-transfer idea on toy models;
``ferry_qwen.py`` distills a downloaded HF model; ``ferry_aster.py`` reproduces
aster-1b in PyTorch for Gemma->Aster KD. **This file does something different and
narrower**: it applies Ferry's *Stage-1 weight transfer* to a real local
checkpoint to **grow** the *trained* ``aster-1b`` safetensors of the sibling
project ``../SLM_FROM_BEGIN`` into an ``aster-3b`` initial checkpoint -- the
"초기 세팅값" (initial seed weights) a subsequent 3B pretraining run can resume
from instead of a cold random start.

This is a **growth** (small -> large) direction: every dimension increases
(``d_model`` 1536 -> 3072, ``n_layers`` 26 -> 28, kv_dim 768 -> 1024,
``ffn_inner`` 6144 -> 8448). Ferry's ``transform_tensor`` already handles growth:
2D weights are SVD-projected onto the (larger) target subspaces and zero-padded
where there is no teacher signal; 1D norms are crop/zero-padded. New transformer
blocks (26, 27) that have no aster-1b counterpart are *freshly initialized* with
the exact scheme the SLM framework uses (truncated-normal std=0.02, residual
projections std=0.02/sqrt(2*n_layers), RMSNorm gamma=1.0), so they are healthy
init rather than zeros.

Scope (locked with the user)
----------------------------
* **Stage 1 transfer only.** Produce the 3B *initial* checkpoint. No distillation,
  no gradient loop, no data -- purely the deterministic weight-space growth.
* **Growth fill = Ferry default** (SVD-project + zero-pad / crop-pad). Unchanged
  ``ferry.transform_tensor`` behaviour.

Hard constraints honored (Ferry AGENTS.md)
------------------------------------------
* **Data-free.** No dataset, no corpus, no probes even -- stage 1 is pure algebra
  on the existing weights.
* **CPU only.** Everything is plain ``torch`` tensor math on CPU.
* The live ``aster-1b`` checkpoint is **read-only**; output is a NEW directory.

Checkpoint contract (matched to SLM_FROM_BEGIN slm-model)
---------------------------------------------------------
The SLM framework's ``TinyCausalLm`` loader is *strict* (see
``crates/slm-model/src/tiny_causal_lm/params.rs``):

* every saved key carries a ``v2.`` prefix (``v2.embed.weight``,
  ``v2.blocks.{i}.q.weight`` ...); the loader rejects any other prefix with a
  ``CheckpointVersionMismatch``;
* tensors are F32; a dtype or shape mismatch is a hard load error;
* weight tying is on, so ``head.weight`` is **not** stored (embed doubles as the
  LM head);
* the full key set must be present: ``embed.weight`` + per-block 9 tensors
  (``attn_norm.gamma``, ``q/k/v/o.weight``, ``ffn_norm.gamma``,
  ``ffn_gate/up/down.weight``) + ``final_norm.gamma``. For 28 blocks that is
  ``1 + 28*9 + 1 = 254`` tensors.

We therefore synthesize the target 3B tensor *spec* directly from the 3B model
TOML (no need to instantiate the Rust model) and write a checkpoint that the
loader accepts as-is.

Run
---
    # dry run: print the 1b->3b transfer plan + report, write nothing
    python grow_aster_1b_to_3b.py --dry-run

    # build the 3b initial checkpoint (default paths from ../SLM_FROM_BEGIN)
    python grow_aster_1b_to_3b.py

    # explicit paths
    python grow_aster_1b_to_3b.py \
        --src   ../SLM_FROM_BEGIN/artifacts/checkpoints/aster-1b/params.safetensors \
        --config ../SLM_FROM_BEGIN/configs/model/pretrain-3b.toml \
        --out   ../SLM_FROM_BEGIN/artifacts/checkpoints/aster-3b-init
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import ferry  # reuse transform_tensor (Stage-1 growth transforms)


# --------------------------------------------------------------------------- #
# defaults (relative to this file, pointing at the sibling SLM_FROM_BEGIN repo)
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_SLM = _HERE.parent / "SLM_FROM_BEGIN"
DEFAULT_SRC = _SLM / "artifacts" / "checkpoints" / "aster-1b" / "params.safetensors"
DEFAULT_CONFIG = _SLM / "configs" / "model" / "pretrain-3b.toml"
DEFAULT_OUT = _SLM / "artifacts" / "checkpoints" / "aster-3b-init"

CKPT_PREFIX = "v2."  # slm-model frozen namespace (DEC-027/DEC-034)
INIT_SEED = 3_000_000_003  # deterministic seed for fresh (new-block) tensors

# --------------------------------------------------------------------------- #
# optimizer-partition contract (ported verbatim from SLM_FROM_BEGIN trainer)
# --------------------------------------------------------------------------- #
# crates/slm-train/src/optim/muonclip.rs route()/MUON_SUFFIXES: a param is routed
# to the **Muon** optimizer iff it is a rank-2 tensor whose name starts with
# "blocks." AND ends with one of these matrix suffixes. EVERYTHING ELSE (all
# *.gamma RMSNorm scales, embed.weight, final_norm.gamma) is routed to **AdamW**.
# The training-resume loader (ckpt.rs:254-347) HARD-REQUIRES the AdamW sidecars
# (optimizer.safetensors + optimizer_state.json); the Muon momentum sidecar is
# OPTIONAL (absent -> "starting Muon momentum from zero"). So a resumable initial
# checkpoint needs the two AdamW sidecars, zero-valued for a from-step-1 start.
MUON_SUFFIXES: tuple[str, ...] = (
    "q.weight",
    "k.weight",
    "v.weight",
    "o.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)

# AdamW hyperparameters serialized into optimizer_state.json. These mirror the
# real aster-1b sidecar (beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1); the
# trainer overrides them from its live config on resume, so they are recorded for
# format completeness rather than as the authoritative schedule.
ADAMW_DEFAULTS: dict[str, float] = {
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.1,
}
OPTIMIZER_STATE_SCHEMA_VERSION = 1


def routes_to_muon(name: str, shape: tuple[int, ...]) -> bool:
    """Port of SLM `route()`: True iff this param is optimized by Muon (not AdamW).

    ``name`` is the *logical* param name (no ``v2.`` prefix).
    """
    return (
        len(shape) == 2
        and name.startswith("blocks.")
        and any(name.endswith(suffix) for suffix in MUON_SUFFIXES)
    )


def adamw_param_names(shapes: dict[str, tuple[int, ...]]) -> list[str]:
    """The subset of params routed to AdamW (i.e. NOT to Muon), in spec order."""
    return [name for name, shape in shapes.items() if not routes_to_muon(name, shape)]


# --------------------------------------------------------------------------- #
# types: a derived model spec (shapes only) -- no Rust model instantiation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """The dimensions needed to synthesize a 3B tensor layout.

    Derived from the SLM framework rules (``crates/slm-types/src/model.rs``):
      * ``head_dim   = d_model / n_heads``
      * ``kv_dim     = n_kv_heads * head_dim``
      * ``ffn_inner  = round(d_model * ffn_mult)``
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_mult: float
    weight_tying: bool

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"
            )
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def ffn_inner(self) -> int:
        return round(self.d_model * self.ffn_mult)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Full ``{logical_name: shape}`` map (no ``v2.`` prefix, no head when tied).

        Block tensor order mirrors ``init.rs`` / ``params.rs`` (attn_norm, q, k,
        v, o, ffn_norm, ffn_gate, ffn_up, ffn_down). PyTorch ``nn.Linear`` stores
        weights as ``[out, in]``, matching the framework's row-major
        ``[out, in]`` shapes seen in the live checkpoint header.
        """
        d, ff, kv = self.d_model, self.ffn_inner, self.kv_dim
        shapes: dict[str, tuple[int, ...]] = {"embed.weight": (self.vocab_size, d)}
        for i in range(self.n_layers):
            p = f"blocks.{i}."
            shapes[p + "attn_norm.gamma"] = (d,)
            shapes[p + "q.weight"] = (d, d)
            shapes[p + "k.weight"] = (kv, d)
            shapes[p + "v.weight"] = (kv, d)
            shapes[p + "o.weight"] = (d, d)
            shapes[p + "ffn_norm.gamma"] = (d,)
            shapes[p + "ffn_gate.weight"] = (ff, d)
            shapes[p + "ffn_up.weight"] = (ff, d)
            shapes[p + "ffn_down.weight"] = (d, ff)
        shapes["final_norm.gamma"] = (d,)
        if not self.weight_tying:
            shapes["head.weight"] = (self.vocab_size, d)
        return shapes


def load_model_spec(config_path: Path) -> ModelSpec:
    """Parse a ``pretrain-*.toml`` into a :class:`ModelSpec`."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    embedding = cfg.get("embedding", {})
    return ModelSpec(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        n_kv_heads=int(cfg.get("n_kv_heads", cfg["n_heads"])),
        ffn_mult=float(cfg["ffn_mult"]),
        weight_tying=bool(embedding.get("weight_tying", True)),
    )


# --------------------------------------------------------------------------- #
# fresh init for tensors with no teacher counterpart (new blocks 26, 27 ...)
# --------------------------------------------------------------------------- #
def _init_tensor(
    name: str, shape: tuple[int, ...], n_layers: int, gen: torch.Generator
) -> torch.Tensor:
    """Fresh parameter init matching SLM ``init.rs`` (truncated normal / gamma=1).

    * RMSNorm ``*.gamma`` (1D): all ones (identity transform at init).
    * residual projections ``o.weight`` / ``ffn_down.weight``: std = 0.02 /
      sqrt(2 * n_layers).
    * everything else (embed, q, k, v, ffn_gate, ffn_up, head): std = 0.02.

    The Gaussian is truncated to |z| <= 2 like the framework's ``SmokeRng``. We
    do NOT reproduce the framework's exact RNG stream (a different PRNG), only its
    *distribution* -- these tensors are a fresh-init starting point either way, so
    distributional fidelity is what matters, not bit-identical samples.
    """
    if name.endswith(".gamma"):
        return torch.ones(shape, dtype=torch.float32)

    is_residual = name.endswith("o.weight") or name.endswith("ffn_down.weight")
    std = 0.02 / ((2.0 * n_layers) ** 0.5) if is_residual else 0.02

    out = torch.empty(shape, dtype=torch.float32)
    # truncated normal in [-2, 2] std units, then scale.
    torch.nn.init.trunc_normal_(out, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=gen)
    out.mul_(std)
    return out


# --------------------------------------------------------------------------- #
# core: grow 1b weights into the 3b spec (Ferry stage-1 transfer, data-free)
# --------------------------------------------------------------------------- #
@dataclass
class GrowResult:
    """Per-tensor outcome of the growth transfer."""

    name: str
    kind: str  # Copy | CropPad | SvdProject | FreshInit
    src_shape: tuple[int, ...] | None  # None when no teacher tensor existed
    dst_shape: tuple[int, ...]


def strip_prefix(name: str) -> str:
    """``v2.blocks.0.q.weight`` -> ``blocks.0.q.weight`` (logical name)."""
    return name[len(CKPT_PREFIX):] if name.startswith(CKPT_PREFIX) else name


def grow_checkpoint(
    src_sd: dict[str, torch.Tensor], spec: ModelSpec
) -> tuple[dict[str, torch.Tensor], list[GrowResult]]:
    """Build a full 3B (logical-name) state dict from a 1B source state dict.

    For each target tensor:
      * if a same-named source tensor exists, run ``ferry.transform_tensor`` to
        map it onto the target shape (Copy / CropPad / SvdProject -- growth means
        SVD-project + zero-pad for 2D, crop/zero-pad for 1D);
      * otherwise (e.g. new blocks beyond the source's depth) fresh-init it.

    Returns ``(logical_state_dict, per_tensor_results)``. Keys are *logical*
    (no ``v2.`` prefix); :func:`write_checkpoint` adds the prefix on save.
    """
    src_logical = {strip_prefix(k): v.float() for k, v in src_sd.items()}
    target = spec.tensor_shapes()
    gen = torch.Generator().manual_seed(INIT_SEED)

    out: dict[str, torch.Tensor] = {}
    results: list[GrowResult] = []

    for name in target:  # deterministic insertion order = spec order
        dst_shape = target[name]
        src = src_logical.get(name)
        if src is None:
            out[name] = _init_tensor(name, dst_shape, spec.n_layers, gen)
            results.append(GrowResult(name, "FreshInit", None, dst_shape))
            continue

        mapped, kind = ferry.transform_tensor(src, dst_shape)
        if kind == "Skip" or tuple(mapped.shape) != dst_shape:
            # Rank/shape could not be reconciled by transform_tensor; fall back to
            # a fresh init so the strict loader still gets a correctly shaped F32
            # tensor. (Not expected for the aster 1b->3b growth, which is all
            # same-rank; guarded for honesty rather than silently shipping a
            # wrong-shaped tensor.)
            out[name] = _init_tensor(name, dst_shape, spec.n_layers, gen)
            results.append(GrowResult(name, "FreshInit", tuple(src.shape), dst_shape))
            continue

        out[name] = mapped.float().contiguous()
        results.append(GrowResult(name, kind, tuple(src.shape), dst_shape))

    return out, results


def grow_report(results: list[GrowResult]) -> dict[str, object]:
    """Small printable summary of a growth transfer."""
    by_kind: dict[str, int] = {}
    for r in results:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
    grown = [r for r in results if r.kind != "FreshInit"]
    return {
        "target_tensors": len(results),
        "grown_from_1b": len(grown),
        "fresh_init": len(results) - len(grown),
        "coverage": round(len(grown) / len(results), 4) if results else 0.0,
        "by_kind": by_kind,
    }


# --------------------------------------------------------------------------- #
# checkpoint writer (v2.-prefixed safetensors + state.json), matched to loader
# --------------------------------------------------------------------------- #
def write_optimizer_sidecars(
    out_dir: Path, logical_sd: dict[str, torch.Tensor]
) -> tuple[Path, Path, int]:
    """Write zero-valued AdamW sidecars so the checkpoint is resume-able.

    The SLM trainer's only weight-loading path is ``--resume-from``, and its
    ``load_training_checkpoint`` HARD-REQUIRES ``optimizer.safetensors`` +
    ``optimizer_state.json`` (a model-only checkpoint errors out). For a *fresh*
    start we emit those two files with **zero** moments and **step 0** for every
    AdamW-routed param, which is exactly the state a cold optimizer would hold
    before its first update -- so resuming from this checkpoint is identical to a
    from-scratch optimizer, but with the grown weights warm-started.

    Layout (frozen contract, ``crates/slm-train/src/ckpt.rs``):
      * ``optimizer.safetensors``: keys ``adamw.<name>.m`` and ``adamw.<name>.v``,
        both F32 zeros, same shape as the param, for each AdamW-routed param.
      * ``optimizer_state.json``: ``OptimizerStateSidecar`` with the AdamW
        hyperparameters and a ``steps`` list of ``[name, 0]`` per AdamW param.

    The Muon momentum sidecar is intentionally NOT written: the loader treats its
    absence as "start Muon momentum from zero", which is the correct fresh state.
    Routing matrix params through a zero Muon sidecar would be redundant.

    Returns ``(safetensors_path, state_json_path, num_adamw_params)``.
    """
    shapes = {name: tuple(t.shape) for name, t in logical_sd.items()}
    adamw_names = adamw_param_names(shapes)

    moments: dict[str, torch.Tensor] = {}
    for name in adamw_names:
        zeros = torch.zeros(shapes[name], dtype=torch.float32)
        moments[f"adamw.{name}.m"] = zeros
        moments[f"adamw.{name}.v"] = zeros.clone()

    opt_path = out_dir / "optimizer.safetensors"
    save_file(moments, str(opt_path))

    sidecar = {
        "schema_version": OPTIMIZER_STATE_SCHEMA_VERSION,
        **ADAMW_DEFAULTS,
        "steps": [[name, 0] for name in adamw_names],
    }
    state_path = out_dir / "optimizer_state.json"
    state_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    return opt_path, state_path, len(adamw_names)


def write_checkpoint(
    out_dir: Path,
    logical_sd: dict[str, torch.Tensor],
    *,
    seed: int,
    optimizer_sidecars: bool = True,
) -> Path:
    """Write a 3B initial checkpoint the SLM loader accepts.

    Saves ``params.safetensors`` with every key ``v2.``-prefixed and F32, plus a
    minimal ``state.json`` (``global_step = 0`` -- this is a fresh seed, not a
    resumed step).

    When ``optimizer_sidecars`` is True (default), also emits zero-valued AdamW
    sidecars (``optimizer.safetensors`` + ``optimizer_state.json``) so the
    checkpoint can be consumed directly by the trainer's ``--resume-from`` path,
    which hard-requires them. The Muon momentum sidecar is never written (its
    absence is a valid "start from zero" for the trainer). 1B optimizer moments
    are deliberately NOT carried over: they index the old shapes and a fresh
    schedule should start their moments at zero anyway.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefixed = {
        CKPT_PREFIX + name: t.to(torch.float32).contiguous()
        for name, t in logical_sd.items()
    }
    params_path = out_dir / "params.safetensors"
    save_file(prefixed, str(params_path))

    state = {"schema_version": 1, "global_step": 0, "lr": 0.0, "seed": int(seed)}
    (out_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")

    if optimizer_sidecars:
        write_optimizer_sidecars(out_dir, logical_sd)
    return params_path


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build(
    src_path: Path,
    config_path: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
    seed: int = INIT_SEED,
    optimizer_sidecars: bool = True,
) -> dict[str, object]:
    """End-to-end: load aster-1b, grow into the 3B spec, write the initial ckpt.

    Returns the growth report (also printed). When ``dry_run`` nothing is written.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {src_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"3b model config not found: {config_path}")

    spec = load_model_spec(config_path)
    src_sd = load_file(str(src_path))

    print(f"[grow_aster] source : {src_path}")
    print(f"[grow_aster]          {len(src_sd)} tensors")
    print(f"[grow_aster] target : {config_path.name} -> "
          f"d_model={spec.d_model}, n_layers={spec.n_layers}, "
          f"n_heads={spec.n_heads}/kv{spec.n_kv_heads}, head_dim={spec.head_dim}, "
          f"ffn_inner={spec.ffn_inner}, vocab={spec.vocab_size}, "
          f"tied={spec.weight_tying}")

    logical_sd, results = grow_checkpoint(src_sd, spec)
    rep = grow_report(results)
    print("[grow_aster] growth report:", rep)

    # show a few representative per-tensor transforms (one per structural kind).
    seen: set[str] = set()
    print("[grow_aster] sample transforms (one per structural tensor):")
    for r in results:
        key = strip_prefix(r.name)
        struct = (
            key
            if "blocks." not in key
            else "blocks.N." + key.split("blocks.", 1)[1].split(".", 1)[1]
        )
        if struct in seen:
            continue
        seen.add(struct)
        print(f"    {r.kind:10s} {str(r.src_shape):>16s} -> "
              f"{str(r.dst_shape):<16s} {struct}")

    if dry_run:
        print("[grow_aster] --dry-run: no checkpoint written.")
        return rep

    params_path = write_checkpoint(
        out_dir, logical_sd, seed=seed, optimizer_sidecars=optimizer_sidecars
    )
    total_bytes = params_path.stat().st_size
    print(f"[grow_aster] wrote {params_path} ({total_bytes / 1e9:.2f} GB)")
    print(f"[grow_aster] wrote {out_dir / 'state.json'} (global_step=0, fresh seed)")
    if optimizer_sidecars:
        n_adamw = len(adamw_param_names({n: tuple(t.shape) for n, t in logical_sd.items()}))
        print(f"[grow_aster] wrote optimizer.safetensors ({n_adamw} AdamW params "
              f"x m,v = {2 * n_adamw} zero tensors) + optimizer_state.json")
        print("[grow_aster] -> checkpoint is resume-able: "
              f"slm pretrain --resume-from {out_dir}")
        print("[grow_aster] NOTE: muon_momentum.safetensors intentionally NOT "
              "written -- the trainer starts Muon momentum from zero when absent.")
    else:
        print("[grow_aster] NOTE: optimizer/muon sidecars NOT written "
              "(--no-optimizer-sidecars); this is a model-only checkpoint and "
              "CANNOT be --resume-from'd as-is.")
    return rep


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grow SLM_FROM_BEGIN aster-1b into an aster-3b initial "
                    "checkpoint (Ferry stage-1 weight transfer; data-free, CPU)."
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="source aster-1b params.safetensors")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="target 3b model TOML (pretrain-3b.toml)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="output checkpoint directory")
    p.add_argument("--dry-run", action="store_true",
                   help="print the transfer plan/report without writing anything")
    p.add_argument("--seed", type=int, default=INIT_SEED,
                   help="deterministic seed for fresh-init tensors + state.json")
    p.add_argument("--no-optimizer-sidecars", dest="optimizer_sidecars",
                   action="store_false",
                   help="skip the zero AdamW sidecars (writes a model-only "
                        "checkpoint that cannot be --resume-from'd as-is)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(
        args.src.resolve(),
        args.config.resolve(),
        args.out.resolve(),
        dry_run=args.dry_run,
        seed=args.seed,
        optimizer_sidecars=args.optimizer_sidecars,
    )
