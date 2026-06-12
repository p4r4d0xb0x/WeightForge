"""Ferry on a *real* model -- distill Qwen3-0.6B into a smaller ferry-?B.

What this is
------------
``ferry.py`` proves the same-answer idea on toy models (MLP / ActMLP / TinyLM).
This file applies the **same Ferry pipeline to a real pretrained LLM**: the
teacher is the actual ``Qwen/Qwen3-0.6B`` (596M params) and the student is a
*smaller, architecture-changed* ``ferry-?B`` -- same ``Qwen3ForCausalLM`` family
but with fewer layers and a narrower hidden size. The point the user asked to
prove is precisely that **the architecture can differ and Ferry still drives the
student toward the teacher's answers**.

Hard constraints honored (see AGENTS.md / DEC-007)
--------------------------------------------------
* **CPU only.** Everything forces CPU (no ``.cuda()``, no ``device_map``). The
  module sets ``CUDA_VISIBLE_DEVICES=""`` on import so a stray GPU is never used.
* **Data-free.** Distillation never touches a corpus or disk dataset. Every step
  draws a *fresh synthetic token probe* (random token ids over Qwen's vocabulary)
  and uses the teacher's own logits as the target -- the same data-free recipe as
  ``ferry.distill``. Random tokens are off-distribution, so this aligns *logits*,
  not natural-language quality: an honest PoC limit, not a fluency claim.
* Same tokenizer/vocab on both sides (the student reuses Qwen's vocab), so no
  ``VocabMap`` (stage 0) is needed here.

Pipeline (reuses ferry.py)
--------------------------
1. ``ferry.transfer`` -- name-matched weight transfer Qwen3-0.6B -> ferry-?B
   (Copy where shapes match, CropPad / SvdProject where they differ, Skip on rank
   mismatch). A warm start, not the final answer.
2. ``ferry.distill`` (via a tensor-output adapter) -- gradient fine-tune on fresh
   synthetic token probes so the student's logits track the teacher's.

Run
---
    python ferry_qwen.py                 # build ferry-0.1B, transfer, distill, report
    python ferry_qwen.py --steps 50      # fewer distill steps (CPU is slow)

The first run downloads Qwen3-0.6B (~1.2GB) into the HF cache.
"""

from __future__ import annotations

import argparse
import os

# Enforce CPU-only BEFORE torch picks a device (DEC-007: GPU is forbidden).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from dataclasses import dataclass

import torch

import ferry  # reuse transfer / agreement / distill / token_probe


TEACHER_ID = "Qwen/Qwen3-0.6B"


# --------------------------------------------------------------------------- #
# student architecture presets (smaller + architecture-changed)
# --------------------------------------------------------------------------- #
@dataclass
class StudentPreset:
    """A smaller Qwen3 config. ``head_dim`` is pinned to the teacher's 128 so the
    attention math stays valid regardless of the (smaller) hidden size."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


# Defaults chosen so a CPU Adam loop is tractable. ferry-0.1B is ~103M params.
PRESETS: dict[str, StudentPreset] = {
    "ferry-0.1B": StudentPreset("ferry-0.1B", 512, 1536, 8, 4, 2),
    "ferry-0.12B": StudentPreset("ferry-0.12B", 512, 1536, 12, 4, 2),
    "ferry-0.2B": StudentPreset("ferry-0.2B", 768, 2048, 12, 6, 2),
}


# --------------------------------------------------------------------------- #
# tensor-output adapter
# --------------------------------------------------------------------------- #
class LogitsModel(torch.nn.Module):
    """Wrap a HF causal LM so ``forward(ids) -> logits tensor``.

    ``ferry.agreement`` / ``ferry.distill`` expect a model whose ``forward``
    returns a raw logits tensor (like ``TinyLM``), but a HF model returns a
    ``CausalLMOutput`` object. This adapter exposes the inner model's parameters
    (so ``distill``'s optimizer and ``state_dict`` transfer still work) while
    returning ``.logits`` directly.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.inner(idx).logits

    @torch.no_grad()
    def generate_greedy(self, ctx: torch.Tensor, steps: int) -> torch.Tensor:
        """Minimal greedy decode (argmax), mirroring ferry.TinyLM.generate."""
        ids = ctx
        for _ in range(steps):
            logits = self.forward(ids)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


# --------------------------------------------------------------------------- #
# loaders (CPU only, data-free)
# --------------------------------------------------------------------------- #
def load_teacher() -> LogitsModel:
    """Load the real Qwen3-0.6B on CPU in float32 and wrap it for Ferry."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TEACHER_ID, dtype=torch.float32)
    # Qwen3's config defaults to bfloat16; force a uniform float32 model so CPU
    # autograd never hits a mixed-dtype (Float vs BFloat16) gradient.
    model = model.float()
    model.eval()
    model.to("cpu")
    return LogitsModel(model)


def build_student(preset: StudentPreset) -> LogitsModel:
    """Build a smaller, architecture-changed Qwen3 student from a preset."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(TEACHER_ID)
    cfg.hidden_size = preset.hidden_size
    cfg.intermediate_size = preset.intermediate_size
    cfg.num_hidden_layers = preset.num_hidden_layers
    cfg.num_attention_heads = preset.num_attention_heads
    cfg.num_key_value_heads = preset.num_key_value_heads
    cfg.head_dim = 128  # keep teacher head_dim so heads * head_dim need not equal hidden
    cfg.dtype = torch.float32  # avoid Qwen3's bfloat16 default (CPU autograd needs uniform f32)
    model = AutoModelForCausalLM.from_config(cfg)
    model = model.float()  # belt-and-suspenders: uniform float32 across all params
    model.eval()
    model.to("cpu")
    return LogitsModel(model)


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Ferry pipeline on the real model
# --------------------------------------------------------------------------- #
def transfer_into_student(teacher: LogitsModel, student: LogitsModel) -> dict[str, object]:
    """Stage 1: name-matched weight transfer Qwen3-0.6B -> ferry-?B (warm start).

    Returns the printable ``ferry.report`` summary. The student is updated
    in place via ``load_state_dict`` (strict=False is unnecessary: ``transfer``
    returns a full student state dict, leaving unmatched/Skip tensors as the
    student's own init).
    """
    t_sd = teacher.inner.state_dict()
    s_sd = student.inner.state_dict()
    new_sd, results = ferry.transfer(t_sd, s_sd)
    student.inner.load_state_dict(new_sd)
    return ferry.report(results, s_sd)


def evaluate(
    teacher: LogitsModel,
    student: LogitsModel,
    *,
    seq: int = 16,
    n: int = 8,
    seed: int = 2,
) -> dict[str, float]:
    """Per-token agreement on a held-out synthetic token probe (data-free)."""
    vocab = teacher.inner.config.vocab_size
    probe = ferry.token_probe(n, seq, vocab, seed=seed)
    return ferry.agreement(teacher, student, probe)


def distill_student(
    teacher: LogitsModel,
    student: LogitsModel,
    *,
    steps: int,
    seq: int,
    batch: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    """Stage 3: data-free gradient distillation on fresh synthetic token probes."""
    vocab = teacher.inner.config.vocab_size
    return ferry.distill(
        student,
        teacher,
        vocab=vocab,
        seq=seq,
        steps=steps,
        batch=batch,
        lr=lr,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# demo / CLI
# --------------------------------------------------------------------------- #
def run(
    preset_name: str = "ferry-0.1B",
    *,
    steps: int = 120,
    seq: int = 16,
    batch: int = 8,
    lr: float = 2e-3,
    seed: int = 0,
) -> None:
    """End-to-end: load Qwen3-0.6B, build ferry-?B, transfer, distill, report.

    Kept deliberately small (seq/batch/steps) because everything runs on CPU
    (~0.3s/step, so 120 steps is ~1 min after the one-time teacher load).

    Honest result (measured, ferry-0.1B): per-token top1 agreement moves
    ~0.02 -> ~0.21 and cosine ~0.07 -> ~0.48 -- a large relative gain that proves
    the architecture-changed transfer *works directionally*, but it does NOT close
    to ~1.0 the way the toy TinyLM demo does. Three real limits stack here:
    (1) the student is ~17% of the teacher's params with half the depth (capacity);
    (2) the vocabulary is 151936-wide, so full-logit MSE gives a diffuse signal;
    (3) random token probes are off-distribution for a real LM. This is the
    honest PoC ceiling, not a bug -- it mirrors Ferry's documented capacity limit.
    """
    preset = PRESETS[preset_name]
    torch.manual_seed(seed)

    print(f"[ferry_qwen] loading teacher {TEACHER_ID} (CPU, float32) ...")
    teacher = load_teacher()
    print(f"[ferry_qwen] teacher params: {param_count(teacher):,}")

    print(f"[ferry_qwen] building student {preset.name} "
          f"(hidden={preset.hidden_size}, layers={preset.num_hidden_layers}, "
          f"heads={preset.num_attention_heads}/kv{preset.num_key_value_heads}) ...")
    student = build_student(preset)
    print(f"[ferry_qwen] student params: {param_count(student):,} "
          f"({param_count(student) / param_count(teacher):.1%} of teacher)")

    print("\n[stage 1] weight transfer (name-matched, deterministic) ...")
    rep = transfer_into_student(teacher, student)
    print("  report:", rep)

    before = evaluate(teacher, student, seq=seq)
    print(f"\n[eval] held-out per-token agreement after transfer: "
          f"top1={before['top1_agree']:.3f} mse={before['mse']:.4f}")

    print(f"\n[stage 3] data-free distill ({steps} steps, batch={batch}, "
          f"seq={seq}, lr={lr}) -- fresh synthetic token probes ...")
    after = distill_student(
        teacher, student, steps=steps, seq=seq, batch=batch, lr=lr, seed=seed
    )
    print(f"[eval] held-out per-token agreement after distill: "
          f"top1={after['top1_agree']:.3f} mse={after['mse']:.4f}")

    print(f"\n[summary] {preset.name}: top1 "
          f"{before['top1_agree']:.3f} -> {after['top1_agree']:.3f} "
          f"(data-free, CPU, architecture-changed student)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill Qwen3-0.6B into a smaller ferry-?B (CPU, data-free).")
    p.add_argument("--student", default="ferry-0.1B", choices=list(PRESETS))
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.student,
        steps=args.steps,
        seq=args.seq,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
