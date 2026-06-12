# WeightForge

> Codename **Ferry** — the pipeline name used throughout the source.

Make a smaller **student** model produce the **same answers** as a larger
**teacher** model — even when they differ in layer count, hidden dimension, and
vocabulary — **without any training data**. Targets come only from the teacher,
driven by synthetic probes (random tensors / token ids). No datasets, no loaders,
no disk I/O for data.

This is a proof of concept. The same-answer guarantee is **conditional** (a rank
condition): it holds only when the student is wide enough to linearly reconstruct
the teacher's output map. When it cannot, WeightForge **reports the residual
honestly** instead of faking a match.

## How it works

| Stage | What | Method |
|---|---|---|
| 0 | Vocabulary reconciliation (LLM only) | `reconcile_vocab` → `VocabMap` (teacher↔student token map + projection) |
| 1 | Weight transfer | name-matched `Copy` / `CropPad` / `SvdProject` / `Skip` (deterministic) |
| 2 | Output alignment | closed-form least squares (`torch.linalg.lstsq`) re-fits the last linear layer |
| 2b | Hidden alignment | closed-form forward sweep over hidden linears (nonlinear teachers) |
| 3 | Gradient distillation | Adam loop, fresh synthetic probe each step, teacher output as target (data-free) |

Stages 1–2b are deterministic closed-form algebra. Gradient training is confined
to Stage 3 (see `theory.html` for the full derivation and honest limits).

## Layout

```
ferry.py                    # core pipeline + runnable 6-part toy demo (MLP / ActMLP / TinyLM)
test_ferry.py               # pytest suite (42 cases, torch-only)
ferry_qwen.py               # real-model: Qwen3-0.6B -> architecture-changed ferry-?B (CPU, data-free)
test_ferry_qwen.py          # gated tests (skip if model unavailable)
transfer_gemma_to_aster.py  # real-model: Gemma-2 -> Aster weight transfer + vocab/byte-compose
ferry_aster.py              # Aster PyTorch reproduction + data-free KD
align_aster_embed.py        # closed-form Procrustes embed-basis alignment
theory.html                 # self-contained theory write-up (no dependencies)
AGENTS.md                   # project conventions and hard constraints
.agents/                    # project docs (GOAL/PLAN/TODO/PROGRESS/DECISION/MEMORY)
```

## Run

```bash
python -m pytest test_ferry.py -q   # toy tests (42 cases, fast, no extra deps)
python ferry.py                     # toy demo, prints transfer report

# real-model extension (CPU-only, data-free) — needs transformers + accelerate
python ferry_qwen.py                # distill Qwen3-0.6B -> ferry-0.1B, report
python -m pytest test_ferry_qwen.py -q
```

`torch` is required (toy core has no other deps). `ferry_qwen.py` additionally
needs `transformers` + `accelerate` and downloads `Qwen/Qwen3-0.6B` (~1.2GB) on
first run. **GPU is disabled by design** — everything runs on CPU in float32.

## Honest limits

- Same-answer is guaranteed only under the rank condition; narrow students plateau
  below 100% agreement (capacity sweep in `theory.html` §6).
- Nonlinear teachers need Stage 2b; shallower students cannot fully close the gap.
- Autoregressive residual is flattened by Stage 3, not erased.
- Real-model runs are a directional PoC ceiling, not a fluency claim.

## License

Source-available, **publication-reserved (strict)** — see [`LICENSE`](./LICENSE).
A narrow, revocable privilege for private, non-public, non-commercial evaluation
and research only. Publication, research-credit, redistribution, commercial use,
patenting, and using the Work or its outputs to train/distill or build other
models are reserved to the Author and require prior written consent.
