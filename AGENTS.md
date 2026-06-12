# AGENTS.md

## What this is

**Ferry** — a PoC that makes a *student* model produce the **same answers** as a
*teacher* model, even when they differ in layer count / hidden dimension. Three
stages: (1) weight-space transfer, (2) closed-form alignment (output + hidden),
and (3) gradient distillation that actually *closes* the nonlinear / depth /
autoregressive limits. All driven by synthetic probes — no external dataset is
ever used (the teacher supplies the targets).

## Goal (the thing being proven)

`align_output` re-fits the student's last linear layer (least-squares, closed
form) so `student(x) == teacher(x)` for **any** `x` — guaranteed **iff** the
student's penultimate width can linearly reconstruct the teacher's output map
(rank condition). Too narrow a student = mathematically impossible; Ferry then
reports the residual instead of faking a match. See `theory.html`.

## Hard constraints (do not violate without explicit user approval)

- **No training data / no datasets.** No data loaders, no real samples, no labels,
  no disk I/O for data. A prior adapter+KD plan was rejected for needing data.
  This is still in force — the project stays **data-free**.
- **Synthetic probes ARE allowed** (user-approved, supersedes the earlier
  "no behavioral probing" rule). `synthetic_probe()` / `token_probe()` make random
  tensors / token ids to compare teacher/student outputs and to drive alignment
  and distillation. Real/loaded data stays banned.
- **Gradient training is now ALLOWED** (DEC-005, user explicitly lifted the earlier
  "no gradient loop" constraint). Stage 3 `distill` runs an Adam loop. It stays
  data-free: every step draws a *fresh* synthetic probe and uses the teacher's own
  output as the target (no dataset, no disk). Stages 1–2 remain deterministic
  closed-form (copy / crop-pad / SVD / `lstsq`); the gradient loop is confined to
  Stage 3.

## Layout

Flat by design (user wants to edit it easily). Core is two files, no packaging:
- `ferry.py` — all logic + a runnable toy demo. Models: `MLP` (linear, exact
  guarantee), `ActMLP` (nonlinear, honest-limit demo), and `TinyLM` (tiny
  GPT-style transformer, autoregressive-limit demo). `_demo` has six parts:
  `_demo_linear_transfer` / `_demo_capacity_sweep` / `_demo_nonlinear_limit` /
  `_demo_llm_like` / `_demo_vocab_mismatch` (different teacher/student vocab) /
  `_demo_combined_mismatch` (vocab + depth + width all differ, scrambled/partial
  vocab map — the realistic worst case).
- `test_ferry.py` — pytest suite (42 cases, torch-only, no external deps).

Real-model extension (DEC-007), separate so the toy core stays dep-light:
- `ferry_qwen.py` — applies the SAME Ferry pipeline to a real pretrained LLM:
  teacher = actual `Qwen/Qwen3-0.6B` (596M), student = a smaller,
  **architecture-changed** `ferry-?B` (same `Qwen3ForCausalLM` family, fewer
  layers + narrower hidden; presets `ferry-0.1B`/`-0.12B`/`-0.2B`). Reuses
  `ferry.transfer` + `ferry.distill` via a `LogitsModel` adapter (HF
  `(ids)->CausalLMOutput` → raw logits tensor). **CPU-only, data-free**
  (synthetic token probes). Needs `transformers` + `accelerate` and a ~1.2GB
  model download — the one sanctioned exception to "no external deps".
- `test_ferry_qwen.py` — gated tests (`pytest.importorskip('transformers')`,
  skip if model unavailable); load the real teacher once via a module fixture.

No `pyproject.toml`. The project is published as a public git repo at
`github.com/p4r4d0xb0x/WeightForge` (codename **Ferry** in the source) with a
minimal `README.md` and a source-available, publication-reserved `LICENSE`
(DEC-015). Do not add packaging/CLI/multi-module structure unless asked —
simplicity is a stated requirement. (`ferry_qwen.py` has a small argparse CLI
only because real-model runs need knobs; the toy core stays CLI-free.)

Project docs live in `.agents/` (GOAL/PLAN/TODO/PROGRESS/DECISION/MEMORY).
Session handoff notes live in `handoff.md` at the project root.

## Operating rules

- **TODO↔Issue policy: `off`** (DEC-004). Flat, single-file-style PoC; now
  published as a public git repo (DEC-015) but ticket linkage stays suggested
  only, never enforced.

## Commands

```bash
python -m pytest test_ferry.py -q   # run toy tests (42 cases, fast, no deps)
python ferry.py                     # run toy demo, prints transfer report

# real-model extension (CPU-only, data-free) — needs transformers + accelerate
python ferry_qwen.py                # distill Qwen3-0.6B -> ferry-0.1B, report
python ferry_qwen.py --steps 200 --student ferry-0.2B   # bigger student / longer
python -m pytest test_ferry_qwen.py -q   # gated tests (skip if model unavailable)
```

`torch` is already installed (2.10.0+cu128, Python 3.12). No install step for the
toy core. `ferry_qwen.py` needs `transformers` + `accelerate` (installed) and
downloads `Qwen/Qwen3-0.6B` (~1.2GB) on first run. **GPU is forbidden** (DEC-007):
`ferry_qwen.py` sets `CUDA_VISIBLE_DEVICES=""` on import and runs everything on
CPU in float32.

## Core pipeline (in `ferry.py`)

Stage 0 — vocabulary reconciliation (LLM only):
An LM emits a distribution over *its own* vocabulary, so a teacher/student pair
built with different tokenizers cannot be aligned head-to-head (different LM-head
widths; token id `j` means different tokens). `reconcile_vocab(student, teacher,
t_for_s=None) → VocabMap` builds the correspondence. `VocabMap` carries
`t_for_s` (LongTensor `(V_s,)`: teacher id per student id, `-1` = student-only)
and a `(V_t, V_s)` selection-matrix `projection`. Two ops: `remap_ids(student_ids)`
translates a student-space probe into teacher ids (so the same probe feeds both
models; `-1` clamped to 0); `project(teacher_logits)` maps teacher logits
`(.., V_t) → (.., V_s)` (column `j` = teacher logit of student token `j`'s match;
student-only columns zero = no teacher signal). Default `t_for_s` is the
*shared-prefix* map (`arange(min(V_t,V_s))`, tail `-1`); pass a real tokenizer-
derived map in production. `build_vocab_map(t_for_s, size_t)` constructs it
directly. `shared_token_probe(n, seq, vmap, seed)` samples only mapped tokens.
`agreement` / `align_output` / `distill` all take an optional `vocab_map=None`
arg — when set they feed the teacher remapped ids and project its logits into
student space before comparing/fitting; `None` is the exact prior behaviour
(same-vocab no-op). `distill` rejects a `vocab_map` outside token mode.

Stage 1 — weight transfer:
`extract_spec → match_tensors (by name) → transform_tensor → transfer → report`
Transform kinds: `Copy` (same shape), `CropPad` (same rank, crop/zero-pad),
`SvdProject` (2D dim mismatch, truncated SVD), `Skip` (rank mismatch, student kept).
To add a transform, extend the branch in `transform_tensor` only.

Stage 2 — output alignment:
`synthetic_probe → agreement → align_output`. `align_output` hooks the student's
last `nn.Linear`, captures its input features, solves `[F|1] @ W = teacher(probe)`
via `lstsq`, and writes `W` back. `agreement` returns `{mse, top1_agree, cosine}`.
Both flatten leading dims via `_flatten_logits` (`(...,d) -> (prod,d)`) so they
handle sequence-model logits `(n, seq, vocab)` as well as `(n, out)` — no-op for
2D. `token_probe(n, seq, vocab)` makes random token-id probes for `TinyLM`.

Stage 2b — hidden alignment (nonlinear support):
`align_hidden(student, teacher, probe)` reshapes the student's *internal* basis so
nonlinear teachers can be matched, not just reported. Forward sweep: for each
matched hidden `nn.Linear` (via `_linear_chain` over `model.net`), solve
`lstsq([student_in|1] @ W = teacher_preact)` so the student's pre-activation tracks
the teacher's, then fit the head via `align_output`. Still closed-form (only
`lstsq`), no gradients, synthetic-probe-only. Scope: flat MLP family (`MLP`/
`ActMLP`); `TinyLM` has no flat chain so `_linear_chain` returns `[]` and it falls
back to head-only. Width mismatch is handled by crop/zero-pad of the teacher
pre-activation target to the student layer's width.

Stage 3 — gradient distillation (closes the limits):
`distill(student, teacher, *, in_dim=… | vocab=…, seq=…, steps, batch, lr, seed)`
runs an Adam loop that minimizes `mse_loss(student(probe), teacher(probe))`. Run it
AFTER the closed-form stages (warm start → fast convergence). **Data-free**: each
step draws a *fresh* synthetic probe (`synthetic_probe` for continuous / MLP family,
`token_probe` for `TinyLM`) and uses the teacher's output as the target — resampling
every step is what makes the fit *generalize* to held-out probes (a fixed probe
overfits). Exactly one input mode must be chosen (`in_dim` XOR `vocab`+`seq`), else
`ValueError`. Returns held-out `agreement` (eval probe at `seed-99`, unseen in the
loop). This is the stage that turns the honest *limits* into *near-closures*:
nonlinear depth-matched ActMLP → ~0.99 held-out top-1; `TinyLM` per-token
0.41→0.89 and the autoregressive decay curve flattens (step8 .34→.61).

## Gotchas

- `test_ferry.py` imports `import ferry as clone` (alias kept from the old
  filename so test bodies read `clone.*`). If you rename `ferry.py`, update this
  alias line — the bodies intentionally still say `clone.`.
- `report()`'s `mean_relative_error` (stage 1) measures **weight-space drift vs the
  original random student init**, NOT answer quality. Stage-1 error being large is
  expected and fine — stage 2 (`align_output`) is what makes answers match.
- **Same-answer guarantee is conditional.** It holds only when the student is wide
  enough (rank condition). The demo's capacity sweep shows narrow widths plateau
  below 100% `top1_agree`; that is the honest limit, not a bug to patch.
- **Nonlinear teachers need `align_hidden`, not just `align_output`.** With a
  nonlinear teacher (`ActMLP`), head-only `align_output` reaches only a best linear
  fit of the student's own nonlinear features, so held-out top-1 stays < 1.0. Stage
  2b `align_hidden` reshapes the student's internal basis (closed-form forward
  sweep) and lifts held-out top-1 to ~0.97 **when depths match** (demo: relu
  .646→.977, gelu .717→.971, tanh .803→.971). When the student is **shallower**
  than the teacher it improves but cannot fully close the gap (demo: relu
  .711→.844, gelu .742→.896, tanh .828→.959) — too few layers to track every
  teacher layer; this is an honest partial limit, documented in `theory.html` §7.
  The linear-`MLP` exact guarantee is unchanged (head-only already exact there).
  **Stage 3 `distill` closes this**: a depth-matched, adequately wide student
  reaches ~0.99 held-out top-1 after warm-started gradient distillation (demo:
  relu/gelu/tanh ≈ .996/.992/.996). The closed-form limit is no longer a wall once
  the gradient loop is allowed.
- **LLM-like (`TinyLM`) compounding is now mostly closed by `distill`.** Stage-1
  transfers all named transformer tensors and stage-2 fits the LM head per token
  position, but closed-form alone leaves a large per-token residual that COMPOUNDS
  under greedy autoregressive `generate` (stage-2 only: .516 → .336 across steps).
  Stage 3 `distill` (gradient fine-tune on fresh token probes) lifts per-token
  top-1 0.41→0.89 and flattens the decay curve (demo +distill: step1 .52→.88,
  step8 .34→.61). A residual still grows with horizon (deep transformer, finite
  steps), so it is *flattened, not erased* — the honest remaining limit is
  documented in `theory.html` §8.
- **Different vocabularies need Stage 0 (`reconcile_vocab`), not just stage 2.**
  An LM's output axis IS its vocabulary, so a teacher (V=64) and student (V=48)
  have non-comparable LM-head widths and per-id meanings. Without a `VocabMap`,
  `align_output`/`distill` would try to fit a 64-wide target onto a 48-wide head
  (shape error) or compare unrelated token columns. `reconcile_vocab` builds the
  `t_for_s` correspondence + a `(V_t,V_s)` projection; pass `vocab_map=` into
  `agreement`/`align_output`/`distill` and probe with `shared_token_probe`. Demo
  (`_demo_vocab_mismatch`, V 64→48 shared-prefix): per-token .410 → .706 (align)
  → .904 (distill). The built-in default map is the shared prefix — a real
  deployment must supply a tokenizer-string-derived `t_for_s`; student-only tokens
  (`-1`) get a zero target column (no teacher signal). `vocab_map=None` is a strict
  no-op, so all same-vocab MLP/TinyLM behaviour and tests are unchanged.
- **The realistic worst case is all three axes at once** (`_demo_combined_mismatch`):
  LM-head/vocab (72→48), middle-layer depth (4→2), and hidden width (80→40) all
  differ, AND the vocab map is *scrambled + partial* (40/48 mapped, arbitrary
  positions) instead of the clean shared prefix. Each earlier part isolates one
  axis; this stacks them. The scrambled map makes pre-alignment agreement collapse
  to near zero (per-token top1 ≈ .016 — the honest "two genuinely different LMs"
  baseline, vs .41 for the clean shared-prefix demo). The staged pipeline still
  recovers it: stage-0 reconcile → stage-2 head align ≈ .53 → stage-3 distill ≈ .86.
  Regression tests `test_combined_mismatch_baseline_is_near_zero` (base < .10) and
  `test_combined_mismatch_pipeline_recovers_all_three_axes` (monotone lift, distill
  > .8) lock this in. Use `_scrambled_vocab_map` (random teacher ids in random
  student slots) to build such a map — the shared-prefix default understates real
  tokenizer difficulty.
- **Capacity-sweep numbers are tied to the demo teacher** `MLP[32,128,96,64,10]`
  (held-out top-1: w4=.471, w8=.467, w16=.562, w48=1.000). If you change the demo
  teacher, resync `theory.html` §2 (SVD energy) and §6 (capacity bars).
- **`ferry_qwen.py` real-model results are an honest PoC ceiling, not ~1.0.** The
  toy TinyLM closes to ~0.89; the real Qwen3-0.6B→ferry-0.1B run only reaches
  per-token top1 ~0.00→0.20 (mse 13.0→4.2, cosine 0.07→0.48) in 120 CPU steps.
  Three real limits stack: student is ~17% params + half depth (capacity), vocab
  is 151936-wide (diffuse full-logit MSE signal), and random token probes are
  off-distribution for a real LM. The relative gain proves the architecture-changed
  transfer works *directionally*; it is not a fluency claim.
- **Qwen3 defaults to bfloat16; `ferry_qwen.build_student`/`load_teacher` force
  `.float()`.** Without a uniform float32 model, CPU autograd raises "Found dtype
  Float but expected BFloat16" in `distill`'s backward. Keep the `.float()` casts.
- **`ferry_qwen.py` reuses `ferry.distill` via the `LogitsModel` adapter** (HF
  returns a `CausalLMOutput`, but `agreement`/`distill` expect a raw logits
  tensor). Same-vocab teacher/student here → no `VocabMap` needed.
