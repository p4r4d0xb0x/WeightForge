"""Aster aster-1b reproduced in PyTorch, for data-free synthetic-probe KD.

Why this file exists
--------------------
Aster (``../SLM_FROM_BEGIN``) is a from-scratch SLM written in **Rust**, so its
forward pass has no autograd. To run Ferry's Stage-3 gradient distillation
(:func:`ferry.distill`) against it we need a *PyTorch* ``nn.Module`` that
reproduces Aster's forward **exactly** (so a loss reduced here actually transfers
back to the Rust runtime). This module is that faithful reproduction of the
aster-1b architecture:

  * hidden 1536, 26 layers, 16 query heads / 8 KV heads (GQA), head_dim 96
  * hybrid attention: layer ``i % 5 == 4`` (and the last layer) is **global**
    (full causal, RoPE theta 1e6); every other layer is **sliding** (window 512,
    RoPE theta 1e4) -- dual RoPE
  * interleaved (GPT-J style) RoPE over the full head_dim (NOT HF rotate_half)
  * attention logit soft-cap 50, final logit soft-cap 30
  * RMSNorm with **raw** gamma (NOT Gemma's ``1 + gamma``), eps 1e-5
  * GeGLU FFN with gelu-tanh (ffn_inner 6144), weight-tied LM head
  * embedding lookup with **no** ``sqrt(hidden)`` scaling (unlike HF Gemma)

The weight tensors are loaded from a ``params.safetensors`` in Aster's ``v2.*``
namespace (236 tensors) -- e.g. the transfer output
``./test_output/aster-1b-from-gemma-2-2b-embedmap/params.safetensors``.

What it does
------------
1. ``--verify``: a *parity* check. Loads a checkpoint, tokenizes a prompt with the
   real Aster tokenizer, greedy-decodes with this PyTorch forward, and prints the
   text. Compare against ``slm-cli chat ... --temperature 0 --repetition-penalty
   1.0`` on the same checkpoint: identical continuation == correct reproduction.
2. ``--distill`` (default): loads the transferred student, loads the **real
   Gemma-2B teacher** (HF), builds the tokenizer-string vocab map, and runs
   :func:`ferry.distill` (data-free token-probe KD, fresh probe per step, teacher
   logits projected into Aster's 48000-vocab). Saves the KD'd weights as a NEW
   ``params.safetensors`` for ``slm-cli chat`` before/after comparison.

Hard constraints (inherited): **CPU-only** (GPU forbidden, DEC-007), **data-free**
(synthetic token probes only -- no dataset, no disk data), and the live aster-1b
training checkpoint is **never** overwritten (outputs are NEW files).
"""

from __future__ import annotations

import argparse
import math
import os

# CPU-only, GPU forbidden (DEC-007). Must be set before torch picks a device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Reuse the data-free safetensors I/O + tokenizer-string vocab matching verbatim.
from transfer_gemma_to_aster import (  # noqa: E402
    build_vocab_map,
    load_safetensors,
    load_teacher,
    save_safetensors,
)

# Reuse Ferry's Stage-3 gradient distillation (data-free, fresh probe per step).
from ferry import distill  # noqa: E402


# --------------------------------------------------------------------------- #
# aster-1b config (configs/model/pretrain-1b.toml, verified against Rust source)
# --------------------------------------------------------------------------- #
class AsterConfig:
    vocab_size = 48000
    d_model = 1536
    n_layers = 26
    n_heads = 16
    n_kv_heads = 8
    head_dim = 96  # = d_model / n_heads
    ffn_inner = 6144  # ffn_mult 4.0
    rms_eps = 1e-5
    rope_theta = 10000.0  # sliding layers
    rope_theta_global = 1000000.0  # global layers
    sliding_window = 512
    attn_global_every = 5  # layer i % 5 == 4 is global; last layer forced global
    attn_logit_softcap = 50.0
    final_logit_softcap = 30.0

    @classmethod
    def is_global_layer(cls, i: int) -> bool:
        """Match model.rs ``layer_attn_kind``: last layer always global, else i%5==4."""
        if i + 1 == cls.n_layers:
            return True
        return i % cls.attn_global_every == cls.attn_global_every - 1


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """RMSNorm with raw gamma: ``gamma * x * rsqrt(mean(x^2) + eps)``.

    Note: NOT Gemma's ``(1 + weight)`` form -- Aster (norm.rs) uses the gamma
    value directly. Mean is over the hidden axis.
    """

    def __init__(self, dim: int, eps: float = AsterConfig.rms_eps) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * inv)


def _rope_tables(seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for interleaved RoPE: ``inv_freq[k] = theta^(-2k/head_dim)``.

    Returns ``(cos, sin)`` of shape ``(seq, head_dim/2)``.
    """
    half = head_dim // 2
    k = torch.arange(half, dtype=torch.float32)
    inv_freq = theta ** (-2.0 * k / head_dim)
    pos = torch.arange(seq, dtype=torch.float32)
    angle = torch.outer(pos, inv_freq)  # (seq, half)
    return torch.cos(angle), torch.sin(angle)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) RoPE on ``x`` of shape ``(b, s, n_heads, head_dim)``.

    Pair ``(2k, 2k+1)`` rotates by angle ``k``:
        out[2k]   = cos_k * x[2k]   - sin_k * x[2k+1]
        out[2k+1] = sin_k * x[2k]   + cos_k * x[2k+1]
    """
    x_even = x[..., 0::2]  # (b, s, h, half)
    x_odd = x[..., 1::2]
    cos = cos[None, :, None, :]  # (1, s, 1, half)
    sin = sin[None, :, None, :]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


class AsterBlock(nn.Module):
    def __init__(self, cfg: AsterConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_global = cfg.is_global_layer(layer_idx)
        h, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim  # 1536, 768
        self.attn_norm = RMSNorm(h)
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, kv, bias=False)
        self.v = nn.Linear(h, kv, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.ffn_norm = RMSNorm(h)
        self.ffn_gate = nn.Linear(h, cfg.ffn_inner, bias=False)
        self.ffn_up = nn.Linear(h, cfg.ffn_inner, bias=False)
        self.ffn_down = nn.Linear(cfg.ffn_inner, h, bias=False)

    def _attn(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, s, _ = x.shape
        hd, nh, nkv = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        q = self.q(x).view(b, s, nh, hd)
        k = self.k(x).view(b, s, nkv, hd)
        v = self.v(x).view(b, s, nkv, hd)

        theta = cfg.rope_theta_global if self.is_global else cfg.rope_theta
        cos, sin = _rope_tables(s, hd, theta)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: expand each KV head to heads_per_kv query heads (kv_head = h // 2).
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=2)  # (b, s, nh, hd)
        v = v.repeat_interleave(rep, dim=2)

        q = q.permute(0, 2, 1, 3)  # (b, nh, s, hd)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (b, nh, s, s)
        cap = cfg.attn_logit_softcap
        scores = cap * torch.tanh(scores / cap)  # soft-cap BEFORE mask
        scores = scores + mask  # additive -inf mask (causal [+ sliding])
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)  # (b, nh, s, hd)
        ctx = ctx.permute(0, 2, 1, 3).reshape(b, s, nh * hd)
        return self.o(ctx)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        g = F.gelu(self.ffn_gate(x), approximate="tanh")
        u = self.ffn_up(x)
        return self.ffn_down(g * u)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.attn_norm(x), mask)
        x = x + self._ffn(self.ffn_norm(x))
        return x


class AsterForCausalLM(nn.Module):
    """PyTorch reproduction of Aster aster-1b. ``forward(ids) -> logits``.

    ``ids`` is ``(b, s)`` long; output is ``(b, s, vocab)`` -- the LogitsModel-style
    signature :func:`ferry.distill` / :func:`ferry.agreement` expect.
    """

    def __init__(self, cfg: AsterConfig = AsterConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.d_model))
        self.blocks = nn.ModuleList(
            [AsterBlock(cfg, i) for i in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)

    def _masks(self, s: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Additive (global, sliding) masks of shape ``(1, 1, s, s)``."""
        i = torch.arange(s)[:, None]
        j = torch.arange(s)[None, :]
        neg = torch.finfo(dtype).min
        causal = j <= i
        glob = torch.where(causal, 0.0, neg).to(dtype)
        win = causal & (j >= i - (self.cfg.sliding_window - 1))
        slide = torch.where(win, 0.0, neg).to(dtype)
        return glob[None, None], slide[None, None]

    def final_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """Hidden state feeding the (tied) LM head: ``(b, s, d_model)``.

        Everything ``forward`` does *except* the final ``x @ embed.t()`` head and
        the logit soft-cap. Exposed so closed-form embed alignment
        (``align_aster_embed.py``) can regress the tied-embed output projection
        against these features without depending on the (Gemma-basis) embed.
        """
        x = F.embedding(ids, self.embed)  # no sqrt(hidden) scaling
        glob_mask, slide_mask = self._masks(ids.shape[1], x.dtype)
        for blk in self.blocks:
            mask = glob_mask if blk.is_global else slide_mask
            x = blk(x, mask)
        return self.final_norm(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.final_hidden(ids)
        logits = x @ self.embed.t()  # tied LM head
        cap = self.cfg.final_logit_softcap
        return cap * torch.tanh(logits / cap)


# --------------------------------------------------------------------------- #
# weight loading (v2.* namespace -> module params)
# --------------------------------------------------------------------------- #
def load_aster_weights(model: AsterForCausalLM, path: str) -> None:
    """Load an Aster ``params.safetensors`` (v2.* names) into ``model`` in place."""
    sd = load_safetensors(path)

    def grab(name: str) -> torch.Tensor:
        if name not in sd:
            raise KeyError(f"missing tensor {name} in {path}")
        return sd[name].float()

    with torch.no_grad():
        model.embed.copy_(grab("v2.embed.weight"))
        for i, blk in enumerate(model.blocks):
            p = f"v2.blocks.{i}."
            blk.attn_norm.gamma.copy_(grab(p + "attn_norm.gamma"))
            blk.q.weight.copy_(grab(p + "q.weight"))
            blk.k.weight.copy_(grab(p + "k.weight"))
            blk.v.weight.copy_(grab(p + "v.weight"))
            blk.o.weight.copy_(grab(p + "o.weight"))
            blk.ffn_norm.gamma.copy_(grab(p + "ffn_norm.gamma"))
            blk.ffn_gate.weight.copy_(grab(p + "ffn_gate.weight"))
            blk.ffn_up.weight.copy_(grab(p + "ffn_up.weight"))
            blk.ffn_down.weight.copy_(grab(p + "ffn_down.weight"))
        model.final_norm.gamma.copy_(grab("v2.final_norm.gamma"))


def save_aster_weights(model: AsterForCausalLM, path: str) -> None:
    """Write ``model`` back out in Aster's ``v2.*`` namespace (f32, 236 tensors)."""
    sd: dict[str, torch.Tensor] = {}
    sd["v2.embed.weight"] = model.embed.detach().float().contiguous()
    for i, blk in enumerate(model.blocks):
        p = f"v2.blocks.{i}."
        sd[p + "attn_norm.gamma"] = blk.attn_norm.gamma.detach().float().contiguous()
        sd[p + "q.weight"] = blk.q.weight.detach().float().contiguous()
        sd[p + "k.weight"] = blk.k.weight.detach().float().contiguous()
        sd[p + "v.weight"] = blk.v.weight.detach().float().contiguous()
        sd[p + "o.weight"] = blk.o.weight.detach().float().contiguous()
        sd[p + "ffn_norm.gamma"] = blk.ffn_norm.gamma.detach().float().contiguous()
        sd[p + "ffn_gate.weight"] = blk.ffn_gate.weight.detach().float().contiguous()
        sd[p + "ffn_up.weight"] = blk.ffn_up.weight.detach().float().contiguous()
        sd[p + "ffn_down.weight"] = blk.ffn_down.weight.detach().float().contiguous()
    sd["v2.final_norm.gamma"] = model.final_norm.gamma.detach().float().contiguous()
    save_safetensors(sd, path)


# --------------------------------------------------------------------------- #
# teacher (real Gemma-2B) + sparse vocab map
# --------------------------------------------------------------------------- #
class LogitsModel(nn.Module):
    """Wrap an HF causal LM so ``forward(ids) -> logits`` tensor (ferry contract)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.inner(ids).logits


def load_gemma_teacher(repo: str = "google/gemma-2-2b") -> LogitsModel:
    """Load the real Gemma-2B as a frozen CPU f32 teacher (LogitsModel)."""
    from transformers import AutoModelForCausalLM

    inner = (
        AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
        .float()  # force f32: CPU autograd rejects mixed bf16 in the loss path
        .eval()
        .to("cpu")
    )
    for p in inner.parameters():
        p.requires_grad_(False)
    return LogitsModel(inner)


class SparseVocabMap:
    """Memory-light duck-typed :class:`ferry.VocabMap`.

    ``ferry.build_vocab_map`` materializes a dense ``(V_t, V_s)`` selection matrix
    -- for the real sizes ``(256000, 48000)`` that is ~49 GB, infeasible. This
    object exposes the exact subset of the VocabMap interface that
    ``distill`` / ``agreement`` / ``shared_token_probe`` use (``t_for_s``,
    ``remap_ids``, ``project``) but implements ``project`` with an
    ``index_select`` gather instead of a matmul.
    """

    def __init__(self, t_for_s: torch.Tensor, size_t: int) -> None:
        self.t_for_s = t_for_s.long()
        self.size_t = size_t
        self.size_s = int(self.t_for_s.shape[0])
        self._gather = self.t_for_s.clamp_min(0)  # -1 -> 0 (masked out below)
        self._valid = (self.t_for_s >= 0).float()  # (V_s,)

    def remap_ids(self, student_ids: torch.Tensor) -> torch.Tensor:
        return self.t_for_s[student_ids].clamp_min(0)

    def project(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Map teacher logits ``(.., V_t)`` -> student space ``(.., V_s)``.

        Column ``j`` = teacher logit of the token student token ``j`` matches;
        student-only columns (``t_for_s == -1``) are zeroed (no teacher signal).
        """
        gathered = teacher_logits.index_select(-1, self._gather.to(teacher_logits.device))
        return gathered * self._valid.to(teacher_logits.dtype)


# --------------------------------------------------------------------------- #
# verify (parity vs Rust slm-cli) — greedy decode with the real tokenizer
# --------------------------------------------------------------------------- #
DEFAULT_TOKENIZER = "/data/0A_DATASET/L0_LLM/V3/TOKENIZER/tokenizer.json"


@torch.no_grad()
def verify_parity(params: str, tokenizer: str, prompt: str, n_new: int) -> str:
    """Greedy-decode ``n_new`` tokens from ``prompt`` with this PyTorch forward.

    Compare the printed text against ``slm-cli chat --model <dir> --temperature 0
    --repetition-penalty 1.0`` on the *same* params: identical continuation proves
    the PyTorch reproduction matches the Rust runtime.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer)
    model = AsterForCausalLM().eval()
    load_aster_weights(model, params)

    ids = tok.encode(prompt).ids
    cur = list(ids)
    for _ in range(n_new):
        inp = torch.tensor([cur], dtype=torch.long)
        logits = model(inp)  # (1, s, vocab)
        nxt = int(logits[0, -1].argmax().item())
        cur.append(nxt)
    return tok.decode(cur)


# --------------------------------------------------------------------------- #
# distill (data-free synthetic-probe KD against the real Gemma-2B teacher)
# --------------------------------------------------------------------------- #
def run_distill(args: argparse.Namespace) -> None:
    cfg = AsterConfig()
    print(f"[1/5] loading student (transferred): {args.student_params}")
    student = AsterForCausalLM(cfg)
    load_aster_weights(student, args.student_params)

    print(f"[2/5] loading real teacher: {args.teacher} (CPU f32)")
    teacher = load_gemma_teacher(args.teacher)

    print(f"[3/5] building vocab map: {args.student_tokenizer} <-> {args.teacher}")
    t_for_s_list, stats = build_vocab_map(args.student_tokenizer, _teacher_tok(args.teacher))
    t_for_s = torch.tensor(t_for_s_list, dtype=torch.long)
    vmap = SparseVocabMap(t_for_s, size_t=stats["teacher_vocab"])
    print(
        f"      matched {stats['matched']}/{stats['student_vocab']} "
        f"({stats['match_frac']*100:.1f}%), korean {stats['korean_matched']}"
    )

    print(
        f"[4/5] distill (data-free token KD): steps={args.steps} batch={args.batch} "
        f"seq={args.seq} lr={args.lr}"
    )
    before = _quick_agreement(teacher, student, cfg, args, vmap)
    print(f"      before: top1={before['top1_agree']:.4f} mse={before['mse']:.3f} "
          f"cosine={before['cosine']:.4f}")
    result = distill(
        student, teacher,
        vocab=cfg.vocab_size, seq=args.seq,
        steps=args.steps, batch=args.batch, lr=args.lr, seed=0,
        vocab_map=vmap,
    )
    print(f"      after (held-out): top1={result['top1_agree']:.4f} "
          f"mse={result['mse']:.3f} cosine={result['cosine']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "params.safetensors")
    print(f"[5/5] saving KD'd weights: {out_path}")
    save_aster_weights(student, out_path)
    print("done. compare with slm-cli chat before/after (see module docstring).")


def _teacher_tok(repo: str) -> str:
    """Locate the teacher's tokenizer.json in its cached HF snapshot."""
    from transfer_gemma_to_aster import _resolve_snapshot

    return os.path.join(_resolve_snapshot(repo), "tokenizer.json")


@torch.no_grad()
def _quick_agreement(teacher, student, cfg, args, vmap) -> dict:
    from ferry import agreement, shared_token_probe

    probe = shared_token_probe(args.batch, args.seq, vmap, seed=-99)
    return agreement(teacher, student, probe, vmap)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--student-params",
        default="./test_output/aster-1b-from-gemma-2-2b-embedmap/params.safetensors",
        help="transferred Aster params to start KD from",
    )
    ap.add_argument("--teacher", default="google/gemma-2-2b")
    ap.add_argument("--student-tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", default="./test_output/aster-1b-kd-gemma-2-2b")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    # verify mode
    ap.add_argument("--verify", metavar="PARAMS", default=None,
                    help="parity mode: greedy-decode from a params file and print text")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--prompt", default="옛날 옛적에")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    if args.verify is not None:
        text = verify_parity(args.verify, args.tokenizer, args.prompt, args.max_new_tokens)
        print("--- PyTorch Aster greedy continuation ---")
        print(text)
        return

    run_distill(args)


if __name__ == "__main__":
    main()
