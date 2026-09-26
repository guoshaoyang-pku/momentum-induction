"""Burtsev-style ECA transformer config adapter (Section-3 baseline figure).

Config adapter for Burtsev (2024, arXiv:2412.01417, "Learning Elementary
Cellular Automata with Transformers"), who trains a GPT-style decoder on
serialised 1-D ECA orbits ([SEP]-delimited states, rule held out at test)
and reports high per-bit accuracy that decays with orbit length.

This is a *config adapter*, not a paper-faithful replica: his task is 1-D
ECA, ours is the 2-D 18-rule family on 8x8x16 trajectories. We keep the
1-D serialisation spirit on our task -- the trajectory is flattened to a
raster token stream (row-major within frame, frame-major across frames)
exactly like ``art_vanilla`` -- with Burtsev-flavoured choices:

  * post-LN transformer blocks (his configs use the original pre-GPT2
    normalisation placement, no final LayerNorm),
  * sinusoidal absolute positions, no RoPE,
  * ReLU (not GELU) MLPs,
  * deeper-narrower shape (his depth result): d_model=128, n_layer=8,
    n_head=4, mlp_ratio=1, landing in the 0.3M-1M parameter budget
    (~0.8M at defaults).

Leakage safety: prediction for raster token t conditions on strictly
earlier tokens only, implemented by right-shifting the token stream behind
a learned BOS embedding and applying a standard causal attention mask.

Interface contract matches ``scripts/cawm/models/arcnn.py``:
``forward`` / ``rollout`` / ``num_params`` identical in shape and dtype.
Registered with the trainer as ``burtsev_cfg``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    """Standard fixed sinusoidal absolute position table, shape (max_len, d_model)."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _causal_mask(n: int, device: torch.device, inclusive: bool = False) -> torch.Tensor:
    """Boolean attention mask, True = disallowed.

    ``inclusive=False``: strictly upper triangle (position k sees 0..k-1,
    the classic decoder mask used when the logit at slot k predicts token
    k+1).  ``inclusive=True``: also masks the diagonal, so position k sees
    only 0..k-1 and the logit at slot k predicts token k (used here, where
    each slot embeds the CURRENT token)."""
    return torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=device),
        diagonal=1 if not inclusive else 0,
    )


class _PostLNBlock(nn.Module):
    """Original-transformer (post-LN) block with ReLU MLP."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = self.ln1(h + self.drop(a))
        h = self.ln2(h + self.drop(self.mlp(h)))
        return h


class BurtsevConfigWM(nn.Module):
    """Burtsev-flavoured raster-autoregressive world model for the 2-D task.

    ``forward(x)`` with ``x`` of shape (B, T, H, W) in {0, 1} returns per-cell
    logits of shape (B, T, H, W), dtype float32, where the logit for raster
    position t conditions on strictly earlier raster tokens only.

    ``rollout(frames, n_steps)`` autoregressively generates ``n_steps`` future
    frames token-by-token (greedy, logit > 0), returning (B, n_steps, H, W)
    in the dtype of ``frames``.
    """

    def __init__(
        self,
        grid: int = 8,
        arm: str = "emergence",
        head: str = "concat",
        horizon: int = 16,
        dmodel: int = 128,
        n_layer: int = 8,
        n_head: int = 4,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.arm = arm
        self.head_type = head
        self.horizon = horizon
        self.d_model = dmodel
        n_tokens = horizon * grid * grid
        self.max_seq_len = max_seq_len if max_seq_len is not None else n_tokens + 1

        self.tok_emb = nn.Embedding(2, dmodel)
        self.bos = nn.Parameter(torch.zeros(1, 1, dmodel))
        nn.init.normal_(self.bos, std=0.02)
        self.register_buffer(
            "pos_table", _sinusoidal_positions(self.max_seq_len, dmodel)
        )

        self.blocks = nn.ModuleList(
            _PostLNBlock(dmodel, n_head, mlp_ratio * dmodel, dropout)
            for _ in range(n_layer)
        )
        self.head = nn.Linear(dmodel, 1)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Context tokens (B, N) int64 -> logits (B, N + 1) float32.

        Output position i is the logit for token i given tokens[:, :i] only
        (position 0 sees only the BOS embedding).
        """
        b, n = tokens.shape
        if n + 1 > self.max_seq_len:
            raise ValueError(
                f"sequence length {n + 1} exceeds max_seq_len {self.max_seq_len}"
            )
        h = torch.cat([self.bos.expand(b, -1, -1), self.tok_emb(tokens)], dim=1)
        h = h + self.pos_table[: n + 1].unsqueeze(0)
        # Strict prefix conditioning: position k may attend to BOS and tokens
        # 0..k-1, but NOT to the token embedded at its own slot (diagonal
        # masked too).  Output position k is then a pure function of the
        # length-k prefix, so a logit computed in any longer sequence is
        # bit-identical to the incremental-rollout computation (the
        # prefix-consistency invariant the figure's honest-comparison claim
        # rests on).
        mask = _causal_mask(n + 1, tokens.device, inclusive=True)
        # Row 0 (the BOS slot) must see its own embedding: with a fully masked
        # row the fp32 attention kernel yields NaN softmax outputs (bf16
        # kernels silently zero-fill instead), so fp32 eval collapsed to the
        # marginal predictor while bf16 training looked healthy (2026-09-23
        # postmortem). BOS carries no token content, so unmasking [0, 0]
        # keeps strict prefix conditioning intact.
        mask[0, 0] = False
        for block in self.blocks:
            h = block(h, mask)
        return self.head(h).squeeze(-1)

    # ------------------------------------------------------------------ #
    # interface contract (matches arcnn.py)
    # ------------------------------------------------------------------ #

    def forward(self, frames_pm1):
        """(B,1,16,H,W) +-1 -> logits (B,8,H,W) for frames 8..15, teacher-forced.

        Raster autoregression with a shift-by-one-token head: token k's logit
        predicts raster token k+1. Output frame 8+j is predicted from input
        frames 0..7+j only (strict causality)."""
        b = frames_pm1.shape[0]
        t, h, w = 16, self.grid, self.grid
        x01 = ((frames_pm1[:, 0] + 1.0) * 0.5).reshape(b, t * h * w).long()
        logits = self._token_logits(x01)                       # (B, N+1)
        # With the inclusive causal mask, output position k is a pure
        # function of the length-k prefix, and the logit at output index k
        # predicts raster token k: the prediction for raster token m lives at
        # output index m-1.  Frame 8+j covers raster tokens
        # [(8+j)*n, (9+j)*n); their logits sit one index earlier, and raster
        # token (8+j)*n-1 is the LAST token of input frame 7+j - so output
        # frame 8+j depends only on input frames 0..7+j (leakage-safe).
        n = h * w
        outs = []
        for j in range(8):
            lo, hi = (8 + j) * n - 1, (9 + j) * n - 1
            outs.append(logits[:, lo:hi].view(b, h, w))
        return torch.stack(outs, dim=1).float()                # (B,8,H,W)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """(B,1,8,H,W) +-1 -> (B,8,H,W) uint8, greedy autoregressive 8 steps."""
        b, _, t0, h, w = prefix_pm1.shape
        x01 = ((prefix_pm1[:, 0] + 1.0) * 0.5).reshape(b, t0 * h * w).long()
        n_new = 8 * h * w
        tokens = x01
        for _ in range(n_new):
            logits = self._token_logits(tokens)
            nxt = (logits[:, -1:] > 0.0).long()
            tokens = torch.cat([tokens, nxt], dim=1)
        gen = tokens[:, t0 * h * w:].view(b, 8, h, w)
        return gen.to(torch.uint8)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())



def build(**kwargs) -> BurtsevConfigWM:
    """Factory for the trainer registry."""
    return BurtsevConfigWM(**kwargs)


try:  # self-register with the training entry point when importable
    from ..train import register_model

    register_model("burtsev_cfg", build)
except Exception:  # pragma: no cover - registry optional at import time
    pass
