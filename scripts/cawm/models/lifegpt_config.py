"""LifeGPT-style decoder-only causal transformer -- CONFIG ADAPTER, NOT PAPER-FAITHFUL.

Design spirit follows LifeGPT (Berkovich & Buehler 2025, npj AI 1:23;
arXiv:2409.12182): a decoder-only GPT over rasterised binary CA states with
absolute learned positional embeddings and no rule conditioning. The original
model is trained next-token on a FIXED rule (32x32 Life) and reports ~99.9%
per-cell accuracy on that fixed rule. This adapter keeps the architecture
family but is trained under OUR protocol (task L23, SC stream, 8x8 grid,
16-frame trajectories, 8-frame rollout from an 8-frame prefix, BCE on all 512
continuation cells) so the metric comparison in the Section-3 baseline figure
is honest.

Chosen simplification (stated per brief): instead of cell-by-cell next-token
autoregression within a frame, we use a per-frame block scheme -- each frame's
H*W cells are embedded as tokens in raster order, attention is block-causal at
frame granularity (a token in frame f attends to all tokens in frames <= f,
never to any later frame), and every token in frame f predicts the SAME cell
at frame f+1 via a linear logit head. This shifts the prediction target by one
frame instead of one raster position, removing within-frame autoregression,
but it preserves the LifeGPT design decisions that matter for the comparison:
decoder-only stack, raster token order, absolute learned positions over the
full raster sequence, and no rule conditioning. Strict frame-level causality
guarantees leakage safety: logits for output frame 8+j depend only on input
frames 0..7+j.

Interface is identical to `scripts/cawm/models/arcnn.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_MLP_RATIO = 2  # 2x (not GPT-2's 4x) to keep the default config inside 0.5M-1.5M params


class _CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an externally supplied boolean mask."""

    def __init__(self, dmodel, n_head):
        super().__init__()
        assert dmodel % n_head == 0
        self.n_head = n_head
        self.head_dim = dmodel // n_head
        self.qkv = nn.Linear(dmodel, 3 * dmodel)
        self.proj = nn.Linear(dmodel, dmodel)

    def forward(self, x, mask):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, S, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Fused kernel with the same masking semantics as eager
        # qk^T -> masked_fill -> softmax -> @v (bool mask: True = attend).
        # The eager path materialises (B, h, S, S) = (512, 6, 960, 960) and
        # its fp32 softmax under bf16 autocast wedged the CUDA context on
        # another GPU node when sibling lanes ran concurrently (2026-09-23
        # postmortem, faulthandler stack at this line).
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, S, D)
        return self.proj(y)


class _Block(nn.Module):
    """Pre-norm transformer block (attention + MLP)."""

    def __init__(self, dmodel, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(dmodel)
        self.attn = _CausalSelfAttention(dmodel, n_head)
        self.ln2 = nn.LayerNorm(dmodel)
        hidden = _MLP_RATIO * dmodel
        self.mlp = nn.Sequential(
            nn.Linear(dmodel, hidden),
            nn.GELU(),
            nn.Linear(hidden, dmodel),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class LifeGPTConfig(nn.Module):
    """LifeGPT-family decoder-only transformer, adapted to the CAWM protocol.

    Config adapter, not paper-faithful (see module docstring). Defaults are
    chosen so `trainable_param_count()` lands in 0.5M-1.5M (~1.39M).
    """

    def __init__(self, grid=8, arm="emergence", head="concat",
                 dmodel=192, n_layer=4, n_head=6):
        super().__init__()
        self.grid = grid
        self.arm = arm    # kept for interface parity with arcnn.py; unused here
        self.head = head  # kept for interface parity with arcnn.py; unused here
        self.hw = grid * grid
        self.max_frames = 16
        self.dmodel = dmodel

        # Token embedding over the binary cell alphabet {0, 1}.
        self.tok_emb = nn.Embedding(2, dmodel)
        # Absolute learned positions over the full raster sequence (LifeGPT-style).
        self.pos_emb = nn.Parameter(torch.zeros(self.max_frames * self.hw, dmodel))
        self.blocks = nn.ModuleList([_Block(dmodel, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dmodel)
        self.head_out = nn.Linear(dmodel, 1)

        # Block-causal mask at frame granularity: query token i may attend to
        # key token j iff frame(j) <= frame(i). Later frames are NEVER visible.
        f = torch.arange(self.max_frames * self.hw) // self.hw
        mask = f.unsqueeze(1) >= f.unsqueeze(0)
        self.register_buffer("_mask", mask, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _logits(self, frames_pm1):
        """(B,1,T,H,W) +-1 -> (B,T,H,W) logits; entry t predicts frame t+1."""
        B, _, T, H, W = frames_pm1.shape
        ids = ((frames_pm1.squeeze(1) + 1.0) * 0.5).round().long().clamp(0, 1)
        ids = ids.view(B, T * self.hw)
        x = self.tok_emb(ids) + self.pos_emb[: T * self.hw].unsqueeze(0)
        mask = self._mask[: T * self.hw, : T * self.hw]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head_out(x).view(B, T, H, W)

    def forward(self, frames_pm1):
        """(B,1,16,H,W) +-1 -> logits (B,8,H,W) for frames 8..15.

        Teacher-forced on TRUE inputs: frame 8+j is predicted from the tokens
        of frame 7+j, which (via the block-causal mask) see frames 0..7+j only.
        """
        logits = self._logits(frames_pm1[:, :, : self.max_frames - 1])  # frames 0..14
        return logits[:, 7:15]

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """(B,1,8,H,W) +-1 -> (B,8,H,W) uint8, autoregressive 8-step.

        Thresholds logits at 0 and feeds its own predictions back.
        """
        seq = prefix_pm1
        preds = []
        for _ in range(8):
            logits = self._logits(seq)[:, -1]  # (B,H,W): prediction for next frame
            bits = logits > 0
            preds.append(bits.to(torch.uint8))
            nxt = bits.to(seq.dtype).mul(2.0).sub(1.0)
            seq = torch.cat([seq, nxt.unsqueeze(1).unsqueeze(1)], dim=2)
        return torch.stack(preds, dim=1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
