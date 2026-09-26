"""AutomataGPT-style config adapter, no-rule-matrix variant.

Reference: Berkovich, David & Buehler, "AutomataGPT" (arXiv:2506.17333).
The paper conditions a decoder-only transformer on the explicit 18-bit rule
matrix (RM) together with the initial condition and predicts the next game
state; with the RM supplied it reports 98.5% PERFECT one-step forecasts on
unseen rules from the same family. That number may be quoted as a reference
point in the Section-3 figure, but it is NOT reproducible under our L3
protocol, where the rule must be inferred from the observed 8-frame prefix
and is never supplied to the model.

Simplifications vs the paper:
  * The RM input is removed entirely — this is exactly the ablation that
    separates "rule supplied" from "rule inferred", which is what our
    protocol measures.
  * Frame tokens are whole 64-cell boards (one token per frame, embedded by
    a single linear patch embedder) instead of the paper's tokenisation; a
    causal transformer runs over the 16 frame slots with learned absolute
    positions, and a linear head maps each slot back to 64 cell logits.

Interface contract (identical to scripts/cawm/models/arcnn.py):
  forward(frames_pm1): (B, 1, 16, H, W) in {-1, +1} -> logits (B, 8, H, W),
      teacher-forced; output index j is the prediction for frame 8 + j and
      conditions only on frames 0 .. 7 + j (causal mask).
  rollout(prefix_pm1): (B, 1, 8, H, W) in {-1, +1} -> (B, 8, H, W) uint8,
      8 autoregressive steps fed back as ±1 boards.
"""

import torch
import torch.nn as nn


class _CausalBlock(nn.Module):
    """Pre-norm transformer block with a boolean causal attention mask."""

    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class AutomataGPTConfig(nn.Module):
    """AutomataGPT-flavoured causal transformer over whole-board frame tokens."""

    T_TOTAL = 16
    T_PREFIX = 8

    def __init__(
        self,
        grid=8,
        arm="emergence",
        head="concat",
        d_model=160,
        n_layers=4,
        n_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        if head != "concat":
            raise ValueError(
                "AutomataGPTConfig supports head='concat' only, got %r" % (head,)
            )
        self.grid = grid
        self.arm = arm
        self.head = head
        self.n_cells = grid * grid
        self.d_model = d_model

        self.embed = nn.Linear(self.n_cells, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.T_TOTAL, d_model))
        self.blocks = nn.ModuleList(
            [_CausalBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.decode = nn.Linear(d_model, self.n_cells)

        nn.init.trunc_normal_(self.pos, std=0.02)

    @staticmethod
    def _causal_mask(t, device):
        # True = masked (position j may not attend to positions > j).
        return torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1
        )

    def _encode(self, seq):
        # seq: (B, T, n_cells) in {-1, +1}
        t = seq.shape[1]
        h = self.embed(seq) + self.pos[:, :t]
        mask = self._causal_mask(t, seq.device)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.norm(h)

    def forward(self, frames_pm1):
        """(B, 1, 16, H, W) ±1 -> teacher-forced logits (B, 8, H, W)."""
        b, c, t, hh, ww = frames_pm1.shape
        assert c == 1 and t == self.T_TOTAL and hh == self.grid and ww == self.grid, (
            "expected (B, 1, %d, %d, %d), got %s"
            % (self.T_TOTAL, self.grid, self.grid, tuple(frames_pm1.shape))
        )
        seq = frames_pm1[:, 0].reshape(b, t, self.n_cells)
        h = self._encode(seq)
        # Slot j predicts frame j + 1; targets are frames 8..15 -> slots 7..14.
        logits = self.decode(h[:, self.T_PREFIX - 1 : self.T_TOTAL - 1])
        return logits.reshape(b, self.T_PREFIX, hh, ww)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """(B, 1, 8, H, W) ±1 -> (B, 8, H, W) uint8, 8 autoregressive steps."""
        b, c, t, hh, ww = prefix_pm1.shape
        assert c == 1 and t == self.T_PREFIX and hh == self.grid and ww == self.grid, (
            "expected (B, 1, %d, %d, %d), got %s"
            % (self.T_PREFIX, self.grid, self.grid, tuple(prefix_pm1.shape))
        )
        seq = prefix_pm1[:, 0].reshape(b, t, self.n_cells).float()
        outs = []
        for _ in range(self.T_PREFIX):
            h = self._encode(seq)
            logits = self.decode(h[:, -1])
            bits = (logits > 0).to(torch.uint8)
            outs.append(bits.reshape(b, hh, ww))
            nxt = bits.float().mul(2.0).sub(1.0).reshape(b, 1, self.n_cells)
            seq = torch.cat([seq, nxt], dim=1)
        return torch.stack(outs, dim=1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
