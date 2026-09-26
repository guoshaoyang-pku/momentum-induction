"""R1 two-hop ART (E4.4, arm emergence only): the ladder rung between the
structured ART (R2) and the generic transformer (R0).

Unlike ArtInduction, tokens are UNPAIRED: a single-frame detector stack
(Conv2d(1->2,3x3) -> split(2->18) -> conjoin(18->18), all learned) is applied
to EACH prefix frame separately -> 8 frames x H*W = 512 tokens of 18-d
(situation codes only, no situation->outcome pairing). Each token is projected
18->d=64 and given a learned spatio-temporal position embedding. Two layers of
learned multi-head attention (4 heads) + a small FFN process the tokens; a
learned scalar temperature (init 1.0) scales the query-token match.

Query = the same detector stack on the current frame + its position embedding;
head = the same 24-hidden MLP as ArtInduction ([attn out | code] -> logit).
Rollout mirrors ArtInduction: tokens are computed once from the prefix and
frozen; only the query advances. This is the "two-hop" circuit: hop 1 attends
from the query situation to matching prefix situations, hop 2 must recover the
outcome from the following frame's token — a circuit the structured ART gets
for free via the pairing convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constructive_cnn import _circ2d

D = 64
NHEAD = 4
HEAD_HIDDEN = 24


class AttnBlock(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ffn(self.ln2(x))


class ArtTwoHop(nn.Module):
    def __init__(self, grid=8, arm="emergence", head="concat", temp_mode="learned",
                 d=64, pos="learned"):
        super().__init__()
        assert arm == "emergence", "art_twohop is an emergence-only rung"
        assert grid & (grid - 1) == 0 and grid >= 4
        assert temp_mode in ("learned", "frozen20")
        assert d in (64, 128)
        # E12.2: pos="none" drops both position tables (no positional signal at
        # all: the unmasked blocks then see an unordered set of tokens, so the
        # predecessor fetch cannot be learned). "learned" = registered, bit-identical.
        assert pos in ("learned", "none")
        self.grid = grid
        self.arm = arm
        self.head_type = head
        self.temp_mode = temp_mode
        self.pos = pos
        self.ncells = grid * grid

        # ---- single-frame detector stack (learned) ----
        self.feat_b = nn.Conv2d(1, 2, 3)
        self.split_b = nn.Conv2d(2, 18, 1)
        self.conjoin_b = nn.Conv2d(18, 18, 1)

        # ---- token / query projections + learned position embeddings ----
        self.tok_proj = nn.Linear(18, d)
        self.q_proj = nn.Linear(18, d)
        # spatio-temporal PE: 8 frames x H*W cells for tokens; H*W for query
        if pos == "learned":
            self.tok_pos = nn.Parameter(torch.zeros(1, 8 * self.ncells, d))
            self.q_pos = nn.Parameter(torch.zeros(1, self.ncells, d))
            for p in (self.tok_pos, self.q_pos):
                nn.init.normal_(p, std=0.02)

        self.blocks = nn.ModuleList([AttnBlock(d, NHEAD) for _ in range(2)])
        # E4.5 dose point: learned temperature (default, legacy = bit-compat)
        # vs the frozen sharp scale=20 of the scaffolded ART. Diagnostic
        # 2026-08-30: learned temp settles at 0.80 -> retrieval stays soft.
        if temp_mode == "learned":
            self.temp = nn.Parameter(torch.ones(1))          # learned temperature
        else:  # frozen20
            self.register_buffer("temp", torch.full((1,), 20.0))
        self.head = nn.Sequential(
            nn.Linear(d + 18, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, 1)
        )

    # ---- pieces ----
    def _codes(self, frames_pm1):
        """frames_pm1: (N,1,H,W) ±1 -> codes (N,18,H,W)."""
        h = self.feat_b(_circ2d(frames_pm1, 1))
        h = F.relu(self.split_b(h))
        return F.relu(self.conjoin_b(h))

    def _tokens(self, prefix_pm1):
        """prefix_pm1: (B,1,8,H,W) ±1 -> tokens (B, 8*H*W, D) with PE."""
        B = prefix_pm1.shape[0]
        p = prefix_pm1[:, 0].reshape(B * 8, 1, self.grid, self.grid)
        codes = self._codes(p)                               # (B*8,18,H,W)
        codes = codes.reshape(B, 8, 18, self.ncells)
        codes = codes.permute(0, 1, 3, 2).reshape(B, 8 * self.ncells, 18)
        tok = self.tok_proj(codes)                           # (B,512,D)
        if self.pos == "learned":
            tok = tok + self.tok_pos
        for b in self.blocks:
            tok = b(tok)
        return tok

    def _query(self, frame_pm1):
        """frame_pm1: (B,1,H,W) ±1 -> (q (B,H*W,D), code (B,H*W,18))."""
        B = frame_pm1.shape[0]
        codes = self._codes(frame_pm1)                       # (B,18,H,W)
        codes = codes.reshape(B, 18, self.ncells).permute(0, 2, 1)  # (B,HW,18)
        q = self.q_proj(codes)                               # (B,HW,D)
        if self.pos == "learned":
            q = q + self.q_pos
        return q, codes

    def _logits_from(self, tok, frame_pm1):
        q, codes = self._query(frame_pm1)
        scores = torch.einsum("bqd,bkd->bqk", q, tok) * self.temp
        w = scores.softmax(dim=-1)
        out = torch.einsum("bqk,bkd->bqd", w, tok)           # (B,HW,D)
        z = torch.cat([out, codes], dim=-1)                  # (B,HW,D+18)
        return self.head(z).squeeze(-1)                      # (B,HW)

    def forward(self, frames_pm1):
        """Teacher-forced parallel prediction.
        frames_pm1: (B,1,16,H,W) ±1 -> logits (B,8,H,W) for frames 8..15."""
        B = frames_pm1.shape[0]
        tok = self._tokens(frames_pm1[:, :, :8])             # (B,512,D)
        outs = []
        for j in range(8):
            cur = frames_pm1[:, :, 7 + j]                    # (B,1,H,W)
            outs.append(self._logits_from(tok, cur))
        return torch.stack(outs, dim=1).reshape(B, 8, self.grid, self.grid)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """Autoregressive 8-step rollout; tokens frozen at the prefix.
        prefix_pm1: (B,1,8,H,W) ±1 -> predictions (B,8,H,W) uint8."""
        tok = self._tokens(prefix_pm1)
        x = prefix_pm1[:, :, 7]                              # (B,1,H,W)
        preds = []
        for _ in range(8):
            logits = self._logits_from(tok, x).reshape(-1, self.grid, self.grid)
            pred = (logits > 0).to(torch.uint8)
            preds.append(pred)
            x = (2.0 * pred.float() - 1.0).unsqueeze(1)
        return torch.stack(preds, dim=1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
