"""ART with KV-shift pairing (E4.7): no temporal conv; pairing by index shift.

Ladder rung between the pairing-conv scaffold (`art`, 1.0) and the unpaired
two-hop arm (`art_twohop`, 0.80). Question registered in
docs/EXPERIMENT_PLAN.md E4.7: is the load-bearing scaffold the 2-frame
*convolution*, or the situation-outcome *alignment* itself (however
implemented)?

Construction (registered in docs/SETTINGS.md):
  A single-frame situation detector (shared between tokens and queries):
  Conv2d(1->2,3) -> split -> conjoin -> 18-d (s,n) code per cell, applied to
  every frame independently. Token for transition (t -> t+1), cell c is built
  by gating, not convolution:

      tok[2*(9s+n) + o] = code_t[slot] * 1[frame t+1 at c == outcome o]

  i.e. the KEY carries frame t's situation and the VALUE carries frame t+1's
  outcome, aligned by shifting the value source one frame forward along the
  token axis ("KV shift"). 7 transitions x H*W = 448 tokens, identical layout
  to ArtInduction (channel 2*(9s+n)+o).

  Attention, head (with the L4 default pathway), and rollout are inherited
  verbatim from ArtInduction. Equivalence: with analytic weights the tokens
  are bit-equal to the pairing conv's (test_kvshift_tokens_match_pairing_conv)
  — the conv and the index shift build the SAME data structure; the rung
  therefore isolates "pairing as alignment" from "pairing as conv feature
  extraction".

Arms:
  existence — detector, Wk, Wv analytic/frozen; only the head trains (913).
  emergence — all learned except the frozen attention scale 20.0 (2625).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .art_induction import ATTN_SCALE, HEAD_HIDDEN, ArtInduction
from .constructive_cnn import _build_analytic_b, _circ2d


class ArtKVShift(ArtInduction):
    def __init__(self, grid=8, arm="emergence"):
        nn.Module.__init__(self)
        assert arm in ("existence", "emergence")
        assert grid & (grid - 1) == 0 and grid >= 4
        self.grid = grid
        self.arm = arm
        self.attn_scale = ATTN_SCALE

        # ---- shared single-frame situation detector (tokens AND queries) ----
        self.feat_s = nn.Conv2d(1, 2, 3)
        if arm == "existence":
            self.split_s = nn.Conv2d(2, 22, 1)
            self.conjoin_s = nn.Conv2d(22, 18, 1)
            _build_analytic_b(self.feat_s, self.split_s, self.conjoin_s)
            for m in (self.feat_s, self.split_s, self.conjoin_s):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        else:
            self.split_s = nn.Conv2d(2, 18, 1)
            self.conjoin_s = nn.Conv2d(18, 18, 1)

        # ---- fixed/learned k,v projections (identical to ArtInduction) ----
        if arm == "existence":
            wk = torch.zeros(36, 18)
            wv = torch.zeros(36, 18)
            for s in (0, 1):
                for n in range(9):
                    c = 9 * s + n
                    wk[2 * c, c] = 1.0
                    wk[2 * c + 1, c] = 1.0
                    wv[2 * c + 1, c] = 1.0
            self.register_buffer("Wk", wk)
            self.register_buffer("Wv", wv)
        else:
            self.Wk = nn.Linear(36, 18, bias=False)
            self.Wv = nn.Linear(36, 18, bias=False)

        # ---- head: [attention out | code] -> logit (L4 default pathway) ----
        self.head = nn.Sequential(
            nn.Linear(36, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, 1)
        )

    # ---- forward pieces (attention/head/rollout inherited) ----
    def _codes(self, frames_pm1):
        """frames_pm1: (N,1,H,W) ±1 -> codes (N,18,H,W); the shared detector."""
        h = self.feat_s(_circ2d(frames_pm1, 1))
        h = F.relu(self.split_s(h))
        return F.relu(self.conjoin_s(h))

    def _tokens(self, prefix_pm1):
        """prefix_pm1: (B,1,8,H,W) ±1 -> tokens (B, 7*H*W, 36) by KV shift.

        Token (t, cell) = key from frame t's code, value gated by frame t+1's
        outcome bit: channel 2*(9s+n)+o = code_t[slot] * 1[outcome == o]."""
        B = prefix_pm1.shape[0]
        g = self.grid
        p = prefix_pm1[:, 0]                                  # (B,8,H,W)
        codes = self._codes(p.reshape(B * 8, 1, g, g))        # (B*8,18,H,W)
        codes = codes.reshape(B, 8, 18, g * g)
        key = codes[:, :-1]                                   # (B,7,18,HW)
        out01 = ((p[:, 1:] + 1.0) / 2.0).reshape(B, 7, 1, g * g)
        tok = torch.stack([key * (1.0 - out01), key * out01], dim=3)
        tok = tok.reshape(B, 7, 36, g * g)                    # ch = 2*slot+o
        return tok.permute(0, 1, 3, 2).reshape(B, 7 * g * g, 36)
