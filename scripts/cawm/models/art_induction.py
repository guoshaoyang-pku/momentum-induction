"""ART with an induction head (momentum induction, testbed #2).

Design frozen in docs/SETTINGS.md (registered 2026-08-29 chat rounds):

  Token stack (Network A, one forward per sequence): the constructive two-layer
  detection stack (Conv3d(1->3,(2,3,3)) -> split -> conjoin) applied to each
  adjacent frame pair (t, t+1) of the 8-frame prefix -> 7 transitions x H*W
  cells = 448 tokens, each a 36-d slot indicator (channel 2*(9s+n)+o).
  No aggregation: tokens stay per-(transition, cell).

  Query (Network B): 18-d one-hot (s,n) code of the current frame, per cell.

  Attention (single head, NO positional encoding — pure content matching):
    k_j = W_k token_j  (fixed selection of the two context channels 2c, 2c+1
         -> k_j[c] = 1 iff token j has context c),
    v_j = W_v token_j  (fixed selection of outcome channels -> v_j[c] = 1 iff
         token j is context c with outcome 1),
    scores = attn_scale * (q . k)  -> softmax over 448 tokens -> out = w @ v.
  With exact one-hot tokens, a cell whose context appears m times in the
  prefix gets out[ctx] = mean outcome of those m transitions and ~0 elsewhere;
  an unseen context gets out = 0 (all scores tie -> the global mean, which is
  0 for unseen channels). Exactness: with 447 non-match tokens, softmax
  leakage per match-set is 442·e^{-scale}; scale 10 leaves a ~2% relative
  bias, scale 20 gives 9e-7 — frozen at 20.0 for 1e-6-exact induction
  (registered ablation axis).

  Head (per cell): MLP on [attention out (18) | code (18)] -> logit. The
  all-zero-attention case is the L4 default pathway (learned bias).

Arms:
  existence — A/B stacks, W_k, W_v, and the scale are analytic/frozen; only
              the head is trained (913 params).
  emergence — same structure, A/B stacks, W_k, W_v learned from init; scale
              still frozen (3798 trainable params).

Rollout: tokens computed ONCE from the prefix (A forward once, reused — same
discipline as the constructive CNN); the query tracks the evolving frame.

E6.5 variants (all default OFF = bit-identical to the registered ART):
  global_path — a parallel 36-d GLOBAL evidence-count pathway: sum of the
      36-d tokens over all 448 positions (= per-(context,outcome) counts,
      exactly the CNN condition vector), concatenated into the head input
      [attn out 18 | code 18 | global 36]. Gives the head an explicit
      "how much evidence for slot c" silence signal and a path to PARTNER
      slots (L4B negation pairs) that same-slot recall cannot reach.
  null_token — one learned extra key/value token appended to the 448
      evidence tokens (attention sink). When no evidence token matches the
      query, the sink can take the softmax mass and return a learned
      "no evidence" signature v_null, instead of the uniform average of
      all v's. Explicit silence signal, but NO partner-slot path — the
      diagnostic partner of global_path (predicted: fixes L4A, not L4B).
  head_hidden — head MLP width (default 24; capacity control).
  global_norm — scaling of the global count vector: none (raw counts, up to
      448 — first launch; hurt L4B optimisation: 0.66), log1p (bounded,
      preserves the 0-vs-1 silence contrast), mean (count/448).

E21 depth (default reads=1 = bit-identical to the registered ART): R
sequential attention reads over the same frozen prefix memory. Read r >= 2
has its own key/value projections and a residual linear query update from
[query_{r-1} | read_{r-1}]; the head sees [read_1 | ... | read_R | code].
reads=2 is exactly the E17 two-read KV-shift construction (same module
names and creation order), shared here so that the pairing-conv ART and the
KV-shift model scale depth with one implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constructive_cnn import _build_analytic_a, _build_analytic_b, _circ2d, _circ3d

ATTN_SCALE = 20.0  # 448-token exactness: 442·e^-20 ~ 9e-7 leakage; ablation axis
HEAD_HIDDEN = 24


class ArtInduction(nn.Module):
    # class-level defaults so subclasses that bypass __init__ (ArtKVShift)
    # keep the registered behaviour
    global_path = False
    null_token = False
    global_norm = "none"
    reads = 1

    def __init__(self, grid=8, arm="emergence", global_path=False,
                 null_token=False, head_hidden=HEAD_HIDDEN, global_norm="none",
                 reads=1):
        super().__init__()
        assert arm in ("existence", "emergence")
        assert grid & (grid - 1) == 0 and grid >= 4
        self.grid = grid
        self.arm = arm
        self.attn_scale = ATTN_SCALE
        self.global_path = bool(global_path)
        self.null_token = bool(null_token)
        assert global_norm in ("none", "log1p", "mean")
        self.global_norm = global_norm
        assert reads >= 1
        assert reads == 1 or (arm == "emergence" and not global_path
                              and not null_token)
        self.reads = int(reads)

        # ---- token stack (Network A on transition pairs) ----
        self.feat_a = nn.Conv3d(1, 3, kernel_size=(2, 3, 3))
        if arm == "existence":
            self.split_a = nn.Conv3d(3, 24, 1)
            self.conjoin_a = nn.Conv3d(24, 36, 1)
            _build_analytic_a(self.feat_a, self.split_a, self.conjoin_a)
            for m in (self.feat_a, self.split_a, self.conjoin_a):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        else:
            self.split_a = nn.Conv3d(3, 27, 1)
            self.conjoin_a = nn.Conv3d(27, 36, 1)

        # ---- query stack (Network B on the current frame) ----
        self.feat_b = nn.Conv2d(1, 2, 3)
        if arm == "existence":
            self.split_b = nn.Conv2d(2, 22, 1)
            self.conjoin_b = nn.Conv2d(22, 18, 1)
            _build_analytic_b(self.feat_b, self.split_b, self.conjoin_b)
            for m in (self.feat_b, self.split_b, self.conjoin_b):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        else:
            self.split_b = nn.Conv2d(2, 18, 1)
            self.conjoin_b = nn.Conv2d(18, 18, 1)

        # ---- fixed/learned k,v projections (context / outcome selectors) ----
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

        # ---- E6.5 optional pieces (constructed AFTER the registered modules so
        # the registered init stream is untouched when they are off) ----
        if self.null_token:
            self.k_null = nn.Parameter(torch.zeros(18))
            self.v_null = nn.Parameter(torch.zeros(18))

        # ---- head: [attention out | code (| global counts)] -> logit ----
        head_in = 36 + (36 if self.global_path else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1)
        )
        if self.reads > 1:
            self._build_extra_reads(head_hidden)

    def _build_extra_reads(self, head_hidden):
        """E21: reads 2..R, built after the registered modules so the
        single-read init stream is untouched. Read 2 keeps the E17 names."""
        self.query_update = nn.Linear(36, 18)
        self.Wk2 = nn.Linear(36, 18, bias=False)
        self.Wv2 = nn.Linear(36, 18, bias=False)
        for r in range(3, self.reads + 1):
            setattr(self, f"query_update{r}", nn.Linear(36, 18))
            setattr(self, f"Wk{r}", nn.Linear(36, 18, bias=False))
            setattr(self, f"Wv{r}", nn.Linear(36, 18, bias=False))
        self.head = nn.Sequential(
            nn.Linear(36 + 18 * (self.reads - 1), head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1))

    def _read_modules(self, r):
        if r == 2:
            return self.query_update, self.Wk2, self.Wv2
        return (getattr(self, f"query_update{r}"), getattr(self, f"Wk{r}"),
                getattr(self, f"Wv{r}"))

    def _multi_read(self, tok, q_flat):
        prev = self._attend(tok, q_flat)
        outs = [prev]
        q = q_flat
        for r in range(2, self.reads + 1):
            upd, wk, wv = self._read_modules(r)
            q = q + upd(torch.cat([q, prev], dim=-1))
            scores = torch.einsum("bqd,bkd->bqk", q, wk(tok)) * self.attn_scale
            prev = torch.einsum("bqk,bkd->bqd", scores.softmax(-1), wv(tok))
            outs.append(prev)
        logits = self.head(torch.cat(outs + [q_flat], dim=-1)).squeeze(-1)
        return self._with_evidence_residual(tok, q_flat, logits)

    # ---- forward pieces ----
    def _tokens(self, prefix_pm1):
        """prefix_pm1: (B,1,8,H,W) ±1 -> tokens (B, 7*H*W, 36).

        Token index = t*(H*W) + y*W + x for the transition pair (t, t+1), cell
        (y,x); attention is permutation-invariant so the ordering only needs
        internal consistency."""
        B = prefix_pm1.shape[0]
        p = prefix_pm1[:, 0]                                   # (B,8,H,W)
        pairs = torch.stack([p[:, :-1], p[:, 1:]], dim=2)      # (B,7,2,H,W)
        pairs = pairs.reshape(B * 7, 1, 2, self.grid, self.grid)
        h = self.feat_a(_circ3d(pairs, 1))                     # (B*7,3,1,H,W)
        h = F.relu(self.split_a(h))
        tok = F.relu(self.conjoin_a(h))                        # (B*7,36,1,H,W)
        tok = tok.reshape(B, 7, 36, self.grid * self.grid)
        return tok.permute(0, 1, 3, 2).reshape(B, 7 * self.grid * self.grid, 36)

    def _codes(self, frames_pm1):
        """frames_pm1: (N,1,H,W) ±1 -> codes (N,18,H,W)."""
        h = self.feat_b(_circ2d(frames_pm1, 1))
        h = F.relu(self.split_b(h))
        return F.relu(self.conjoin_b(h))

    def _attend(self, tok, q_flat):
        """tok: (B,448,36); q_flat: (B, H*W, 18) -> per-cell attention out
        (B, H*W, 18)."""
        Wk = self.Wk if isinstance(self.Wk, torch.Tensor) else self.Wk.weight.t()
        Wv = self.Wv if isinstance(self.Wv, torch.Tensor) else self.Wv.weight.t()
        k = tok @ Wk                                           # (B,448,18)
        v = tok @ Wv                                           # (B,448,18)
        if self.null_token:
            B = tok.shape[0]
            k = torch.cat([k, self.k_null.expand(B, 1, 18)], dim=1)   # (B,449,18)
            v = torch.cat([v, self.v_null.expand(B, 1, 18)], dim=1)
        scores = torch.einsum("bqd,bkd->bqk", q_flat, k) * self.attn_scale
        w = scores.softmax(dim=-1)
        return torch.einsum("bqk,bkd->bqd", w, v)

    def _logits_from(self, tok, q_flat):
        if self.reads > 1:
            return self._multi_read(tok, q_flat)
        out = self._attend(tok, q_flat)
        parts = [out, q_flat]
        if self.global_path:
            g = tok.sum(dim=1, keepdim=True)                   # (B,1,36) counts
            if self.global_norm == "log1p":                    # bounded, keeps 0-vs-1
                g = torch.log1p(g)
            elif self.global_norm == "mean":
                g = g / tok.shape[1]
            parts.append(g.expand(-1, q_flat.shape[1], -1))
        logits = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        return self._with_evidence_residual(tok,q_flat,logits)

    def _with_evidence_residual(self,tok,q_flat,logits):
        if not hasattr(self,"evidence_adapter"):
            return logits
        counts=torch.log1p(tok.sum(dim=1,keepdim=True)).expand(-1,q_flat.shape[1],-1)
        features=torch.cat([q_flat,counts,logits.unsqueeze(-1)],dim=-1)
        return logits+self.evidence_adapter(features).squeeze(-1)

    def forward(self, frames_pm1):
        """Teacher-forced parallel prediction.

        frames_pm1: (B,1,16,H,W) ±1. Tokens from the prefix (frames 0..7,
        one A forward); queries = codes of frames 7..14 (teacher forcing);
        returns logits (B,8,H,W) for frames 8..15.
        """
        B = frames_pm1.shape[0]
        tok = self._tokens(frames_pm1[:, :, :8])               # (B,448,36)
        cur = frames_pm1[:, 0, 7:15].reshape(B * 8, 1, self.grid, self.grid)
        q = self._codes(cur)                                   # (B*8,18,H,W)
        q = q.reshape(B, 8, 18, self.grid * self.grid)
        q = q.permute(0, 1, 3, 2).reshape(B * 8, self.grid * self.grid, 18)
        logits = self._logits_from(tok.repeat_interleave(8, dim=0), q)
        return logits.reshape(B, 8, self.grid, self.grid)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """Autoregressive 8-step rollout; tokens frozen at the prefix.
        prefix_pm1: (B,1,8,H,W) ±1 -> predictions (B,8,H,W) uint8."""
        B = prefix_pm1.shape[0]
        tok = self._tokens(prefix_pm1)
        x = prefix_pm1[:, :, 7]                                # (B,1,H,W)
        preds = []
        for _ in range(8):
            q = self._codes(x)                                 # (B,18,H,W)
            q = q.reshape(B, 18, self.grid * self.grid).permute(0, 2, 1)
            logits = self._logits_from(tok, q).reshape(B, self.grid, self.grid)
            pred = (logits > 0).to(torch.uint8)
            preds.append(pred)
            x = (2.0 * pred.float() - 1.0).unsqueeze(1)
        return torch.stack(preds, dim=1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
