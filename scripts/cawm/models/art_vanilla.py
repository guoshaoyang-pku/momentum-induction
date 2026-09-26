"""R0-ART baseline (E4.1): a generic decoder-only transformer over rasterized
frames — no detector stack, no pairing, no condition vector, no induction
structure. The generic-transformer control for the ladder
(docs/CLAIMS_AND_INFOFLOW.md §3.1 rung R0).

Input = 16 frames rasterized to 16*H*W = 1024 tokens (8x8). Each token value
in {0,1} -> learned embedding of dim d, plus a learned position embedding
(1024 positions). Standard pre-LN decoder blocks (nhead heads, FFN 4d, causal
mask). Loss = BCE on the tokens of frames 8..15 (the 512 future tokens).
Rollout = greedy autoregressive over the 512 future tokens (temperature 0).

Sizes (CLI --dmodel): d=64 -> 2 layers, d=128 -> 4 layers, d=192 -> 6 layers;
nhead=4 throughout.

E8.4 (2026-09-19): the ladder ties depth to width, so neither axis is identified.
`nlayers`/`nhead` are optional overrides for the E8.4 width x depth / head
controls; both default to None, which selects exactly the registered mapping
above, so every existing ladder row constructs the same modules in the same
order and stays bit-identical.

E7.2 (2026-09-20): `pos_enc` is the registered position-encoding control
(learned absolute vs factorized 2D vs RoPE). The default "learned" selects the
registered `self.pos` parameter and leaves the module tree untouched, so every
existing row stays bit-identical; the other two modes build different (and, for
RoPE, differently-computed) position structure, which is the point of the arm.
E12.6 (2026-09-24) adds "none" (NoPE): no positional parameters and no rotary
blocks; the causal mask is then the only source of order information.

A6 (2026-09-22): `tokenizer` is the pairing-tokenizer arm. "cell" (default) is
the registered behaviour, bit-identical: token i embeds only its own {0,1}
value. "pair" adds `pair_proj`, a linear map of a 10-d feature per token: the
3x3 circular neighbourhood of the same cell in the PREVIOUS frame (9 values,
zeros for frame 0) plus the cell's own value — i.e. the situation-outcome pair
is built into the token. Leakage-safe by construction: frame t-1 is entirely
before frame t in raster order and the own value is token i itself, so the
embedding at position i depends only on tokens at positions <= i.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NHEAD = 4
LAYERS = {64: 2, 128: 4, 192: 6, 256: 8}
POS_ENC = ("learned", "factorized", "rope", "none",
           "rope_axial", "rope_torus", "rope_torus_mismatch", "rope_torus_detuned")
TOKENIZERS = ("cell", "pair", "pair_shift")
ROPE_BASE = 10000.0


def _rope_freqs(L, d_head, device, base=ROPE_BASE):
    """cos/sin tables of shape (L, d_head/2) for rotary position embedding."""
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float()
                         / d_head))
    t = torch.arange(L, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def _apply_rope(x, cos, sin):
    """x: (B, nh, L, dh) -> rotated. Rotates each (even, odd) pair."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


def _axial_rope_angles(coords, d_head, grid, mode):
    """E14: separate time/row/column rotary pairs; no learned parameters.

    coords: (...,3) input-token coordinates, NOT next-token target coordinates.
    At head dimension 16 allocate 2 time, 3 row and 3 column complex pairs.
    Time is always ordinary nonperiodic RoPE. Spatial integer harmonics close
    exactly on the torus; the mismatch control closes at twice the grid size.
    The raster causal mask is unchanged and is not translation-equivariant.
    """
    assert d_head % 2 == 0 and d_head >= 6
    assert mode in ("rope_axial", "rope_torus", "rope_torus_mismatch",
                    "rope_torus_detuned")
    npairs = d_head // 2
    nspace = npairs // 3
    if npairs % 3 == 2:
        nspace += 1
    ntime = npairs - 2 * nspace
    chunks = []
    for axis, count in enumerate((ntime, nspace, nspace)):
        if axis and mode != "rope_axial":
            period = grid * (2 if mode == "rope_torus_mismatch" else 1)
            harmonics = torch.arange(1, count + 1, device=coords.device,
                                     dtype=torch.float32)
            if mode == "rope_torus_detuned":
                harmonics = harmonics + 0.5
            freq = (2 * torch.pi / period) * harmonics
        else:
            freq = ROPE_BASE ** (-torch.arange(
                count, device=coords.device, dtype=torch.float32) / count)
        chunks.append(coords[..., axis:axis + 1].float() * freq)
    return torch.cat(chunks, dim=-1)


def _axial_rope_freqs(L, d_head, device, grid, mode):
    idx = torch.arange(L, device=device)
    coords = torch.stack((idx // (grid * grid),
                          (idx // grid) % grid, idx % grid), dim=-1)
    angles = _axial_rope_angles(coords, d_head, grid, mode)
    return angles.cos(), angles.sin()


class Block(nn.Module):
    def __init__(self, d, nhead, rope=False, rope_mode="rope", grid=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.rope = rope
        self.rope_mode = rope_mode
        self.grid = grid

    def _attn_rope(self, h):
        """SDPA with rotary q/k. Same projections and 1/sqrt(dh) scale as
        nn.MultiheadAttention, so the only delta is the rotation."""
        B, L, d = h.shape
        nh = self.attn.num_heads
        q, k, v = F.linear(h, self.attn.in_proj_weight,
                           self.attn.in_proj_bias).chunk(3, dim=-1)
        dh = d // nh
        q = q.view(B, L, nh, dh).transpose(1, 2)
        k = k.view(B, L, nh, dh).transpose(1, 2)
        v = v.view(B, L, nh, dh).transpose(1, 2)
        if self.rope_mode == "rope":
            cos, sin = _rope_freqs(L, dh, h.device)
        else:
            cos, sin = _axial_rope_freqs(L, dh, h.device, self.grid, self.rope_mode)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, L, d)
        return self.attn.out_proj(a)

    def forward(self, x, mask):
        h = self.ln1(x)
        L = h.shape[1]
        if self.rope:
            x = x + self._attn_rope(h)
            x = x + self.ffn(self.ln2(x))
            return x
        if L > 2048:
            # g16 memory gate (2026-09-10): identical math to nn.MultiheadAttention
            # (same in/out projections, same 1/sqrt(head_dim) scale) but computed
            # with SDPA so the LxL score matrix is never materialized; the naive
            # path needs B*4*L*L*4 bytes (b512@4096 tokens = 137GB). g8 (L=1024)
            # keeps the original path bitwise.
            B, _, d = h.shape
            nh = self.attn.num_heads
            q, k, v = F.linear(h, self.attn.in_proj_weight,
                               self.attn.in_proj_bias).chunk(3, dim=-1)
            q = q.view(B, L, nh, d // nh).transpose(1, 2)
            k = k.view(B, L, nh, d // nh).transpose(1, 2)
            v = v.view(B, L, nh, d // nh).transpose(1, 2)
            # is_causal=True (flash kernel) instead of an explicit mask: an
            # additive (L,L) mask defeats flash and materializes B*nh*L*L
            # scores (b512@4096 = 137GB, ~6s/step). Square causal mask here
            # is exactly is_causal.
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, L, d)
            a = self.attn.out_proj(a)
        else:
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class ArtVanilla(nn.Module):
    def __init__(self, grid=8, arm="emergence", head="concat", dmodel=64,
                 nlayers=None, nhead=None, pos_enc="learned", tokenizer="cell"):
        super().__init__()
        assert grid & (grid - 1) == 0 and grid >= 4
        assert dmodel in LAYERS
        assert pos_enc in POS_ENC, pos_enc
        assert tokenizer in TOKENIZERS, tokenizer
        self.grid = grid
        self.arm = arm
        self.head_type = head
        self.dmodel = dmodel
        # E8.4 overrides; None reproduces the registered ladder exactly.
        self.nlayers = LAYERS[dmodel] if nlayers is None else nlayers
        self.nhead = NHEAD if nhead is None else nhead
        self.pos_enc = pos_enc
        self.tokenizer = tokenizer
        self.ntok = 16 * grid * grid
        self.nfuture = 8 * grid * grid
        self.embed = nn.Embedding(2, dmodel)
        # A6 pairing tokenizer: `pair_proj` only exists in "pair" mode, so the
        # "cell" module tree (and RNG consumption order) is exactly registered;
        # `embed` is kept in both modes so state-dict keys differ only by the
        # added pair_proj.*.
        if tokenizer in ("pair", "pair_shift"):
            self.pair_proj = nn.Linear(10, dmodel)
        # E7.2 position-encoding control. "learned" builds exactly the registered
        # single table; the other modes build their own parameters and no `pos`,
        # so no existing checkpoint or module order is disturbed.
        if pos_enc == "learned":
            self.pos = nn.Parameter(torch.zeros(1, self.ntok, dmodel))
            nn.init.normal_(self.pos, std=0.02)
        elif pos_enc == "factorized":
            self.pos_frame = nn.Embedding(16, dmodel)
            self.pos_row = nn.Embedding(grid, dmodel)
            self.pos_col = nn.Embedding(grid, dmodel)
            for e in (self.pos_frame, self.pos_row, self.pos_col):
                nn.init.normal_(e.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(dmodel, self.nhead, rope=pos_enc.startswith("rope"),
                   rope_mode=pos_enc if pos_enc.startswith("rope") else "rope",
                   grid=grid)
             for _ in range(self.nlayers)]
        )
        self.lnf = nn.LayerNorm(dmodel)
        self.head = nn.Linear(dmodel, 1)
        # causal mask (ntok x ntok), -inf above the diagonal
        m = torch.full((self.ntok, self.ntok), float("-inf"))
        self.register_buffer("causal", torch.triu(m, diagonal=1))

    def _pos_add(self, toks):
        """Positional term added to the token embedding (E7.2). RoPE adds none."""
        if self.pos_enc == "factorized":
            L = toks.shape[1]
            idx = torch.arange(L, device=toks.device)
            f = idx // (self.grid * self.grid)
            rc = idx % (self.grid * self.grid)
            return (self.pos_frame(f) + self.pos_row(rc // self.grid)
                    + self.pos_col(rc % self.grid))
        return 0.0

    def _pair_features(self, toks):
        """A6: 10-d feature per raster position i = (frame t, row r, col c):
        the 9 values of the 3x3 circular neighbourhood of (r,c) in frame t-1
        (all zeros for t=0), then the own value at frame t. toks: (B,L) long,
        L <= ntok -> (B,L,10) float. Leakage-safe: only reads frame t-1
        (entirely before position i in raster order) and token i itself, so
        the zero-padding of positions >= L never reaches positions < L and
        _encode(toks[:, :L]) == _encode(toks)[:, :L] exactly."""
        B, L = toks.shape
        H = self.grid
        if L < self.ntok:
            pad = torch.zeros(B, self.ntok - L, dtype=toks.dtype,
                              device=toks.device)
            full = torch.cat([toks, pad], dim=1)
        else:
            full = toks
        g = full.reshape(B, 16, H, H).float()                # (B,16,H,W)
        neigh = torch.stack(
            [torch.roll(g, shifts=(-dr, -dc), dims=(2, 3))
             for dr in (-1, 0, 1) for dc in (-1, 0, 1)],
            dim=-1)                                          # (B,16,H,W,9)
        # shift by one frame along time: prepend a zero frame, drop the last
        prev = torch.cat([torch.zeros_like(neigh[:, :1]), neigh[:, :-1]],
                         dim=1)                              # (B,16,H,W,9)
        if self.tokenizer == "pair_shift":
            prev = torch.roll(prev, shifts=(H // 2, H // 2), dims=(2, 3))
        feat = torch.cat([prev, g.unsqueeze(-1)], dim=-1)    # (B,16,H,W,10)
        return feat.reshape(B, self.ntok, 10)[:, :L]

    def _encode(self, toks):
        """toks: (B,L) long in {0,1} (L <= ntok) -> hidden (B,L,d)."""
        L = toks.shape[1]
        x = self.embed(toks)
        if self.tokenizer in ("pair", "pair_shift"):
            x = self.pair_proj(self._pair_features(toks)) + x
        x = x + self._pos_add(toks)
        if self.pos_enc == "learned":
            x = x + self.pos[:, :L]
        for b in self.blocks:
            x = b(x, self.causal[:L, :L])
        return self.lnf(x)

    def forward(self, frames_pm1):
        """Teacher-forced next-token prediction.
        frames_pm1: (B,1,16,H,W) ±1 -> logits (B,8,H,W) for frames 8..15
        (the logits at positions predicting each future token)."""
        B = frames_pm1.shape[0]
        toks = ((frames_pm1[:, 0] + 1) / 2).long().reshape(B, self.ntok)
        h = self._encode(toks)                               # (B,ntok,d)
        logits = self.head(h).squeeze(-1)                    # (B,ntok)
        # position i predicts token i+1; future tokens are ntok-nfuture..ntok-1,
        # predicted by positions ntok-nfuture-1..ntok-2.
        start = self.ntok - self.nfuture
        fut = logits[:, start - 1:self.ntok - 1]             # (B,nfuture)
        return fut.reshape(B, 8, self.grid, self.grid)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        """Greedy autoregressive rollout over the 512 future tokens.
        prefix_pm1: (B,1,8,H,W) ±1 -> predictions (B,8,H,W) uint8."""
        B = prefix_pm1.shape[0]
        dev = prefix_pm1.device
        nprefix = 8 * self.grid * self.grid
        toks = ((prefix_pm1[:, 0] + 1) / 2).long().reshape(B, nprefix)
        toks = torch.cat([toks, torch.zeros(B, self.nfuture, dtype=torch.long,
                                            device=dev)], dim=1)
        for i in range(self.nfuture):
            pos = nprefix + i
            h = self._encode(toks[:, :pos])                  # (B,pos,d)
            logit = self.head(h[:, -1]).squeeze(-1)          # (B,)
            toks = toks.clone()
            toks[:, pos] = (logit > 0).long()
        fut = toks[:, nprefix:]                              # (B,nfuture)
        return fut.reshape(B, 8, self.grid, self.grid).to(torch.uint8)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
