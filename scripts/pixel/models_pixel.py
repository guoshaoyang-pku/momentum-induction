"""Pixel-space models: a learnable perception front-end, the mainline
inference core on the latent cell grid, and a pixel back-end.

PixelCNN: per-frame conv encoder -> evidence counting over frame pairs ->
per-cell situation read -> bilinear gate head -> project-up decoder.
Everything except the evidence-count pathway stays translation
equivariant, so random grid offsets are handled for free.

PixelART: patchify (one patch ~ one cell) -> 3x3 situation conv on the
patch grid -> paired (situation, outcome) tokens -> sharp cosine
attention retrieval -> per-patch render head. With s=16 and patch=16 the
token layout reproduces the mainline ART exactly (64 cells x 7 pairs).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    """Shared per-frame encoder: `levels` stride-2 stages. Returns the final
    map plus the feature pyramid (for the decoder skips of the current frame)."""

    def __init__(self, width=32, levels=3):
        super().__init__()
        chs = [1] + [width, width * 2, width * 4, width * 4][:levels]
        self.convs = nn.ModuleList(
            nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1)
            for i in range(levels))
        self.levels = levels

    def forward(self, x):
        skips = []
        h = x
        for c in self.convs:
            h = F.relu(c(h))
            skips.append(h)
        return h, skips


class PixelCNN(nn.Module):
    def __init__(self, width=32, levels=3):
        super().__init__()
        self.enc = FrameEncoder(width, levels)
        C = [width, width * 2, width * 4, width * 4][levels - 1]
        self.C = C
        self.levels = levels
        # evidence path (network A)
        self.pair = nn.Conv2d(2 * C, C, 3, padding=1)
        self.gate = nn.Conv2d(C, 1, 1)    # transition gate: 0 when frames repeat
        self.valid = nn.Conv2d(C, 1, 1)   # arena mask: 0 on background
        # situation path (network B)
        self.sit = nn.Conv2d(C, C, 3, padding=1)
        # bilinear head
        self.cond_proj = nn.Linear(C, C)
        self.mix = nn.Conv2d(C, C, 1)
        self.lat_out = nn.Conv2d(C, 1, 1)  # aux head: latent-res cell logits
        # project-up decoder with encoder skips
        skip_chs = [width, width * 2, width * 4, width * 4][:levels]
        dec = []
        in_ch = C
        for j in range(levels - 2, -2, -1):  # j = skip index after each upsample
            if j >= 0:
                dec.append(nn.Conv2d(in_ch + skip_chs[j], skip_chs[j], 3, padding=1))
                in_ch = skip_chs[j]
            else:
                dec.append(nn.Conv2d(in_ch, in_ch, 3, padding=1))
        self.dec = nn.ModuleList(dec)
        self.out = nn.Conv2d(in_ch, 1, 1)

    def forward(self, hist):
        B, T, _, Fsz, _ = hist.shape
        frames = hist.reshape(B * T, 1, Fsz, Fsz)
        fin, sk = self.enc(frames)
        C = self.C
        fin = fin.view(B, T, C, *fin.shape[2:])
        cur_skips = [s_.view(B, T, *s_.shape[1:])[:, -1] for s_ in sk]
        cond = 0
        for t in range(T - 1):
            a, b = fin[:, t], fin[:, t + 1]
            ev = F.relu(self.pair(torch.cat([a, b], 1)))
            g = torch.sigmoid(self.gate(b - a))
            v = torch.sigmoid(self.valid(a))
            cond = cond + (ev * g * v).sum(dim=(2, 3))
        cond = cond / max(T - 1, 1)
        sit = F.relu(self.sit(fin[:, -1]))
        h = F.relu(sit * self.cond_proj(cond)[:, :, None, None])
        h = F.relu(self.mix(h))
        lat = self.lat_out(h)
        x = h
        for j, conv in zip(range(self.levels - 2, -2, -1), self.dec):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if j >= 0:
                x = torch.cat([x, cur_skips[j]], 1)
            x = F.relu(conv(x))
        logits = self.out(x)
        return logits, lat


class PixelART(nn.Module):
    def __init__(self, patch=16, d=64, scale=20.0, key_bottleneck=False):
        super().__init__()
        self.embed = nn.Conv2d(1, d, patch, patch)
        self.sit = nn.Conv2d(d, d, 3, padding=1)
        self.key_bottleneck = key_bottleneck
        if key_bottleneck:
            # discrete analog of the mainline ART: keys/queries live on the
            # 18-slot simplex; retrieval is a slot-identity match
            self.slot_head = nn.Linear(d, 18)
            self.k_null_kb = nn.Parameter(torch.zeros(18))
        else:
            self.wk = nn.Linear(d, d)
            self.wq = nn.Linear(d, d)
            self.k_null = nn.Parameter(torch.randn(d) * 0.02)
        self.wv = nn.Linear(d, d)
        self.v_null = nn.Parameter(torch.zeros(d))
        self.scale = scale
        self.patch = patch
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(),
                                  nn.Linear(d, patch * patch))

    def forward(self, hist, return_aux=False):
        B, T, _, Fsz, _ = hist.shape
        P = self.patch
        Ph = Fsz // P
        frames = hist.reshape(B * T, 1, Fsz, Fsz)
        e = self.embed(frames).view(B, T, -1, Ph, Ph)
        sit = F.relu(self.sit(e.reshape(B * T, -1, Ph, Ph))).view(B, T, -1, Ph, Ph)
        d = e.shape[2]
        vs = self.wv(e[:, 1:].permute(0, 1, 3, 4, 2))
        V = vs.reshape(B, -1, d)
        V = torch.cat([V, self.v_null.view(1, 1, -1).expand(B, 1, d)], 1)
        if self.key_bottleneck:
            kk = F.softmax(self.slot_head(
                sit[:, :-1].permute(0, 1, 3, 4, 2)), dim=-1)   # (B,T-1,Ph,Ph,18)
            qq = F.softmax(self.slot_head(
                sit[:, -1].permute(0, 2, 3, 1)), dim=-1)       # (B,Ph,Ph,18)
            K = kk.reshape(B, -1, 18)
            K = torch.cat([K, self.k_null_kb.view(1, 1, -1).expand(B, 1, 18)], 1)
            scores = self.scale * (qq.reshape(B, -1, 18) @ K.transpose(1, 2))
            q = sit[:, -1].permute(0, 2, 3, 1)  # render-side features stay d-dim
        else:
            ks = self.wk(sit[:, :-1].permute(0, 1, 3, 4, 2))   # (B,T-1,Ph,Ph,d)
            K = ks.reshape(B, -1, d)
            K = torch.cat([K, self.k_null.view(1, 1, -1).expand(B, 1, d)], 1)
            q = self.wq(sit[:, -1].permute(0, 2, 3, 1))          # (B,Ph,Ph,d)
            scores = self.scale * (F.normalize(q.reshape(B, -1, d), dim=-1)
                                   @ F.normalize(K, dim=-1).transpose(1, 2))
        att = torch.softmax(scores, dim=-1)
        out = (att @ V).view(B, Ph, Ph, d)
        pl = self.head(torch.cat([q, out], -1))              # (B,Ph,Ph,P*P)
        pl = pl.view(B, Ph, Ph, P, P).permute(0, 1, 3, 2, 4)
        logits = pl.reshape(B, 1, Fsz, Fsz)
        if return_aux:
            return logits, None, {"e": e, "sit": sit, "v": vs}
        return logits, None


def build(name, s=8, frame=128, width=32, patch=16, d=64, attn_scale=20.0,
          key_bottleneck=False):
    if name == "pcnn":
        import math
        levels = int(math.log2(frame // 16)) + 1  # latent 16x16 at frame 128
        if s == 16 and frame == 128:
            levels = 4                            # latent 8x8 = exact cell grid
        return PixelCNN(width=width, levels=levels)
    if name == "part":
        return PixelART(patch=patch, d=d, scale=attn_scale,
                        key_bottleneck=key_bottleneck)
    raise ValueError(name)
