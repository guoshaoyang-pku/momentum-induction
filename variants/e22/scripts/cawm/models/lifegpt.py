"""LifeGPT raster decoder, following lamm-mit/LifeGPT f79ce8c.

The 256/12/8/64/256 configuration loads the released epoch-50 weights and
has exactly 12,735,744 parameters. Bias-free LayerNorm is verified from that
checkpoint, as are the 32 rotary dimensions. The CAWM adaptation changes
the alphabet to two cells and supplies eight history frames; it does not
add spatial features, frame-parallel attention, pairing, or rule retrieval.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LifeNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, None, 1e-5)


def forgetful_mask(batch, length, probability, device):
    # x-transformers AutoregressiveWrapper: exact number of randomly hidden
    # KV positions, shared by layers/heads; first token always remains visible.
    scores = torch.randn(batch, length, device=device)
    scores[:, 0] = -torch.finfo(scores.dtype).max
    count = min(int((length + 1) * probability), length - 1)
    return torch.ones_like(scores, dtype=torch.bool).scatter_(
        1, scores.topk(count, dim=-1).indices, False)


class LifeBlock(nn.Module):
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.ln1, self.ln2 = LifeNorm(dim), LifeNorm(dim)
        self.qkv = nn.Linear(dim, 3 * heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))

    def forward(self, x, cos, sin, mask=None, cache=None, offset=0):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(
            b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        def rotate(t):
            r = cos.shape[-1] * 2
            a, z = t[..., :r:2].float(), t[..., 1:r:2].float()
            rotated = torch.stack((a*cos-z*sin, a*sin+z*cos), -1).flatten(-2)
            return torch.cat((rotated.to(t.dtype), t[..., r:]), -1)
        q, k = rotate(q), rotate(k)
        if cache is not None:
            cache[0][:, :, offset:offset+n] = k
            cache[1][:, :, offset:offset+n] = v
            k, v = cache[0][:, :, :offset+n], cache[1][:, :, :offset+n]
        # At cached single-token decode every key is in the past. Causal=True
        # would incorrectly use the upper-left single-row triangle in SDPA.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=mask is None and n > 1)
        x = x + self.out(y.transpose(1, 2).reshape(b, n, -1))
        return x + self.ff(self.ln2(x))


class LifeGPT(nn.Module):
    def __init__(self, grid=8, dmodel=256, nlayers=12, nhead=8,
                 head_dim=64, vocab=2, fcm=0.15, **unused):
        super().__init__()
        assert head_dim >= 32 and head_dim % 2 == 0
        self.grid, self.nlayers = grid, nlayers
        self.nhead, self.head_dim, self.fcm = nhead, head_dim, fcm
        self.embed = nn.Embedding(vocab, dmodel)
        self.blocks = nn.ModuleList([LifeBlock(dmodel, nhead, head_dim)
                                    for _ in range(nlayers)])
        self.norm = LifeNorm(dmodel)
        self.to_logits = nn.Linear(dmodel, vocab, bias=False)
        r = max(head_dim // 2, 32)
        self.register_buffer('inv_freq', 1 / (10000 ** (
            torch.arange(0, r, 2).float() / r)))
        nn.init.kaiming_normal_(self.embed.weight)

    def token_logits(self, tokens, kv_mask=None, cache=None, offset=0):
        n = tokens.shape[1]
        angles = torch.outer(torch.arange(offset, offset+n,
                             device=tokens.device).float(), self.inv_freq)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        mask = None
        if kv_mask is not None:
            causal = torch.ones(n, n, device=tokens.device, dtype=torch.bool).tril()
            mask = causal[None, None] & kv_mask[:, None, None, :]
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, mask,
                      None if cache is None else cache[i], offset)
        return self.to_logits(self.norm(x))

    def forward(self, frames_pm1):
        b = frames_pm1.shape[0]
        tokens = ((frames_pm1[:, 0] + 1) / 2).long().reshape(b, -1)
        inputs = tokens[:, :-1]
        mask = (forgetful_mask(b, inputs.shape[1], self.fcm, inputs.device)
                if self.training and self.fcm > 0 else None)
        logits = self.token_logits(inputs, mask)
        # Binary CE == BCE(logit_1-logit_0); preserves the two-class softmax.
        start = 8 * self.grid * self.grid - 1
        return (logits[:, start:, 1] - logits[:, start:, 0]).reshape(
            b, 8, self.grid, self.grid)

    @torch.no_grad()
    def generate(self, prefix, length):
        assert not self.training, 'generation requires eval mode (FCM disabled)'
        b, offset = prefix.shape
        dtype = (torch.get_autocast_dtype(prefix.device.type)
                 if torch.is_autocast_enabled(prefix.device.type)
                 else self.embed.weight.dtype)
        shape = (b, self.nhead, offset+length, self.head_dim)
        cache = [(torch.empty(shape, device=prefix.device, dtype=dtype),
                  torch.empty(shape, device=prefix.device, dtype=dtype))
                 for _ in self.blocks]
        logits = self.token_logits(prefix, cache=cache)
        outputs = []
        for i in range(length):
            token = logits[:, -1].argmax(-1, keepdim=True)
            outputs.append(token)
            if i + 1 < length:
                logits = self.token_logits(token, cache=cache, offset=offset+i)
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def rollout(self, prefix_pm1):
        b = prefix_pm1.shape[0]
        prefix = ((prefix_pm1[:, 0] + 1) / 2).long().reshape(b, -1)
        return self.generate(prefix, 8*self.grid*self.grid).reshape(
            b, 8, self.grid, self.grid).to(torch.uint8)

    def load_official(self, state):
        """Strict weight conversion; no ignored or randomly initialized tensors."""
        src = dict(state)
        dst = {'embed.weight': src.pop('net.token_emb.emb.weight'),
               'norm.weight': src.pop('net.attn_layers.final_norm.weight'),
               'to_logits.weight': src.pop('net.to_logits.weight'),
               'inv_freq': src.pop('net.attn_layers.rotary_pos_emb.inv_freq')}
        for i in range(self.nlayers):
            a, f = (f'net.attn_layers.layers.{2*i}.',
                    f'net.attn_layers.layers.{2*i+1}.')
            p = f'blocks.{i}.'
            dst[p+'ln1.weight'] = src.pop(a+'0.0.weight')
            dst[p+'ln2.weight'] = src.pop(f+'0.0.weight')
            dst[p+'qkv.weight'] = torch.cat([src.pop(a+'1.to_'+v+'.weight')
                                            for v in ('q', 'k', 'v')])
            dst[p+'out.weight'] = src.pop(a+'1.to_out.weight')
            for ours, theirs in [('0', '0.0'), ('2', '2')]:
                for field in ('weight', 'bias'):
                    dst[p+'ff.'+ours+'.'+field] = src.pop(f+'1.ff.'+theirs+'.'+field)
        assert not src, f'Unmapped official tensors: {list(src)}'
        self.load_state_dict(dst, strict=True)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())
