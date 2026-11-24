"""Exact causal KV-cache decoding for the registered raw learned-position R0 ART.

No new parameters, changed training path or ground truth future tokens.
Only eval-mode, raw cell tokenizer and learned positions are supported.
"""
import torch
import torch.nn.functional as F


class RawArtCache:
    def __init__(self, model, batch):
        if model.training or model.tokenizer != "cell" or model.pos_enc != "learned":
            raise ValueError("cache requires eval-mode raw learned-position ArtVanilla")
        self.model = model
        self.length = 0
        self.keys, self.values = [], []
        for b in model.blocks:
            nh = b.attn.num_heads
            self.keys.append(model.pos.new_empty(batch, nh, model.ntok, model.dmodel // nh))
            self.values.append(torch.empty_like(self.keys[-1]))

    @torch.no_grad()
    def append(self, tokens):
        """Append a prefill chunk, or one generated token; return next-token logits."""
        m = self.model
        B, L = tokens.shape
        start, end = self.length, self.length + L
        if L < 1 or end > m.ntok or (start and L != 1):
            raise ValueError("prefill once, then append one token at a time")
        x = m.embed(tokens) + m.pos[:, start:end]
        for i, block in enumerate(m.blocks):
            h = block.ln1(x)
            nh = block.attn.num_heads
            q, k, v = F.linear(h, block.attn.in_proj_weight,
                               block.attn.in_proj_bias).chunk(3, dim=-1)
            q, k, v = (z.reshape(B, L, nh, -1).transpose(1, 2) for z in (q, k, v))
            self.keys[i][:, :, start:end].copy_(k)
            self.values[i][:, :, start:end].copy_(v)
            # Prefill is square causal. Decode query is the LAST position;
            # all keys are past/current. is_causal=True on a 1xK mask is wrong.
            a = F.scaled_dot_product_attention(q, self.keys[i][:, :, :end],
                                               self.values[i][:, :, :end],
                                               is_causal=(start == 0))
            x = x + block.attn.out_proj(a.transpose(1, 2).reshape(B, L, m.dmodel))
            x = x + block.ffn(block.ln2(x))
        self.length = end
        return m.head(m.lnf(x)).squeeze(-1)


@torch.no_grad()
def rollout_cached(model, prefix_pm1, nframes=8):
    """Greedy free-running prediction, optionally only the first future frame."""
    if not 1 <= nframes <= 8:
        raise ValueError("nframes must be in [1,8]")
    B = prefix_pm1.shape[0]
    if prefix_pm1.shape[1:] != (1, 8, model.grid, model.grid):
        raise ValueError("expected exactly eight observed frames")
    toks = ((prefix_pm1[:, 0] + 1) / 2).long().reshape(B, -1)
    cache = RawArtCache(model, B)
    logits = cache.append(toks)[:, -1:]
    predictions = []
    for i in range(nframes * model.grid ** 2):
        tok = (logits > 0).long()
        predictions.append(tok)
        if i + 1 < nframes * model.grid ** 2:
            logits = cache.append(tok)
    return torch.cat(predictions, dim=1).reshape(B, nframes, model.grid, model.grid).to(torch.uint8)


@torch.no_grad()
def rollout_sliding_cached(model, prefix_pm1, nframes=8):
    """Autonomous sliding 8-frame context for FIRST-FRAME trained R0.

    Restart the cache and absolute positions at each generated frame.
    Only positions used in first-frame training (0..9*g*g-2) are read.
    This is a separate inference protocol, not native full-context R0 rollout.
    """
    if not isinstance(nframes, int) or isinstance(nframes, bool) or nframes < 1:
        raise ValueError("positive integer horizon required")
    if prefix_pm1.ndim != 5 or prefix_pm1.shape[1:] != (1, 8, model.grid, model.grid):
        raise ValueError("expected exactly eight observed frames")
    if not torch.all((prefix_pm1 == -1) | (prefix_pm1 == 1)):
        raise ValueError("expected binary +/-1 prefix")
    context = prefix_pm1
    predictions = []
    for _ in range(nframes):
        frame = rollout_cached(model, context, 1)
        predictions.append(frame)
        next_pm1 = 2 * frame.to(context.dtype).unsqueeze(1) - 1
        context = torch.cat((context[:, :, 1:], next_pm1), dim=2)
    return torch.cat(predictions, dim=1)
