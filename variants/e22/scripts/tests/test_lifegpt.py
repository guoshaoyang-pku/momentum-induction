import torch
from cawm.models.lifegpt import LifeGPT, forgetful_mask
from cawm.train import build_model, model_kwargs


def test_official_parameter_count():
    assert LifeGPT(vocab=256).total_param_count() == 12735744


def test_causality_and_cache():
    torch.manual_seed(9)
    m = LifeGPT(grid=4, dmodel=32, nlayers=2, nhead=1, head_dim=32).eval()
    tokens = torch.randint(0, 2, (2, 20))
    changed = tokens.clone()
    changed[:, 11:] = 1 - changed[:, 11:]
    with torch.no_grad():
        full = m.token_logits(tokens)
        torch.testing.assert_close(full[:, :11], m.token_logits(changed)[:, :11])
        cache = [(torch.empty(2, 1, 24, 32), torch.empty(2, 1, 24, 32))
                 for _ in m.blocks]
        m.token_logits(tokens[:, :10], cache=cache)
        for i in range(10, 20):
            one = m.token_logits(tokens[:, i:i+1], cache=cache, offset=i)
            torch.testing.assert_close(one[:, 0], full[:, i], atol=2e-6, rtol=2e-5)
        slow = tokens.clone()
        for _ in range(4):
            slow = torch.cat((slow, m.token_logits(slow)[:, -1].argmax(-1)[:, None]), 1)
        assert torch.equal(m.generate(tokens, 4), slow[:, -4:])


def test_fcm_and_loss_alignment():
    torch.manual_seed(4)
    mask = forgetful_mask(4, 99, .15, 'cpu')
    assert mask[:, 0].all() and ((~mask).sum(1) == 15).all()
    m = LifeGPT(grid=4, dmodel=32, nlayers=1, nhead=1, head_dim=32).eval()
    x = torch.randint(0, 2, (2, 1, 16, 4, 4)).float()*2-1
    logits = m(x)
    altered = x.clone().reshape(2, -1)
    altered[:, 129:] *= -1
    other = m(altered.reshape_as(x)).flatten(1)
    torch.testing.assert_close(logits.flatten(1)[:, :2], other[:, :2])
    assert logits.shape == (2, 8, 4, 4)
    m.train()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), (x[:, 0, 8:]+1)/2)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_checkpoint_model_kwargs():
    args = dict(grid=8, arm='emergence', head='concat', dmodel=128,
                nlayers=4, nhead=4, life_head_dim=64, life_fcm=.15)
    m = build_model(42, model='lifegpt', **model_kwargs(args))
    assert m.nlayers == 4 and m.nhead == 4 and m.head_dim == 64
