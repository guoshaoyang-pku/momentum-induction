import itertools
import torch
from cawm.models.lifegpt import LifeGPT
from eval_lifegpt import (sample_token, lifegpt_cached_generate,
                         binary_trajectory_log_probability)


def test_temperature_cached_sampling_matches_full_causal_path():
    torch.manual_seed(19)
    m = LifeGPT(grid=4, dmodel=32, nlayers=2, nhead=1, head_dim=32).eval()
    prefix = torch.randint(0, 2, (3, 17))
    assert torch.equal(m.generate(prefix, 11), lifegpt_cached_generate(m, prefix, 11))
    gen1 = torch.Generator().manual_seed(123)
    sampled = lifegpt_cached_generate(m, prefix, 11, 1.0, gen1)
    gen2 = torch.Generator().manual_seed(123)
    slow = prefix.clone()
    with torch.no_grad():
        for _ in range(11):
            token = sample_token(m.token_logits(slow)[:, -1], 1.0, gen2)
            slow = torch.cat((slow, token), 1)
    assert torch.equal(sampled, slow[:, -11:])
    assert torch.equal(sampled, lifegpt_cached_generate(
        m, prefix, 11, 1.0, torch.Generator().manual_seed(123)))


def test_temperature_full_softmax_categorical_law():
    logits = torch.tensor([0.0, 1.0986122886681098]).expand(50000, 2)
    sampled = sample_token(logits, 1.0, torch.Generator().manual_seed(42))
    assert abs(sampled.float().mean().item() - .75) < .01
    assert sample_token(logits[:10], 0.0).eq(1).all()


def test_expected_exact_probability_enumerated_autoregression():
    torch.manual_seed(72)
    m = LifeGPT(grid=4, dmodel=32, nlayers=1, nhead=1, head_dim=32).eval()
    prefix = torch.tensor([[0, 1, 1]])
    total = 0.0
    with torch.no_grad():
        for path in itertools.product((0, 1), repeat=3):
            target = torch.tensor([path])
            full = torch.cat((prefix, target), 1)
            logits = m.token_logits(full[:, :-1])[:, prefix.shape[1]-1:]
            logp = binary_trajectory_log_probability(logits[..., 1]-logits[..., 0], target)
            slow_p = 1.0
            for i, token in enumerate(path):
                p = m.token_logits(full[:, :prefix.shape[1]+i])[:, -1].double().softmax(-1)
                slow_p *= p[0, token].item()
            assert abs(logp.exp().item() - slow_p) < 1e-7
            total += logp.exp().item()
    assert abs(total - 1.0) < 1e-7
