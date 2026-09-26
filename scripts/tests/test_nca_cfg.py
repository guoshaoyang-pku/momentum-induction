import torch

from cawm.models.nca_config import NCAConfigWM


def test_shapes_and_params():
    torch.manual_seed(0)
    m = NCAConfigWM(grid=8)
    x = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    m.eval()
    with torch.no_grad():
        logits = m(x)
    assert logits.shape == (2, 8, 8, 8)
    n = m.trainable_param_count()
    assert 500_000 <= n <= 1_500_000, n


def test_rollout_deterministic_and_binary():
    torch.manual_seed(0)
    m = NCAConfigWM(grid=8)
    m.eval()
    prefix = torch.randint(0, 2, (2, 1, 8, 8, 8)).float() * 2 - 1
    a = m.rollout(prefix)
    b = m.rollout(prefix)
    assert a.dtype == torch.uint8
    assert torch.equal(a, b)
    assert set(a.unique().tolist()) <= {0, 1}


def test_stochastic_updates_only_in_training():
    torch.manual_seed(0)
    m = NCAConfigWM(grid=8, fire_rate=0.5)
    x = torch.randint(0, 2, (2, 1, 16, 8, 8)).float() * 2 - 1
    m.train()
    with torch.no_grad():
        o1, o2 = m(x), m(x)
    assert not torch.equal(o1, o2)          # fire mask differs step to step
    m.eval()
    with torch.no_grad():
        e1, e2 = m(x), m(x)
    assert torch.equal(e1, e2)              # deterministic at inference
