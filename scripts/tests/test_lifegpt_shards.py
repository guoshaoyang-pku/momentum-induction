"""Uneven DDP shards must equal the loss over the full effective batch."""
import torch
from cawm.models.lifegpt import LifeGPT


def test_uneven_shard_gradient_equivalence():
    torch.manual_seed(871)
    m = LifeGPT(grid=4, dmodel=32, nlayers=1, nhead=1, head_dim=32, fcm=0)
    x = torch.randint(2, (11, 1, 16, 4, 4)).float()*2-1
    target = (x[:, 0, 8:]+1)/2
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    loss(m(x), target).backward()
    expected = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True)
    world_size = 3
    for rank in range(world_size):
        start, end = 11*rank//world_size, 11*(rank+1)//world_size
        # Local backward multiplied by world_size, then averaged by DDP.
        (loss(m(x[start:end]), target[start:end]) * (end-start)/11).backward()
    for p, wanted in zip(m.parameters(), expected):
        torch.testing.assert_close(p.grad, wanted, atol=3e-7, rtol=3e-5)
