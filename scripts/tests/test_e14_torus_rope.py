"""E14: topology, causal-prefix safety and checkpoint compatibility."""
import pytest
import torch

from cawm.models.art_vanilla import (ArtVanilla, _apply_rope,
                                      _axial_rope_angles)
from cawm.train import build_model, model_kwargs


def test_torus_rotations_wrap_spatially_but_not_in_time():
    torch.manual_seed(13)
    coords = torch.tensor([[2, 0, 7], [3, 7, 0], [5, 3, 4]])
    x = torch.randn(1, 2, 3, 16)
    def rotate(c, mode="rope_torus"):
        a = _axial_rope_angles(c, 16, 8, mode)
        return _apply_rope(x, a.cos(), a.sin())
    base = rotate(coords)
    for axis in (1, 2):
        shifted = coords.clone()
        shifted[:, axis] += 8
        torch.testing.assert_close(rotate(shifted), base, atol=8e-6, rtol=1e-5)
        assert not torch.allclose(rotate(shifted, "rope_torus_mismatch"),
                                  rotate(coords, "rope_torus_mismatch"))
    later = coords.clone()
    later[:, 0] += 8
    assert not torch.allclose(rotate(later), base)


def test_torus_relative_dot_product_across_both_seams():
    # Same signed nearest-neighbour offset in the interior and at the seam.
    torch.manual_seed(4)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    def score(qpos, kpos):
        a = _axial_rope_angles(torch.tensor([qpos]), 16, 8, "rope_torus")
        b = _axial_rope_angles(torch.tensor([kpos]), 16, 8, "rope_torus")
        return (_apply_rope(q, a.cos(), a.sin()) *
                _apply_rope(k, b.cos(), b.sin())).sum()
    for delta in ((1, 0), (0, 1), (1, 1)):
        interior = score([3, 4, 4], [2, 4-delta[0], 4-delta[1]])
        wrapped = score([3, 0, 0], [2, (-delta[0]) % 8, (-delta[1]) % 8])
        torch.testing.assert_close(interior, wrapped, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["rope_axial", "rope_torus",
                                  "rope_torus_mismatch", "rope_torus_detuned"])
def test_e14_causality_init_and_reload(mode):
    def make(pe):
        torch.manual_seed(42)
        return ArtVanilla(grid=8, dmodel=64, pos_enc=pe).eval()
    baseline, model = make("rope"), make(mode)
    assert model.trainable_param_count() == 100289
    assert all(torch.equal(v, baseline.state_dict()[k])
               for k, v in model.state_dict().items())
    toks = torch.randint(0, 2, (1, 90))
    with torch.no_grad():
        full = model._encode(toks)
        prefix = model._encode(toks[:, :65])
        changed = toks.clone()
        changed[:, 65:] = 1 - changed[:, 65:]
        altered = model._encode(changed)
    torch.testing.assert_close(prefix, full[:, :65], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(full[:, :65], altered[:, :65], atol=2e-6, rtol=1e-5)
    assert not torch.allclose(full, baseline._encode(toks))
    args = dict(grid=8, arm="emergence", head="concat", dmodel=64, pos_enc=mode)
    rebuilt = build_model(42, model="art_vanilla", **model_kwargs(args)).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(full, rebuilt._encode(toks), atol=0, rtol=0)
