import pytest
import torch
from cawm.models.art_vanilla import ArtVanilla, LAYERS
from cawm.models.art_cached import RawArtCache, rollout_cached

@pytest.mark.parametrize("depth", [1, 2, 3])
def test_cache_logits_and_rollout(depth, monkeypatch):
    monkeypatch.setitem(LAYERS, 8, 1)
    torch.manual_seed(120 + depth)
    model = ArtVanilla(dmodel=8, nlayers=depth, nhead=4).eval()
    toks = torch.randint(0, 2, (2, 1024))
    with torch.no_grad():
        ref = model.head(model._encode(toks)).squeeze(-1)
        cache = RawArtCache(model, 2)
        torch.testing.assert_close(cache.append(toks[:, :512]), ref[:, :512],
                                   atol=2e-6, rtol=2e-5)
        for j in range(512, 1023):
            pred = cache.append(toks[:, j:j+1])
            torch.testing.assert_close(pred[:, 0], ref[:, j], atol=2e-6, rtol=2e-5)
        prefix = (2*toks[:, :512].reshape(2,1,8,8,8).float()-1)
        cached = rollout_cached(model, prefix)
        assert torch.equal(cached, model.rollout(prefix))
        assert torch.equal(cached[:, :1], rollout_cached(model, prefix, 1))

def test_unsupported():
    model = ArtVanilla(dmodel=64, nlayers=1, tokenizer="pair").eval()
    with pytest.raises(ValueError):
        RawArtCache(model, 1)
    model = ArtVanilla(dmodel=64, nlayers=1)
    with pytest.raises(ValueError):
        RawArtCache(model, 1)
