import pytest
import torch
from cawm.models.art_vanilla import ArtVanilla, LAYERS
from cawm.models.art_cached import rollout_cached, rollout_sliding_cached

@pytest.mark.parametrize("depth", [1,2,3])
def test_sliding_independent_reference_and_inactive_positions(depth, monkeypatch):
    monkeypatch.setitem(LAYERS, 8, 1)
    torch.manual_seed(142+depth)
    m=ArtVanilla(dmodel=8,nlayers=depth,nhead=4).eval()
    x=torch.randint(0,2,(2,1,8,8,8)).float()*2-1
    with torch.no_grad():
        y=rollout_sliding_cached(m,x,3)
        assert torch.equal(y[:,:1],rollout_cached(m,x,1))
        # Independent uncached reference: raster causal calls from scratch.
        context=x
        out=[]
        for _ in range(3):
            t=((context[:,0]+1)/2).long().reshape(2,-1)
            for _ in range(64):
                z=m.head(m._encode(t)[:,-1:]).squeeze(-1)
                t=torch.cat((t,(z>0).long()),1)
            frame=t[:,-64:].reshape(2,1,8,8).to(torch.uint8)
            out.append(frame)
            context=torch.cat((context[:,:,1:],2*frame.float().unsqueeze(1)-1),2)
        assert torch.equal(y,torch.cat(out,1))
        full_before=rollout_cached(m,x,3)
        m.pos[:,575:].normal_(mean=0,std=30)
        assert torch.equal(y,rollout_sliding_cached(m,x,3))
        # Full-context negative control does read these untrained positions.
        full_after=rollout_cached(m,x,3)
        assert torch.equal(y[:,:1],full_after[:,:1])
        assert not torch.equal(full_before[:,1:],full_after[:,1:])

def test_validation(monkeypatch):
    monkeypatch.setitem(LAYERS,8,1)
    m=ArtVanilla(dmodel=8,nlayers=1,nhead=4).eval()
    x=torch.ones(1,1,8,8,8)
    for h in (0,-1,1.5,True):
        with pytest.raises(ValueError):rollout_sliding_cached(m,x,h)
    with pytest.raises(ValueError):rollout_sliding_cached(m,torch.zeros_like(x),1)
    with pytest.raises(ValueError):rollout_sliding_cached(m,x[:,:,:7],1)
