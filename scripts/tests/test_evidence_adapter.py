import torch
from cawm.train import build_model,model_kwargs
from cawm.initialization import initialize_and_freeze

def test_adapter_zero_init_exact_and_base_frozen(tmp_path):
    for kind in ('art','art_kvshift'):
        base=build_model(42,model=kind,grid=8,arm='emergence',head='concat')
        p=tmp_path/(kind+'.pt');torch.save(dict(step=40,args=dict(steps=40,seed=42,task='L23'),model_ema=base.state_dict()),p)
        new=build_model(42,model=kind,grid=8,arm='emergence',head='concat',evidence_adapter=True)
        info=initialize_and_freeze(new,p)
        x=torch.randint(0,2,(2,1,16,8,8)).float()*2-1
        with torch.no_grad():
            assert torch.equal(base(x),new(x))
        assert new.total_param_count()-base.total_param_count()==15489
        assert new.trainable_param_count()==15489
        frozen={k:v.clone() for k,v in base.state_dict().items()}
        loss=torch.nn.functional.binary_cross_entropy_with_logits(new(x),(x[:,0,8:]+1)/2)
        loss.backward();opt=torch.optim.Adam([p for p in new.parameters() if p.requires_grad]);opt.step()
        assert all(torch.equal(v,new.state_dict()[k]) for k,v in frozen.items())
        assert not torch.equal(base(x),new(x))
        assert model_kwargs(dict(grid=8,arm='emergence',head='concat',evidence_adapter=True))['evidence_adapter']


def test_adapter_does_not_see_future_or_labels():
    m=build_model(43,model='art_kvshift',grid=8,arm='emergence',head='concat',evidence_adapter=True)
    torch.nn.init.normal_(m.evidence_adapter[-1].weight)
    x=torch.randint(0,2,(2,1,16,8,8)).float()*2-1;y=x.clone();y[:,:,8:]=-y[:,:,8:]
    # First predicted frame uses only frames0..7; later teacher-forced frames differ.
    assert torch.equal(m(x)[:,0],m(y)[:,0])
