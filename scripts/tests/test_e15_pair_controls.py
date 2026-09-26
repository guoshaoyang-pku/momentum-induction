"""E15: matched misalignment, causal-prefix and checkpoint contract."""
import torch
from cawm.models.art_vanilla import ArtVanilla
from cawm.train import build_model, model_kwargs

def make(mode, pos='learned'):
    torch.manual_seed(42)
    return ArtVanilla(tokenizer=mode, pos_enc=pos).eval()

def test_shift_preserves_parameters_and_neighbourhood_multiset():
    paired, shifted = make('pair'), make('pair_shift')
    assert paired.trainable_param_count() == shifted.trainable_param_count()
    for k, v in paired.state_dict().items():
        assert torch.equal(v, shifted.state_dict()[k]), k
    toks = torch.randint(0, 2, (2, 1024))
    a = paired._pair_features(toks).reshape(2,16,8,8,10)
    b = shifted._pair_features(toks).reshape(2,16,8,8,10)
    assert torch.equal(a[...,9], b[...,9])
    assert torch.equal(torch.roll(a[...,:9], (4,4), (2,3)), b[...,:9])
    assert not torch.equal(a[:,1:,...,:9], b[:,1:,...,:9])

def test_shift_is_causal_and_prefix_consistent():
    m = make('pair_shift')
    toks = torch.randint(0, 2, (1,1024))
    altered = toks.clone(); altered[:,700:] = 1-altered[:,700:]
    with torch.no_grad():
        a = m._encode(toks)
        b = m._encode(altered)
        prefix = m._encode(toks[:,:700])
    assert torch.allclose(a[:,:700], b[:,:700], atol=1e-6, rtol=0)
    assert torch.allclose(a[:,:700], prefix, atol=1e-5, rtol=0)

def test_controls_reload_and_nope_removes_only_table():
    for mode in ('pair','pair_shift'):
        p,n = make(mode),make(mode,'none')
        assert p.trainable_param_count()-n.trainable_param_count()==1024*64
        kw=model_kwargs(dict(grid=8,arm='emergence',head='concat',tokenizer=mode,pos_enc='none',dmodel=64))
        r=build_model(42,model='art_vanilla',**kw)
        r.load_state_dict(n.state_dict())
        assert r.tokenizer==mode and r.pos_enc=='none'
