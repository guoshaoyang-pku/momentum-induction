import numpy as np
import torch
import torch.nn.functional as F
from cawm.evidence_loss import evidence_groups, evidence_bce
from cawm.data import ChunkedStreamDataset
from cawm.simulate import neighbor_count
from cawm.train import build_model


def test_relation_equal_not_cell_weighted():
    z=torch.tensor([0.,0.,0.,2.],requires_grad=True)
    y=torch.zeros(4);g=torch.tensor([0,0,0,1])
    loss=evidence_bce(z,y,g)
    expected=(F.softplus(z[:3]).mean()+F.softplus(z[3:]).mean())/2
    torch.testing.assert_close(loss,expected)
    loss.backward();torch.testing.assert_close(z.grad[:3],torch.full((3,),1/12))


def test_absent_groups_and_slots():
    z=torch.tensor([0.,1.,2.]);y=torch.zeros(3)
    for mode,g in [('relation',torch.zeros(3,dtype=torch.long)),('slot_relation',torch.tensor([2,2,4]))]:
        expected=F.softplus(z).mean() if mode=='relation' else (F.softplus(z[:2]).mean()+F.softplus(z[2:]).mean())/2
        torch.testing.assert_close(evidence_bce(z,y,g,mode),expected)


def test_group_prefix_only_and_numpy_agreement():
    ds=ChunkedStreamDataset(1948,half='train',grid=8,n_chunks=1,chunk_size=64,sc_filter=True,l4v2='l4b')
    b=ds[0];f=b['frames'];cov=b['cov']
    src=f.numpy()[:,7:15];sl=(9*src+neighbor_count(src.reshape(-1,8,8)).reshape(src.shape)).astype(int)
    direct=np.take_along_axis(cov.numpy(),sl.reshape(len(f),-1),1).reshape(sl.shape)
    g=evidence_groups(f,cov,'relation')
    assert np.array_equal(g.numpy(),~direct)
    q=f.clone();q[:,15]=1-q[:,15] # final outcomes cannot affect any source label
    assert torch.equal(g,evidence_groups(q,cov))
    s=evidence_groups(f,cov,'slot_relation')
    assert np.array_equal(s.numpy(),18*(~direct)+sl)


def test_frontier_default_bitcompat_and_weight_alignment():
    model=build_model(42,model='diffusion',grid=8,arm='emergence',head='concat',frontier=True,t_per_frame=True)
    x=torch.randint(0,2,(5,1,16,8,8)).float()*2-1
    torch.manual_seed(81);a=model.frontier_loss(x)
    torch.manual_seed(81);b=model.frontier_loss(x,evidence_groups=None,evidence_mode='none')
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    # All cells same group -> exact ordinary frontier BCE (up to reduction rounding).
    torch.manual_seed(81);c=model.frontier_loss(x,evidence_groups=torch.zeros(5,8,8,8,dtype=torch.long),evidence_mode='relation')
    torch.testing.assert_close(a[0],c[0])
    # Observe selected IDs to verify each sample uses its actual sampled k.
    import cawm.evidence_loss as el
    orig=el.evidence_bce;seen=[]
    def record(logits,targets,groups,mode):
        seen.append(groups.clone());return orig(logits,targets,groups,mode)
    groups=torch.arange(8)[None,:,None,None].expand(5,8,8,8).clone()
    torch.manual_seed(81);expected_k=torch.randint(0,8,(5,))
    el.evidence_bce=record
    try:
        torch.manual_seed(81);model.frontier_loss(x,evidence_groups=groups,evidence_mode='slot_relation')
    finally:el.evidence_bce=orig
    assert torch.equal(seen[0][:,0,0],expected_k)


def test_slot_loss_and_training_source_labels():
    z=torch.tensor([0.,0.,0.,2.]);y=torch.zeros(4);g=torch.tensor([5,5,5,7])
    expected=(F.softplus(z[:3]).mean()+F.softplus(z[3:]).mean())/2
    torch.testing.assert_close(evidence_bce(z,y,g,'slot'),expected)
    f=torch.zeros(2,16,8,8);cov=torch.zeros(2,18,dtype=torch.bool)
    assert torch.equal(evidence_groups(f,cov,'slot'),torch.zeros(2,8,8,8,dtype=torch.long))
    f[:,7:15]=1
    assert torch.equal(evidence_groups(f,cov,'slot'),torch.full((2,8,8,8),17,dtype=torch.long))
