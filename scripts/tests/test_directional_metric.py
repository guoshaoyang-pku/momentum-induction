import numpy as np
import torch
from cawm.models.directional_metric import LocalMetric,patterns,evidence,rollout
from run_directional_metric import dataset,qa,slots,prep,oracle

def test_torus_and_no_future():
    qa()

def test_kernel_against_direct_prefix_reference():
    x,_=dataset(55,3)
    c,z=evidence(x[:,:8])
    model=LocalMetric(2)
    table=model.table(torch.from_numpy(c),torch.from_numpy(z)).detach().numpy()
    raw=((np.arange(512)[:,None]>>np.arange(9))&1)*2-1
    emb=raw@model.encoder.weight.detach().numpy().T
    pats=patterns(x[:,:7]).reshape(len(x),-1)
    y=x[:,1:8].reshape(len(x),-1).astype(float)*2-1
    for i in range(len(x)):
        k=np.exp(-((emb[:,None]-emb[pats[i]][None,:])**2).sum(-1))
        ref=torch.nn.functional.softplus(model.gain).item()*(k@y[i])/k.sum(1).clip(1e-6)+model.bias.item()
        np.testing.assert_allclose(table[i],ref,atol=2e-6)

def test_oracle_disambiguates_axial_and_diagonal():
    s=slots()
    assert s[1<<1]!=s[1<<0]
    assert s[1<<1]==s[1<<7]
    assert s[1<<4]!=s[1<<1]
    x,_=dataset(77,64)
    loss,acc,ro=oracle(x)
    assert 0<=loss<=np.log(2) and 0<=acc<=1 and 0<=ro['seq8']<=1

def test_prefix_counts():
    x,_=dataset(19,4)
    c,z=evidence(x[:,:8])
    assert np.all(c.sum(1)==7*64)
    assert np.all(np.abs(z)<=c)
    assert np.array_equal((c+z).sum(1)/2,x[:,1:8].sum((1,2,3)))

def test_frozen_geometry_updates_only_calibration():
    w=np.arange(18,dtype=np.float32).reshape(2,9)/10
    m=LocalMetric(2,w)
    x,_=dataset(8,4)
    c,z,q,y=prep(x)
    t=m.table(c,z)
    l=torch.nn.functional.binary_cross_entropy_with_logits(t.gather(1,q.flatten(1)).reshape(q.shape),y)
    l.backward()
    assert m.encoder.weight.grad is None and m.gain.grad is not None and m.bias.grad is not None
    np.testing.assert_array_equal(m.encoder.weight.detach().numpy(),w)
