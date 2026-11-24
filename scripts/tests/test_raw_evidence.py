import unittest
import numpy as np
import torch
from cawm.models.raw_evidence import RawEvidenceCA
from cawm.simulate import simulate

class RawEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        rng=np.random.default_rng(42)
        self.x=torch.from_numpy(simulate(rng.integers(0,1<<18,8),rng.integers(0,2,(8,8,8),dtype=np.uint8),15))
    def test_literal_wrap(self):
        x=torch.zeros(1,8,8,dtype=torch.long);x[0,7,7]=1
        a=RawEvidenceCA();b=RawEvidenceCA(shifted=True)
        self.assertEqual(int(a.query(x)[0,0,0]),256)
        self.assertEqual(int(b.query(x)[0,0,0]),0)
        for i in range(8):
            for j in range(8):
                v=sum(int(x[0,(i-dy)%8,(j-dx)%8])*(1<<k) for k,(dy,dx) in enumerate((a,b) for a in (-1,0,1) for b in (-1,0,1)))
                self.assertEqual(int(a.query(x)[0,i,j]),v)
    def test_causality_rollout_and_grad(self):
        for mode in ("mean","count","sqrt"):
            m=RawEvidenceCA(mode=mode)
            x=self.x.float()*2-1
            full=x[:,None];p=full[:,:,:8];y=full.clone();y[:,:,8:]*=-1
            logits=m(full)
            torch.testing.assert_close(m.first(p),logits[:,0])
            torch.testing.assert_close(m.first(p),m(y)[:,0])
            for j in range(8):
                y=full.clone();y[:,:,8+j:]*=-1
                torch.testing.assert_close(logits[:,:j+1],m(y)[:,:j+1])
            pred=m.rollout(p)
            autonomous=torch.cat([p,(pred.float()*2-1)[:,None]],2)
            self.assertTrue(torch.equal(m(autonomous)>0,pred.bool()))
            torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],self.x[:,8].float()).backward()
            self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters()))
    def test_factory(self):
        from cawm.train import build_model
        for name,count in (("raw_evidence_mean",298),("raw_evidence_count",299),("raw_evidence_sqrt",299),("raw_evidence_shift",299)):
            m=build_model(42,model=name,grid=8,arm="emergence",head="concat")
            self.assertEqual(m.trainable_param_count(),count)
            self.assertEqual(m.shifted,name.endswith("shift"))
    def test_sqrt_evidence_confidence(self):
        m=RawEvidenceCA(mode="sqrt")
        c=torch.zeros(3,512);z=c.clone()
        c[:,0]=torch.tensor([1.,4.,16.]);z[:,0]=c[:,0]
        v=m.table(c,z)[:,0]
        self.assertTrue(bool((v[1:]>v[:-1]).all()))
        self.assertTrue(bool((m.table(c,-z)[:,0]<0).all()))
        self.assertEqual(m.trainable_param_count(),299)
    def test_count_evidence_confidence(self):
        m=RawEvidenceCA()
        c=torch.zeros(3,512);z=c.clone()
        c[:,0]=torch.tensor([1.,2.,10.]);z[:,0]=c[:,0]
        v=m.table(c,z)[:,0]
        self.assertTrue(bool((v[1:]>v[:-1]).all()))
        self.assertTrue(bool((m.table(c,-z)[:,0]<0).all()))
        self.assertEqual(m.trainable_param_count(),299)
if __name__=="__main__":unittest.main()
