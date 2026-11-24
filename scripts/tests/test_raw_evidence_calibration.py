import unittest
import torch
from cawm.models.raw_evidence import RawEvidenceCA
from cawm.train import build_model

class CalibrationTest(unittest.TestCase):
    def test_support_limit_and_saturation(self):
        m=RawEvidenceCA(mode="support")
        c=torch.zeros(4,512);z=c.clone()
        c[:,0]=torch.tensor([0.,.01,1.,100.]);z.copy_(c)
        a=m.table(c,z)[:,0]
        self.assertAlmostEqual(float(a[0].detach()),0.,places=6)
        self.assertTrue(bool((a[1:]>a[:-1]).all()))
        torch.testing.assert_close(m.table(c,-z)[:,0],-a,atol=1e-5,rtol=1e-5)
        a.sum().backward()
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters()))
        self.assertEqual(m.trainable_param_count(),300)
    def test_factory_and_prefix(self):
        m=build_model(42,model="raw_evidence_support",grid=8,arm="emergence",head="concat")
        self.assertEqual(m.mode,"support")
        x=torch.randint(0,2,(3,1,16,8,8)).float()*2-1
        p=x[:,:,:8];y=x.clone();y[:,:,8:]*=-1
        torch.testing.assert_close(m.first(p),m(y)[:,0])
        r=m.rollout(p)
        full=torch.cat([p,(r.float()*2-1)[:,None]],2)
        self.assertTrue(torch.equal(m(full)>0,r.bool()))
        z=torch.zeros(1,8,8,dtype=torch.long);z[0,7,7]=1
        self.assertEqual(int(m.query(z)[0,0,0]),256)

if __name__=="__main__":unittest.main()
