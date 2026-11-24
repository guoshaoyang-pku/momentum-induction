import unittest
import numpy as np
import torch
from cawm.simulate import simulate
from run_raw_evidence_support import coverage, diagnose
from cawm.models.raw_evidence import RawEvidenceCA

class CoverageTest(unittest.TestCase):
    def test_literal_slots(self):
        rng=np.random.default_rng(42)
        x=simulate(rng.integers(0,1<<18,4),rng.integers(0,2,(4,8,8),dtype=np.uint8),15)
        seen,raw=coverage(x)
        self.assertTrue(np.all(~raw | seen))
        def pattern(frame,i,j):
            return tuple(int(frame[(i+dy)%8,(j+dx)%8]) for dy in (-1,0,1) for dx in (-1,0,1))
        def slot(p):return p[4],sum(p)-p[4]
        for b in range(len(x)):
            hist={pattern(x[b,t],i,j) for t in range(7) for i in range(8) for j in range(8)}
            slots={slot(p) for p in hist}
            for t in range(8):
                for i,j in ((0,0),(0,7),(7,0),(7,7),(3,4)):
                    p=pattern(x[b,7+t],i,j)
                    self.assertEqual(bool(raw[b,t,i,j]),p in hist)
                    self.assertEqual(bool(seen[b,t,i,j]),slot(p) in slots)
    def test_coverage_is_not_model_input(self):
        x=np.zeros((2,16,8,8),dtype=np.uint8)
        y=x.copy();y[1,8:]=1
        m=RawEvidenceCA(mode="sqrt")
        px=(torch.from_numpy(x[:,:8]).float()*2-1)[:,None]
        py=(torch.from_numpy(y[:,:8]).float()*2-1)[:,None]
        self.assertTrue(torch.equal(m.rollout(px),m.rollout(py)))
        self.assertTrue(coverage(x)[0].all())
        self.assertFalse(coverage(y)[0][1,1:].any())
        r=diagnose(m,y)
        self.assertEqual(r["fully_covered"]["n"],1)
        self.assertEqual(r["incomplete"]["n"],1)

if __name__=="__main__":unittest.main()
