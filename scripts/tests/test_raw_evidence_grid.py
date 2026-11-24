import unittest
import torch
from cawm.models.raw_evidence import RawEvidenceCA

class GridTest(unittest.TestCase):
    def test_grid_and_wrap(self):
        torch.manual_seed(42)
        m=RawEvidenceCA(mode="mean")
        for g in (4,8,16):
            corner=torch.zeros(1,g,g,dtype=torch.long);corner[0,-1,-1]=1
            self.assertEqual(int(m.query(corner)[0,0,0]),256)
            self.assertNotEqual(int(RawEvidenceCA(shifted=True).query(corner)[0,0,0]),256)
            p=torch.randint(0,2,(2,1,8,g,g)).float()*2-1
            roll=m.rollout(p,steps=24)
            self.assertEqual(roll.shape,(2,24,g,g))
            self.assertTrue(torch.equal(roll[:,:8],m.rollout(p)))
            self.assertTrue(torch.equal(m.rollout(torch.roll(p,(1,-2),(-2,-1)),24),torch.roll(roll,(1,-2),(-2,-1))))
            torch.testing.assert_close(m.first(p),m(torch.cat([p,torch.ones_like(p)],2))[:,0])
            torch.testing.assert_close(m.first(p),m(torch.cat([p,-torch.ones_like(p)],2))[:,0])
        with self.assertRaises(ValueError):m.rollout(p,0)
        with self.assertRaises(ValueError):m.rollout(p[:,:,:7])
if __name__=="__main__":unittest.main()
