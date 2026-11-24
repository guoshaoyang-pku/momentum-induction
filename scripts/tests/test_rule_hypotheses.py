import unittest
import numpy as np
from cawm.models.rule_hypotheses import *
from cawm.simulate import simulate,neighbor_count

class HypothesisTest(unittest.TestCase):
    def test_wrap_classes(self):
        x=np.zeros((8,8),np.uint8);x[-1,-1]=1
        self.assertEqual(raw_pattern(x)[0,0],256)
        c=outer_totalistic_classes()
        np.testing.assert_array_equal(c[raw_pattern(x)],9*x+neighbor_count(x[None])[0])
        self.assertNotEqual(c[raw_pattern(x)][0,0],c[raw_pattern(np.roll(x,(4,4),(0,1)))][0,0])
    def test_complete_posterior(self):
        # Search tiny horizon; independently enumerate all compatible rules.
        rng=np.random.default_rng(90)
        c=outer_totalistic_classes()
        checked=0
        for _ in range(12):
            x=simulate([int(rng.integers(1<<18))],rng.integers(0,2,(1,4,4),dtype=np.uint8),7)[0]
            t=prefix_table(x,c);missing=np.flatnonzero(t<0)
            if len(missing)>12:continue
            ids=np.arange(1<<len(missing));tab=np.tile(t,(len(ids),1))
            tab[:,missing]=(ids[:,None]>>np.arange(len(missing)))&1
            rules=tab@(1<<np.arange(18))
            y=simulate(rules,np.repeat(x[-1:],len(ids),axis=0),2)[:,1:]
            _,counts=np.unique(y.reshape(len(y),-1),axis=0,return_counts=True)
            checked+=1
            got=decode(x,c,2,top_k=4,max_nodes=100000)
            self.assertAlmostEqual(got['paths'][0]['probability'],counts.max()/len(ids))
            self.assertTrue((y==got['paths'][0]['frames']).all(axis=(1,2,3)).any())
            np.testing.assert_allclose([v['probability'] for v in got['paths']],sorted(counts/len(ids),reverse=True)[:4])
        self.assertGreater(checked,0)
    def test_shared_unknown_and_budget(self):
        # Observed all-zero history fixes class 0, so complete rollout has mass 1.
        x=np.zeros((8,4,4),np.uint8);c=outer_totalistic_classes()
        r=decode(x,c,8,top_k=4)
        self.assertEqual(len(r['paths']),1);self.assertEqual(r['paths'][0]['probability'],1)
        self.assertFalse(r['exhausted'])
        rng=np.random.default_rng(6)
        for _ in range(100):
            x=simulate([int(rng.integers(1<<18))],rng.integers(0,2,(1,4,4),dtype=np.uint8),7)[0]
            p=marginal_first(x,c)
            if (p==.5).any():break
        else:self.fail('no uncertain prefix')
        r=decode(x,c,1,top_k=1,max_nodes=1)
        self.assertTrue(r['exhausted']);self.assertEqual(r['paths'],[])
        r=decode(x,c,1,top_k=16)
        self.assertTrue(all((a['frames'][0][p!=.5]==p[p!=.5]).all() for a in r['paths']))
        self.assertLessEqual(sum(a['probability'] for a in r['paths']),1)
    def test_partition_and_conflict(self):
        c=outer_totalistic_classes()
        k=(c[:,None]==c[None,:]).astype(float)
        g=learned_classes(k)
        np.testing.assert_array_equal(g[:,None]==g[None,:],k.astype(bool))
        x=np.zeros((8,4,4),np.uint8);x[1]=1
        self.assertIsNone(prefix_table(x,np.zeros(512,dtype=np.int16)))
        self.assertTrue(decode(x,np.zeros(512,dtype=np.int16))['conflict'])
    def test_future_is_not_argument(self):
        x=np.zeros((16,4,4),np.uint8);c=outer_totalistic_classes()
        with self.assertRaises(ValueError):decode(x,c)
        a=decode(x[:8],c);x[8:]=1;b=decode(x[:8],c)
        np.testing.assert_array_equal(a['paths'][0]['frames'],b['paths'][0]['frames'])
    def test_history_only_merges(self):
        # Splitting the same true slot into raw patterns cannot add conflicts.
        rng=np.random.default_rng(7)
        x=simulate(rng.integers(1<<18,size=96),rng.integers(0,2,(96,4,4),dtype=np.uint8),7)
        c=outer_totalistic_classes()
        g,merges=merge_observed_classes(c,x,min_overlap=12)
        self.assertTrue(all(prefix_table(p,g) is not None for p in x))
        self.assertEqual(len(merges),0)
        x=simulate(rng.integers(1<<18,size=128),rng.integers(0,2,(128,8,8),dtype=np.uint8),7)
        sub=c.copy()
        sub[np.flatnonzero(c==4)[::2]]=18
        g,merges=merge_observed_classes(sub,x,min_overlap=12)
        self.assertTrue(merges)
        np.testing.assert_array_equal(g[:,None]==g[None,:],c[:,None]==c[None,:])
        with self.assertRaises(ValueError):merge_observed_classes(c,np.concatenate([x,x],axis=1))
if __name__=="__main__":unittest.main()
