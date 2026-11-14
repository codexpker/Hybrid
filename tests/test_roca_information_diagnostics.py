import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.roca_representation_diagnostics import branch_distance
from hybrid_tsad.roca_fixed_reference_geometry import fixed_five_nn

class InformationTests(unittest.TestCase):
    def test_raw_uniform_scaling_scales_CAL_tau_and_keeps_alarm(self):
        from hybrid_tsad.hybrid_v1.scoring import weighted_quantile
        rng=np.random.default_rng(3031);ref=rng.normal(size=(64,7));cal=rng.normal(size=(32,7));query=rng.normal(size=(13,7))
        a=fixed_five_nn(ref,cal,reference_role='FIT',normalization='raw')['score']
        b=fixed_five_nn(ref,query,reference_role='FIT',normalization='raw')['score']
        aa=fixed_five_nn(ref*7,cal*7,reference_role='FIT',normalization='raw')['score']
        bb=fixed_five_nn(ref*7,query*7,reference_role='FIT',normalization='raw')['score']
        tau=weighted_quantile(a,np.ones(32)/32,.99);newtau=weighted_quantile(aa,np.ones(32)/32,.99)
        self.assertTrue(np.allclose(aa,a*49,rtol=1e-14));self.assertAlmostEqual(newtau,tau*49,places=10)
        self.assertTrue(np.array_equal(b>tau,bb>newtau))
    def test_near_identical_and_identical_use_direct_differences(self):
        r=np.tile([1.,.2,.3],(64,1));r[:,1]+=np.arange(64)*1e-10;q=r[[0,7,18]].copy()
        actual=fixed_five_nn(r,q,reference_role='FIT')['score']
        from hybrid_tsad.roca_fixed_reference_geometry import normalized
        a,_=normalized(r);b,_=normalized(q)
        expected=np.sort(np.square(b[:,None]-a[None]).sum(2),axis=1)[:,:5].mean(1)
        self.assertTrue(np.array_equal(actual,expected));self.assertTrue((actual>0).all())
        same=fixed_five_nn(np.tile([1.,2.],(64,1)),np.array([[1.,2.]]),reference_role='FIT')
        self.assertEqual(same['score'][0],0.)
    def test_cosine_and_fixed_geometry_positive_scale_rotation_invariance(self):
        rng=np.random.default_rng(2025);r=rng.normal(size=(64,6));q=rng.normal(size=(9,6));c=rng.normal(size=6)
        self.assertTrue(np.allclose(branch_distance(q,c),branch_distance(q*17,c*.3),atol=1e-14))
        rotation=np.linalg.qr(rng.normal(size=(6,6)))[0]
        a=fixed_five_nn(r,q,reference_role='FIT')['score']
        b=fixed_five_nn(r*2@rotation,q*19@rotation,reference_role='FIT')['score']
        self.assertTrue(np.allclose(a,b,atol=2e-15))
    def test_fixed_reference_permissions_and_zero_vectors(self):
        r=np.ones((64,3));q=np.array([[0.,0.,0.],[1.,1.,1.]])
        a=fixed_five_nn(r,q,reference_role='FIT')
        raw=fixed_five_nn(r,q,reference_role='FIT',normalization='raw')
        self.assertTrue(np.array_equal(a['valid'],raw['valid']))
        self.assertEqual(a['valid'].tolist(),[False,True]);self.assertTrue(np.isnan(a['score'][0]))
        with self.assertRaises(PermissionError):fixed_five_nn(r,q,reference_role='QUERY')
        r[:61]=0;a=fixed_five_nn(r,q,reference_role='FIT')
        self.assertFalse(a['valid'].any());self.assertEqual(a['reference_valid'],3)

if __name__=='__main__':unittest.main(verbosity=2)
