import unittest
import numpy as np
from hybrid_tsad.normal_evolution_evaluation import aligned_values, threshold

class FixedEvaluationTest(unittest.TestCase):
    def test_cal_requires_original_axis(self):
        ix=np.arange(446173,495000)
        cal=np.arange(len(ix),dtype=float)
        tau,selected=threshold(cal,ix,'SWAT')
        self.assertEqual(tau,np.quantile(cal[selected-ix[0]],.99,method='higher'))
        with self.assertRaises(ValueError): threshold(cal[1:],ix[1:],'SWAT')

    def test_baseline_has_no_gap_interpolation(self):
        bank={'index':np.array([1,2,4]),'score':np.array([.1,.2,.4])}
        np.testing.assert_array_equal(aligned_values(bank,np.array([1,4])),[.1,.4])
        with self.assertRaises(ValueError): aligned_values(bank,np.array([1,3]))

if __name__=='__main__': unittest.main()
