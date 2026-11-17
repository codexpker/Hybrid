"""Synthetic normalization/axis fixtures; no real labels or model weights."""
import unittest
import numpy as np
from hybrid_tsad.timercd_fixed import normalize_once,window_layout,padded_windows


class TimeRCDDataTests(unittest.TestCase):
    def test_official_float64_population_std_and_zero_variance(self):
        x=np.array([[1.,3.],[2.,3.],[4.,3.]],dtype=np.float64)
        normalized,record=normalize_once(x,'A_official_role')
        expected=(x-x.mean(0))/np.where(x.std(0)==0,1e-8,x.std(0))
        np.testing.assert_array_equal(normalized,expected)
        self.assertEqual(normalized.dtype,np.float64);self.assertEqual(record['std_ddof0_safe'][1],1e-8)

    def test_fit_once_does_not_recenter_query(self):
        x=np.array([[10.,20.],[12.,24.]])
        y,record=normalize_once(x,'B_fit_only',np.array([0.,0.]),np.array([2.,4.]))
        np.testing.assert_array_equal(y,[[5.,5.],[6.,6.]])
        self.assertNotEqual(float(y.mean()),0.);self.assertEqual(record['normalization_count'],1)

    def test_tail_real_indices_padding_and_role_boundaries(self):
        index=np.arange(106084,111101,dtype=np.int64);layout=window_layout(index)
        np.testing.assert_array_equal(layout['valid_lengths'],[5000,17])
        self.assertEqual(layout['padding_points'],4983)
        self.assertEqual(layout['internal_patch_padding'],8)
        self.assertEqual(layout['window_end_index'][-1],111100)
        self.assertTrue(layout['window_valid_mask'][1,:17].all())
        self.assertFalse(layout['window_valid_mask'][1,17:].any())
        x=np.arange(5017*2).reshape(5017,2);padded=padded_windows(x,5000)
        np.testing.assert_array_equal(padded[5017:],np.repeat(x[-1:],4983,axis=0))

    def test_short_role_does_not_change_long_role_setting(self):
        self.assertEqual(window_layout(np.arange(33))['effective_window'],33)
        self.assertEqual(window_layout(np.arange(5017))['effective_window'],5000)
        with self.assertRaises(ValueError):window_layout(np.array([0,2]))
        with self.assertRaises(ValueError):normalize_once(np.array([[np.nan,1.]]),'A_official_role')


if __name__=='__main__':unittest.main()
