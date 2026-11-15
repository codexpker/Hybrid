"""Incremental data-footprint and validation-state tests for the new runner."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hybrid_tsad.normality_density_runner import RoleReader, sample_batch, selection, validation_mode


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Dropout(.5)
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def loss(self, x):
        torch.rand(1); np.random.rand(); random.random()
        return x.square().mean()*self.scale


class NormalityRunner(unittest.TestCase):
    def test_h_plus_one_contains_strict_past_and_arrived_target(self):
        values = np.arange(24,dtype=np.float32).reshape(2,12,1)
        x=sample_batch(values,4,np.array([0,1]),np.array([4,10]),torch.device('cpu'))
        self.assertEqual(x.shape,(2,5,1))
        self.assertEqual(x[0,-1,0],4)
        self.assertEqual(x[1,-2,0],21)
        with self.assertRaises(AssertionError):
            sample_batch(values,4,np.array([0]),np.array([3]),torch.device('cpu'))

    def test_validation_restores_heterogeneous_modes_and_all_cpu_rng_on_error(self):
        m=Toy().train();m.child.eval()
        py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with validation_mode(m,torch.device('cpu')):
                self.assertFalse(m.training)
                m.loss(torch.ones(2,5,1))
                raise RuntimeError('expected')
        self.assertTrue(m.training);self.assertFalse(m.child.training)
        self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(nr[1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(tr,torch.get_rng_state()))

    def test_selection_uses_role_contained_windows(self):
        x=np.arange(24,dtype=np.float32).reshape(2,12,1)
        m=Toy()
        got=selection(m,x,4,torch.device('cpu'),3)
        expected=np.stack([x[r,t-4:t+1]for r in range(2)for t in [4,8]])
        self.assertAlmostEqual(got,float((expected**2).mean()),places=4)

    def test_forbidden_role_rejected_before_file_hash_or_array_read(self):
        roles=object.__new__(RoleReader)
        roles.allowed={'FIT'}
        with patch('hybrid_tsad.normality_density_runner.sha',side_effect=AssertionError('must not hash')):
            with self.assertRaises(PermissionError):roles.load('QUERY')


if __name__=='__main__':unittest.main()
