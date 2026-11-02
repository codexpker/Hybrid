"""Metadata guards only; no Torch/model/weight/data construction."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hybrid_tsad.hybrid_v1 import initialization_audit as audit


class InitializationGuards(unittest.TestCase):
    def test_preview_zero_file_io(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('No reads')),contextlib.redirect_stdout(io.StringIO()) as log:
            audit.main(['--config','not_present'])
        self.assertIn('"preview": true',log.getvalue())

    def test_pending_before_reference_or_Torch(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'cfg';f.write_text('{}')
            with patch.object(audit,'prove',side_effect=AssertionError('No models')):
                with self.assertRaises(PermissionError):audit.admit(f,None,None)

    def test_only_named_fusion_difference_allowed(self):
        configs={name:dict(seed=2025,history=512,horizon=32,model=dict(fusion=mode,width=128),sources={},source_sha256={}) for name,mode in audit.MODES.items()}
        audit.config_equivalence(configs)
        for key,value in [('seed',2026),('history',100)]:
            bad=copy.deepcopy(configs);bad['none'][key]=value
            with self.assertRaises(ValueError):audit.config_equivalence(bad)
        bad=copy.deepcopy(configs);bad['feature_mean']['model']['width']=64
        with self.assertRaises(ValueError):audit.config_equivalence(bad)


if __name__=='__main__':unittest.main()
