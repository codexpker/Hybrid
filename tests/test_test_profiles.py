"""F06 lifecycle and import portability regressions; synthetic CPU only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

class TestProfiles(unittest.TestCase):
    def test_resource_unavailable_still_imports_pure_helpers(self):
        code="""
import builtins
original=builtins.__import__
def deny(name,*args,**kwargs):
    if name=='resource': raise ModuleNotFoundError('Windows fixture has no resource')
    return original(name,*args,**kwargs)
builtins.__import__=deny
from hybrid_tsad.tep_normal_assets import window_schedule
from hybrid_tsad.engine.weak_lag_wave import require_release
assert window_schedule()
"""
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='')
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_manifest_rules_are_explicit_and_unique(self):
        rules=json.loads((ROOT/'tests/profiles.json').read_text())['integration']
        keys=[r['id_prefix'] for r in rules]
        self.assertEqual(len(set(keys)),len(keys))
        self.assertTrue(all(r['requires'] for r in rules))

if __name__=='__main__':unittest.main()
