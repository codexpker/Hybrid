"""F06 lifecycle and import portability regressions; synthetic CPU only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import importlib.util
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]

class TestProfiles(unittest.TestCase):
    def runner(self):
        spec=importlib.util.spec_from_file_location('bounded_test_runner',ROOT/'scripts/run_test_profiles.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

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

    def test_case_timeout_is_identified_and_next_case_still_runs(self):
        import tempfile
        runner=self.runner();calls=[]
        def run(cmd, **kwargs):
            case=cmd[cmd.index('--case')+1];calls.append((case,kwargs['timeout']))
            if case=='synthetic.Case.slow':raise subprocess.TimeoutExpired(cmd,kwargs['timeout'])
            Path(cmd[cmd.index('--child-output')+1]).write_text(json.dumps(dict(status='PASS',tests_run=1,failures=[],errors=[],skipped=[])))
            return SimpleNamespace(returncode=0,stderr='')
        with tempfile.TemporaryDirectory() as tmp:
            result=runner.case_processes(['python','synthetic-runner'],dict(module='synthetic.py',cases=['synthetic.Case.slow','synthetic.Case.next'],excluded=[]),timeout=3,output_dir=Path(tmp),env={},cwd=ROOT,run=run)
        self.assertEqual(calls,[('synthetic.Case.slow',3),('synthetic.Case.next',3)])
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual([r['status'] for r in result['cases']],['TIMEOUT','PASS'])
        self.assertEqual(result['cases'][0]['test'],'synthetic.Case.slow')

    def test_empty_and_missing_child_reports_never_count_as_PASS(self):
        import tempfile
        runner=self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            args=dict(timeout=3,output_dir=Path(tmp),env={},cwd=ROOT)
            empty=runner.case_processes([],dict(module='synthetic.py',cases=[],excluded=[{'test':'external'}]),**args)
            self.assertEqual(empty['status'],'INTEGRATION_NOT_RUN')
            bad=runner.case_processes([],dict(module='synthetic.py',cases=['synthetic.Case.no_report'],excluded=[]),run=lambda *a,**k:SimpleNamespace(returncode=0,stderr='no report'),**args)
            self.assertEqual(bad['status'],'FAIL')
            self.assertEqual(bad['cases'][0]['status'],'PROCESS_ERROR')

if __name__=='__main__':unittest.main()
