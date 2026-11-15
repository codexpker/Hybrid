"""Synthetic identity fixtures test permission checks, never model execution."""
import json
from pathlib import Path
import tempfile
import unittest
from hybrid_tsad import g36_persistent_runner as runner


class FrozenReadoutPermissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.w=Path(self.tmp.name)
        self.cfg=dict(output='old/models',data='old/data');self.config=self.w/'config.json'
        self.config.write_text(json.dumps(self.cfg));root=self.w/self.cfg['output'];train=root/'linear/train';assess=root/'linear/assess'
        train.mkdir(parents=True);assess.mkdir();(self.w/'old/data').mkdir()
        for p in [train/'selected.pth',assess/'FIT_PRIOR.npz',self.w/'old/data/manifest.json',self.w/'old/data/scaler.npz']:p.write_bytes(b'synthetic permission fixture; no tensor is loaded')
        self.pt=runner.sha(train/'selected.pth');self.prior=runner.sha(assess/'FIT_PRIOR.npz')
        (train/'TRAIN_RECEIPT.json').write_text(json.dumps(dict(method='linear',selected_update=1200,selected_sha256=self.pt)))
        (assess/'ASSESS_RECEIPT.json').write_text(json.dumps(dict(method='linear',selected_update=1200,checkpoint_sha256=self.pt,prior_sha256=self.prior)))
        self.stop=root/'NORMAL_GATE.json';self.stop.write_text(json.dumps(dict(status='NORMAL_GATE_FAILED_STOP')));self.stop_hash=runner.sha(self.stop)
        paths=dict(config_sha256=self.config,train_receipt_sha256=train/'TRAIN_RECEIPT.json',checkpoint_sha256=train/'selected.pth',assess_receipt_sha256=assess/'ASSESS_RECEIPT.json',prior_sha256=assess/'FIT_PRIOR.npz',stop_gate_sha256=self.stop,data_manifest_sha256=self.w/'old/data/manifest.json',scaler_sha256=self.w/'old/data/scaler.npz',runner_sha256=Path(runner.__file__),model_sha256=Path(runner.__file__).with_name('g36_persistent_model.py'))
        self.release=dict(status='LINEAR_FROZEN_READOUT_RELEASED',method='linear',output='results/g36_linear_frozen_readout_20261116_v1/linear/score',**{k:runner.sha(v)for k,v in paths.items()})
        self.release_file=self.w/'release.json';self.save()
    def tearDown(self):
        self.assertEqual(self.stop_hash,runner.sha(self.stop));self.tmp.cleanup()
    def save(self):self.release_file.write_text(json.dumps(self.release))
    def test_exact_new_permission_does_not_change_old_stop(self):
        _,_,_,out=runner.frozen_linear_inputs(self.w,self.config,self.cfg,self.release_file,'linear')
        self.assertFalse(out.exists())
        with self.assertRaises(AssertionError):runner.score(self.w,self.config,self.cfg,out,'linear',None,self.stop)
    def test_full_old_gate_and_wrong_output_rejected(self):
        with self.assertRaises(PermissionError):runner.frozen_linear_inputs(self.w,self.config,self.cfg,self.release_file,'full')
        self.release['status']='NORMAL_GATE_PASS_QUERY_RELEASED';self.save()
        with self.assertRaises(PermissionError):runner.frozen_linear_inputs(self.w,self.config,self.cfg,self.release_file,'linear')
        self.release['status']='LINEAR_FROZEN_READOUT_RELEASED';self.release['output']='old/models/linear/score';self.save()
        with self.assertRaises(PermissionError):runner.frozen_linear_inputs(self.w,self.config,self.cfg,self.release_file,'linear')
    def test_changed_checkpoint_prior_or_runtime_rejected(self):
        for key in ['checkpoint_sha256','prior_sha256','runner_sha256','assess_receipt_sha256','train_receipt_sha256']:
            original=self.release[key];self.release[key]='wrong';self.save()
            with self.assertRaisesRegex(PermissionError,'identity mismatch'):runner.frozen_linear_inputs(self.w,self.config,self.cfg,self.release_file,'linear')
            self.release[key]=original


if __name__=='__main__':unittest.main()
