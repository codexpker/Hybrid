"""Pure metadata refusal tests; no actual model/input bytes or native infer."""
import ast
from pathlib import Path
import unittest
from unittest import mock
from hybrid_tsad import tep_normal_control_runtime as runtime

class ReadoutMetadata(unittest.TestCase):
    def test_pending_inputs_reject_before_refs_and_release(self):
        c=dict(binding_state='PENDING_NORMAL_CONTROL_INPUT_ACCEPTANCE',input_acceptance=None,input_manifest=None)
        with mock.patch.object(runtime,'read',return_value=c),mock.patch.object(runtime,'sha',side_effect=AssertionError('No hash-body')),mock.patch.object(runtime,'verified_ref',side_effect=AssertionError('No refs')):
            for stage in ['left','ridge']:
                with self.assertRaises(PermissionError):runtime.require_readout_release('cfg','release','a'*64,stage)

    def test_forged_trusted_release_hash_rejected_before_refs(self):
        c=dict(binding_state='BOUND_NORMAL_CONTROL_READOUT',input_acceptance={'path':'metadata','sha256':'b'*64},input_manifest={})
        with mock.patch.object(runtime,'read',return_value=c),mock.patch.object(runtime,'sha',return_value='b'*64),mock.patch.object(runtime,'verified_ref',side_effect=AssertionError('No refs')):
            with self.assertRaises(PermissionError):runtime.require_readout_release('cfg','release','a'*64,'left')

    def test_left_release_cannot_authorize_ridge(self):
        c=dict(binding_state='BOUND_NORMAL_CONTROL_READOUT',operation='TEP_NORMAL_CONTROL_LEFT_READOUT',input_acceptance={},input_manifest={})
        release=dict(status='PRIMARY_RELEASED_TEP_NORMAL_CONTROL_READOUT',approved_by='root',operation=c['operation'],
            config_sha256='a'*64,allow_normal20_scoring=True,allow_fit=False,allow_recalibration=False,allow_metrics=False)
        with mock.patch.object(runtime,'read',side_effect=lambda p:c if p=='cfg' else release),mock.patch.object(runtime,'sha',return_value='a'*64),mock.patch.object(runtime,'verified_ref',side_effect=AssertionError('No refs')):
            with self.assertRaises(PermissionError):runtime.require_readout_release('cfg','release','a'*64,'ridge')

    def test_no_model_import_at_module_import_and_no_old_guard_override(self):
        tree=ast.parse(Path(runtime.__file__).read_text(encoding='utf-8'))
        top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        names=[n.module if isinstance(n,ast.ImportFrom) else n.names[0].name for n in top]
        self.assertNotIn('torch',names);self.assertNotIn('pyreadr',names)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertIn('model_runtime',calls);self.assertIn('collect_normal_bank',calls)
        self.assertNotIn('execute_cost',calls);self.assertNotIn('collect_bank',calls)
        self.assertNotIn('fit',calls);self.assertNotIn('backward',calls)

if __name__=='__main__':unittest.main(verbosity=2)
