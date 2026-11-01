"""Pure controller/metadata fixtures: no model, array body, or optimizer."""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hybrid_tsad import left_normal_public as adapter


class TrainingContractTests(unittest.TestCase):
    def test_pending_release_refuses_before_array_or_model(self):
        cfg={'operation':'LEFT_TEP_C3_H100_NORMAL_TRAIN','adapter_hashes':{'a':'b'},
            'seed':2026,'job_id':'LEFT_TEP_C3_H100_NORMAL_SEED2026','physical_gpu_index':1,
            'expected_gpu_uuid':'514a18de-bcd9-8992-c47f-775f98aed4b6'}
        with patch.object(adapter,'preview_train',return_value=(cfg,Path('.'),{})), \
             patch.object(adapter.baseline,'read_json',return_value={'status':'PENDING'}), \
             patch.object(adapter.baseline,'file_sha',return_value='fixture'), \
             patch.object(adapter,'load_normal_arrays',side_effect=AssertionError('Actual array not permitted')) as arrays:
            with self.assertRaises(PermissionError):adapter.train_normal('fake','pending')
            arrays.assert_not_called()

    def test_native_loop_control_first_eligible_epoch4_and_early_stop(self):
        class Native:
            def __init__(self,**kw):self.counter=0;self.early_stop=False;self.val_loss_min=float('inf');self.saved=[]
            def __call__(self,loss,model,path):
                if loss<=self.val_loss_min:
                    self.counter=0;self.val_loss_min=loss;self.saved.append(model.step)
                else:self.counter+=1;self.early_stop=self.counter>=3
        official=SimpleNamespace(EarlyStopping=Native);events=[]
        with adapter.eligible_checkpoints(official,events):
            stop=official.EarlyStopping()
            for epoch,loss in enumerate([.01,.01,.01,10,11,12,13],1):
                stop(loss,SimpleNamespace(step=epoch*94),'no actual checkpoint')
                if epoch<4:self.assertEqual((stop.saved,stop.counter,stop.early_stop),([],0,False))
            self.assertEqual(stop.saved,[376]);self.assertTrue(stop.early_stop)
        self.assertIs(official.EarlyStopping,Native)

    def test_training_entry_reuses_parent_train_and_prohibits_other_data(self):
        tree=ast.parse(Path(adapter.__file__).read_text())
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_normal')
        calls=[n for n in ast.walk(train) if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
            and n.func.value.id=='exp' and n.func.attr=='train' for n in calls),1)
        for forbidden in ('test','evaluate_spot','backward','step'):
            self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr==forbidden for n in calls))
        # Native optimizer hooks observe updates, never perform them themselves.
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=='register_step_post_hook' for n in calls))

    def test_preview_rejects_seed_weights_caps_and_unknown_operation(self):
        path=os.environ['NORMAL_TRAIN_CONFIG'];cfg=json.loads(Path(path).read_text())
        for field,value in [('seed',2028),('checkpoint','forbidden.pt'),('max_seconds',600),
                            ('max_epochs',21),('operation','score'),('eligibility_step',368)]:
            altered=copy.deepcopy(cfg);altered[field]=value
            with patch.object(adapter.baseline,'read_json',return_value=altered):
                with self.assertRaises(PermissionError):adapter.preview_train('fake')

    def test_exact_predeclared_seed_device_mapping_and_release_whitelist(self):
        for seed,(gpu,uuid) in adapter.TRAIN_SEED_DEVICES.items():
            cfg={'seed':seed,'job_id':f'LEFT_TEP_C3_H100_NORMAL_SEED{seed}',
                'physical_gpu_index':gpu,'expected_gpu_uuid':uuid,'operation':'LEFT_TEP_C3_H100_NORMAL_TRAIN',
                'adapter_hashes':{'fake':'fixture'}}
            adapter.validate_training_identity(cfg)
            for key,value in [('physical_gpu_index',(gpu+1)%4),('expected_gpu_uuid','wrong'),
                              ('job_id','LEFT_TEP_C3_H100_NORMAL_SEED2028')]:
                with self.assertRaises(PermissionError):adapter.validate_training_identity(dict(cfg,**{key:value}))
            release={'status':'PRIMARY_RELEASED_LEFT_TEP_C3_H100_NORMAL_TRAIN','approved_by':'root',
                'operation':cfg['operation'],'config_sha256':'fixture','adapter_hashes':cfg['adapter_hashes'],
                'seed':seed,'job_id':cfg['job_id'],'allowed_jobs':[cfg['job_id']],
                'physical_gpu_index':gpu,'expected_gpu_uuid':uuid,'allow_normal_FIT_SELECT':True,
                'allow_CAL':False,'allow_Faulty_Testing':False,'allow_query_labels':False,
                'allow_checkpoint_initialization':False}
            with patch.object(adapter.baseline,'read_json',return_value=release), \
                 patch.object(adapter.baseline,'file_sha',return_value='fixture'):
                adapter.require_train_release('fake','fake',cfg)
            for field,value in [('seed',2028),('allowed_jobs',[]),('physical_gpu_index',(gpu+1)%4),
                                ('expected_gpu_uuid','wrong')]:
                with patch.object(adapter.baseline,'read_json',return_value=dict(release,**{field:value})), \
                     patch.object(adapter.baseline,'file_sha',return_value='fixture'):
                    with self.assertRaises(PermissionError):adapter.require_train_release('fake','fake',cfg)


if __name__=='__main__':unittest.main()
