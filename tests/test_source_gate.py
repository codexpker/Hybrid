"""Synthetic-only source L0 authorization/bounds tests; no scientific results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid_tsad.engine import source_gate as gate

READY=all(importlib.util.find_spec(n) is not None for n in ('torch','numpy','pandas'))


@unittest.skipUnless(READY,'Optional existing torch/numpy/pandas required; never install for this test')
class SourceGateSyntheticTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pandas as pd
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        p=self.root
        (p/'contract.md').write_text('SYNTHETIC source-only five-step diagnostic fixture')
        (p/'official.py').write_text('# synthetic external identity; actual model factory is mocked')
        self.official_sha=gate.sha256(p/'official.py')
        rows=[]
        for role in ('fit','select','cal'):
            for source in ('PSM','SMD_SUBSET'):
                values=np.stack([np.sin(np.arange(200)/9),np.cos(np.arange(200)/7)],axis=1).astype('float32')
                if role=='fit' and source=='PSM':values[50,0]=np.nan;values[101,0]=np.nan
                f=p/f'{source}_{role}.csv'
                if source=='PSM':pd.DataFrame({'time':np.arange(200),'a':values[:,0],'b':values[:,1]}).to_csv(f,index=False)
                else:np.savetxt(f,values,delimiter=',')
                rows.append(dict(dataset=source,entity=source,role=role,source=str(f),history=100,
                    horizon=5,local_start=0,local_end_exclusive=200,columns=['a','b'],forecast_origins=[100]))
        self.write('manifest.json',{'records':rows});self.write('rows.json',rows)
        arrays={}
        for role,n,offset in [('fit',640000,0),('select',256,2),('cal',256,4)]:
            arrays[role+'_record']=(offset+(np.arange(n)//64)%2).astype('uint16')
            arrays[role+'_origin']=np.full(n,100,'uint32');arrays[role+'_channel']=np.zeros(n,'uint8')
        np.savez_compressed(p/'plan.npz',**arrays)
        hashes={r['source']:gate.sha256(r['source']) for r in rows}
        hashes.update({str(p/'manifest.json'):gate.sha256(p/'manifest.json'),str(p/'official.py'):self.official_sha})
        self.write('inputs.json',{'manifest':str(p/'manifest.json'),'official':str(p/'official.py'),'hashes':hashes})
        self.write('entry.json',{'status':'PENDING_NOT_A_PASS'})
        self.write('train_approval.json',{'status':'PENDING','training_released':False})
        self.jobs={f'fixture_{m}':{'mode':m,'sources':['PSM','SMD_SUBSET'],'plan':'same','seed':0}
                   for m in ('temporal','tf','hybrid')}
        self.wave={'history':100,'horizon':5,'target':'TEP','steps':10000,'batch':64,'lr':.0003,
                   'weight_decay':.01,'warmup':500,'validate_every':500,'query_evaluation':False,
                   'mask_seed':84551,'jobs':self.jobs}
        self.write('wave.json',self.wave)
        case={'original_plan_index':0,'record':0,'origin':100,'channel':0}
        self.write('cases.json',{'history':100,'horizon':5,'first5_batch_slices':[[i*64,(i+1)*64] for i in range(5)],
                   'source_mask_cases':{'first_internal_history_missing':case,'first_partial_future_missing':case}})
        self.write('gate_approval.json',{'status':'PENDING'})
        src=Path(gate.__file__).resolve().parents[1]
        self.cfg={'diagnostic_schema_version':1,'purpose':'source_gate_l0','contract':self.asset('contract.md'),
            'approval':self.asset('gate_approval.json'),'source_operations':{},'cases':self.asset('cases.json'),
            'package_code':{str(f):gate.sha256(f) for f in src.rglob('*.py')},
            'parameters':{'optimizer_steps':5,'batch':64,'device':'cpu',
                'seed_by_job':{k:0 for k in self.jobs},'mask_seed':84551,
                'max_seconds_per_process':120,'max_total_process_GPU_hours':.1},'output':str(p/'diagnostic')}
        self.refresh()

    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):(self.root/name).write_text(json.dumps(value,indent=2))
    def asset(self,name):return {'path':str(self.root/name),'sha256':gate.sha256(self.root/name)}

    def refresh(self,approved=True):
        self.write('wave.json',self.wave)
        for key in self.jobs:
            op={'engine_schema_version':3,'operation':'train','contract':self.asset('contract.md'),
                'approval':self.asset('train_approval.json'),'inputs':{},'parameters':{'job':key,'device':'cpu'},
                'score_space':'history_normalized_abs_channelmean_div_source_median_v1','output':str(self.root/'formal'/key)}
            for k,v in {'wave_config':'wave.json','source_inputs':'inputs.json','source_manifest':'manifest.json',
                        'plan':'plan.npz','rows':'rows.json','official':'official.py','entry_gate':'entry.json'}.items():op['inputs'][k]=self.asset(v)
            self.write(key+'.json',op);self.cfg['source_operations'][key]=self.asset(key+'.json')
        self.cfg['approval']=self.asset('gate_approval.json');self.write('gate.json',self.cfg)
        digest=gate.prepare(self.root/'gate.json')[-1]
        self.write('gate_approval.json',{'status':'APPROVED' if approved else 'PENDING','scope':'SOURCE_L0_ONLY',
            'contract_sha256':self.asset('contract.md')['sha256'],'diagnostic_manifest_sha256':digest,
            'allowed_jobs':list(self.jobs),'training_released':False})
        self.cfg['approval']=self.asset('gate_approval.json');self.write('gate.json',self.cfg)

    def test_preview_never_creates_model_or_outputs(self):
        self.refresh(approved=False)
        with patch.object(gate,'_create_model',side_effect=AssertionError('model opened')):
            r=gate.run(self.root/'gate.json')
        self.assertFalse(r['executed']);self.assertFalse((self.root/'diagnostic').exists())

    def test_pending_approval_rejected(self):
        self.refresh(approved=False)
        with self.assertRaisesRegex(ValueError,'SOURCE_L0_ONLY'):
            gate.run(self.root/'gate.json',execute=True,job='fixture_temporal')

    def test_unknown_job_and_budget_rejected(self):
        with self.assertRaisesRegex(ValueError,'approved diagnostic jobs'):
            gate.run(self.root/'gate.json',execute=True,job='unknown')
        self.cfg['parameters']['optimizer_steps']=6;self.write('gate.json',self.cfg)
        with self.assertRaisesRegex(ValueError,'five batch64'):gate.prepare(self.root/'gate.json')

    def test_rebound_wrong_seed_is_rejected(self):
        self.wave['jobs']['fixture_hybrid']['seed']=1;self.refresh()
        with patch.object(gate,'OFFICIAL_SHA',self.official_sha):
            with self.assertRaisesRegex(ValueError,'seed differs'):
                gate.validate_inputs(self.root/'gate.json','fixture_temporal')

    def test_rebound_target_source_is_rejected(self):
        self.wave['jobs']['fixture_hybrid']['sources']=['PSM','TEP'];self.refresh()
        with patch.object(gate,'OFFICIAL_SHA',self.official_sha):
            with self.assertRaisesRegex(ValueError,'target exclusion'):
                gate.validate_inputs(self.root/'gate.json','fixture_temporal')

    def test_five_kernel_updates_masks_no_checkpoint_or_gate_promotion(self):
        import torch
        from torch import nn
        class Tiny(nn.Module):
            def __init__(self,history,mode):
                super().__init__();self.history=history;self.mode=mode;self.head=nn.Linear(history,5)
                self.inactive=nn.Parameter(torch.ones(1),requires_grad=False)
            def parameter_counts(self):return {'stored':sum(p.numel() for p in self.parameters()),'active':sum(p.numel() for p in self.parameters() if p.requires_grad)}
            def forward(self,x):
                valid=torch.isfinite(x);raw=torch.nan_to_num(x);count=valid.sum(1,keepdim=True)
                mean=(raw*valid).sum(1,keepdim=True)/count
                scale=(((raw-mean).square()*valid).sum(1,keepdim=True)/count+1e-5).sqrt()
                pred=self.head(((raw-mean)/scale).squeeze(-1)).unsqueeze(-1)
                return {'prediction_normalized':pred,'prediction':pred*scale+mean,'mean':mean,'scale':scale,
                        'tf_reconstruction':None,'tf_history_normalized':None,'tf_valid':None}
        with patch.object(gate,'OFFICIAL_SHA',self.official_sha),patch.object(gate,'_create_model',
                side_effect=lambda official,history,mode,device:Tiny(history,mode).to(device)):
            report=gate.run(self.root/'gate.json',execute=True,job='fixture_temporal')
        self.assertEqual(report['status'],'CPU_ONLY_NOT_A_GPU_ENTRY_GATE')
        self.assertFalse(report['formal_checkpoints_written']);self.assertFalse(report['training_released'])
        r=report['results'][0];self.assertEqual(r['optimizer_steps'],5);self.assertEqual(r['exposure_windows'],320)
        self.assertEqual(len(r['mask_checks']),2);self.assertEqual(r['curve'][0]['lr'],.0003/500)
        self.assertEqual(json.loads((self.root/'entry.json').read_text())['status'],'PENDING_NOT_A_PASS')
        self.assertFalse(list(self.root.rglob('*.pt')));self.assertFalse((self.root/'formal').exists())


if __name__=='__main__':unittest.main()
