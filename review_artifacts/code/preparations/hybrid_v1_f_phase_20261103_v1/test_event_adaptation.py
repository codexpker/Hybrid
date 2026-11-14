"""Synthetic NumPy + mocked kernel interface tests; no numerical model forward."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import event_adaptation as f,event_data as e
from hybrid_tsad.hybrid_v1.data import resolve_plan


class FPhase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        ident=dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test')
        labels=np.zeros(17129,np.int8)
        cat=[dict(event_id='SMD:machine-2-1:E%d'%i,start=a,end=b) for i,(a,b) in enumerate(e.EVENTS)]
        for x in cat: labels[x['start']:x['end']]=1
        normal=[dict(matched_event_id=cat[i]['event_id'],start=a,end=b) for i,(a,b) in enumerate(e.NORMAL)]
        r=e.construct(np.zeros((16585,2),np.float32),labels,identity=ident,roles=e.ROLES,catalogue=cat,normal=normal,history=512,horizon=32,counts=dict(NORMAL_SELECT=32,NORMAL_CAL=512))
        npz=self.root/'synthetic.npz';np.savez(npz,values=r['values'],observed=r['observed'])
        manifest=dict(runs=[dict(ident,role=role,path=npz.name,format='npz',key='values',observed_key='observed',sha256=f.sha(npz),expected_shape=[16585,2],bounds=b,original_start=0) for role,b in e.ROLES.items()])
        self.cfg=dict(operation=f.OPERATION,scope='synthetic',choice='K1_choice0',device='cpu',output=str(self.root/'out'),history=512,horizon=32,seed=2025,steps=6,validate_every=2,lr=1e-4,weight_decay=.01,kind='huber',select_count=32,cal_count=512,model={},sources={},active_components=['r_tf','r_ms','r_pred'])
        support=dict(K=1,events=r['combinations'][0]['events'],normal=r['combinations'][0]['normal'],forbidden=[dict(ident,start=9477,end=23694)])
        for key,data in [('manifest',manifest),('plans',dict(history=512,horizon=32,seed=2025,**r['plans'])),('support',support),('source_config',{k:self.cfg[k] for k in ('model','sources','history','horizon','active_components','seed')}),('source_receipt',dict(status='COMPLETED'))]:
            p=self.root/(key+'.json');p.write_text(json.dumps(data));self.cfg[key]=str(p);self.cfg[key+'_sha256']=f.sha(p)
        p=self.root/'source.pt';p.write_bytes(b'SYNTHETIC_NOT_TORCH');self.cfg['source_checkpoint']=str(p);self.cfg['source_checkpoint_sha256']=f.sha(p)
        self.cfg['permissions']={'query':False,'labels':False,'phase':'target_event_train'}

    def train(self):return f.preflight(self.cfg,'target_event_train')

    def backend(self):
        self.calls={}
        def model():return SimpleNamespace(bits={'event_head.weight':0,'backbone.weight':0,'buffer':0},named_parameters=lambda: [('event_head.weight',SimpleNamespace(requires_grad=True)),('backbone.weight',SimpleNamespace(requires_grad=True))])
        def adapt(source,route,fit,select,**kw):
            self.calls['adapt']=(route,fit,select,kw);self.assertNotIn('sampling_plan',kw)
            resolve_plan(select,kw['validation_plan'],512,32)
            after=model();after.bits['event_head.weight']=1;s=kw['support'];ev=s['events'][0];nm=s['normal'][0]
            rows=[dict(attempt=i,updated=True,label=float(i%2==0),event_id=ev['event_id'],footprint=[(ev if i%2==0 else nm)['start']-512,(ev if i%2==0 else nm)['start']+32],normal_ssl_footprint=[nm['start']-512,nm['start']+32]) for i in range(6)]
            return after,dict(steps=rows,updated_names=['event_head.weight'],updates=6,selected_attempt=5,selected_seen_event_ids=[ev['event_id']],selected_seen_matched_normal_ids=[ev['event_id']])
        def infer(model,runs,**kw):
            self.calls['infer']=kw;resolve_plan(runs,kw['index_plan'],512,32,arrived=True)
            return [dict(**{k:x[k] for k in ('domain','entity','run','index')},event_logit=float(i)/100,replay=dict(input_footprint=x['footprint'])) for i,x in enumerate(kw['index_plan'])]
        cli=SimpleNamespace(load_model=lambda c,p:model(),save_runtime=lambda *a:None,save_model=lambda p,m:Path(p).write_bytes(b'SYNTHETIC_ADAPTED'))
        kernel=SimpleNamespace(adapt=adapt,iter_infer=infer,logit_threshold=lambda records:float(np.quantile([r['event_logit'] for r in records],.99,method='higher')))
        return cli,kernel

    def test_train_routes_exact_F_plan_and_frozen_source(self):
        metadata=self.train()
        with patch.object(f,'backend',self.backend),patch.object(f,'state',lambda m:copy.deepcopy(m.bits)):
            receipt=f._action(self.cfg,'target_event_train',metadata)
        self.assertEqual(receipt['updates'],6);route,fit,select,kw=self.calls['adapt']
        self.assertEqual(route,'F');self.assertEqual([r.role for r in fit],['SUPPORT']);self.assertEqual([r.role for r in select],['NORMAL_SELECT']);self.assertEqual(len(kw['validation_plan']),32)

    def test_CAL_exact_512_arrived_logits_q99_zero_updates(self):
        with patch.object(f,'backend',self.backend),patch.object(f,'state',lambda m:copy.deepcopy(m.bits)):
            trained=f._action(self.cfg,'target_event_train',self.train())
        c=copy.deepcopy(self.cfg);c['adapted_checkpoint']=str(self.root/'out/adapted.pt');c['adapted_checkpoint_sha256']=f.sha(c['adapted_checkpoint']);c['adapted_receipt']=str(self.root/'out/receipt.json');c['adapted_receipt_sha256']=f.sha(c['adapted_receipt']);c['output']=str(self.root/'cal');c['permissions']={'query':False,'labels':False,'phase':'target_event_calibrate'}
        metadata=f.preflight(c,'target_event_calibrate')
        with patch.object(f,'backend',self.backend),patch.object(f,'state',lambda m:copy.deepcopy(m.bits)):
            receipt=f._action(c,'target_event_calibrate',metadata)
        self.assertEqual(receipt['updates'],0);self.assertEqual(len(self.calls['infer']['index_plan']),512);self.assertTrue(self.calls['infer']['event_logits']);self.assertEqual(receipt['threshold'],5.06)

    def test_missing_or_swapped_plan_fails_before_arrays_model(self):
        for mode in ('missing','changed'):
            c=copy.deepcopy(self.cfg)
            if mode=='missing':del c['plans']
            else:
                p=self.root/'replacement.json';p.write_text('{}');c['plans']=str(p)
            with patch.object(f,'load_manifest',side_effect=AssertionError('ARRAY')),patch.object(f,'backend',side_effect=AssertionError('MODEL')):
                with self.assertRaises((KeyError,PermissionError)):f.preflight(c,'target_event_train')

    def test_public_arbitrary_rehashed_plan_or_K5_rejected(self):
        c=copy.deepcopy(self.cfg);c['scope']='public_target';c['steps']=300;c['validate_every']=50
        with self.assertRaises(PermissionError):f.preflight(c,'target_event_train')
        c['source_checkpoint_sha256']='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812'
        with self.assertRaisesRegex(PermissionError,'Accepted exact'):f.preflight(c,'target_event_train')
        c['choice']='K5'
        with self.assertRaises(ValueError):f.preflight(c,'target_event_train')

    def test_rolewide_emptyplan_labels_and_source_change_rejected(self):
        for key,value in [('NORMAL_SELECT',[]),('NORMAL_CAL',[])]:
            p=f.read(self.cfg['plans']);p[key]=value;Path(self.cfg['plans']).write_text(json.dumps(p));self.cfg['plans_sha256']=f.sha(self.cfg['plans'])
            with self.assertRaises(ValueError):self.train()
        with self.assertRaises(PermissionError):f.no_labels(dict(annotation_ledger='NO'))
        Path(self.cfg['source_checkpoint']).write_bytes(b'CHANGED')
        with self.assertRaises(PermissionError):self.train()

    def test_trace_requires_all_K_and_correct_normalSSL(self):
        s=self.train()[2]
        bad=dict(history=512,horizon=32,steps=[],selected_seen_event_ids=[],selected_seen_matched_normal_ids=[])
        with self.assertRaises(ValueError):f.verify_trace(bad,s)
        with patch.object(f,'backend',self.backend),patch.object(f,'state',lambda m:copy.deepcopy(m.bits)):
            f._action(self.cfg,'target_event_train',self.train())
        trace=f.read(self.root/'out/training.json');trace['steps'][0]['normal_ssl_footprint'][0]+=1
        with self.assertRaises(ValueError):f.verify_trace(trace,s)

    def test_frozen_buffer_mutation_is_rejected(self):
        cli,kernel=self.backend();original=kernel.adapt
        def mutated(*args,**kwargs):
            adapted,trace=original(*args,**kwargs);adapted.bits['buffer']=1;return adapted,trace
        kernel.adapt=mutated
        with patch.object(f,'backend',lambda:(cli,kernel)),patch.object(f,'state',lambda m:copy.deepcopy(m.bits)):
            with self.assertRaisesRegex(ValueError,'Frozen'):f._action(self.cfg,'target_event_train',self.train())

    def test_no_release_cannot_load_arrays(self):
        p=self.root/'config.json';p.write_text(json.dumps(self.cfg))
        with patch.object(f,'load_manifest',side_effect=AssertionError('ARRAY')):
            with self.assertRaises(PermissionError):f.run_phase(p,None,None,'target_event_train')

    def test_truecold_metadata_preview_no_torch_or_worktree(self):
        source=Path(f.__file__).resolve().parents[2];cold=self.root/'cold';cold.mkdir()
        for name in f.RUNTIME_FILES:
            dest=cold/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
        c=self.root/'config.json';c.write_text(json.dumps(self.cfg))
        code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);assert not any('research_workspace' in p.lower() for p in sys.path);from hybrid_tsad.hybrid_v1 import event_adaptation as f;f.main(['--config',sys.argv[2],'--phase','target_event_train']);f.backend();files={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n=='hybrid_tsad' or n.startswith('hybrid_tsad.')};assert all(Path(sys.argv[1]).resolve() in Path(p).parents for p in files.values());assert 'torch' not in sys.modules;print(json.dumps(files))"
        r=subprocess.run([sys.executable,'-I','-c',code,str(cold),str(c)],cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr);print('F_COLD_PREVIEW '+r.stdout)

if __name__=='__main__':unittest.main()
