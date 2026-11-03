"""Synthetic identities and arrays only. No public files/Torch imports."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import normal_readout as q, smd_recalibration as overlay
from hybrid_tsad.hybrid_v1 import event_adaptation as event, bank_recalibration as old_overlay
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,read_records,read_entries,_fingerprint,IDENTITY_KEYS
from hybrid_tsad.hybrid_v1.scoring import apply_calibration

def fixture(mode='Z_SOURCE_R3'):
    is_N=mode=='N_NEG300';choice='K1_choice0' if is_N else 'COMMON_SOURCE'
    assets={k:dict(path='/synthetic/'+k+'.json',sha256=(str(i%10)*64)) for i,k in enumerate(sorted(q.COMMON|(q.NORMAL_EXTRA if is_N else set())),1)}
    assets['parent_source_checkpoint']['sha256']=q.normal.SOURCE
    cfg=dict(operation=q.OPERATION,profile=q.PROFILE,mode=mode,choice=choice,method_id='N_NEG300_'+choice+'_EVENT_LOGIT' if is_N else 'Z_SOURCE_R3',
        score_kind='raw_event_logit' if is_N else 'source_calibrated_R3',source_seed=2025,inference_seed=2025,history=512,horizon=32,chunk_size=64,
        model=dict(history=512,width=128),sources={'synthetic':'external'},active_components=['r_tf','r_ms','r_pred'],assets=assets)
    base={k:cfg[k] for k in ('model','sources','history','horizon','active_components')};source=dict(base,seed=2025,operation='HYBRID_V1_SOURCE_STAGE1')
    sr=dict(phase='source_train',synthetic_non_scientific_results=False,files={'source.pt':q.normal.SOURCE,'config.json':assets['source_config']['sha256']})
    docs=dict(source_config=source,source_receipt=sr,training_config=copy.deepcopy(source),training_receipt=copy.deepcopy(sr),
        calibration_config=dict(base,operation='HYBRID_V1_SOURCE_STAGE1',checkpoint_sha256=q.normal.SOURCE),
        calibration=dict(source_domain_equal_weight=True,active_components=cfg['active_components'],weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),scales=dict(r_tf=1.,r_ms=2.,r_pred=3.),threshold=.5,valid_windows=1024,domains=['SMD','Sunspot']))
    for name in ('training_acceptance','calibration_acceptance'):docs[name]=dict(approved_by='root',status='ACCEPTED_SYNTHETIC_SCHEMA')
    if not is_N:
        assets['checkpoint']['sha256']=q.normal.SOURCE;assets['training_config']=assets['source_config'];assets['training_receipt']=assets['source_receipt']
        assets['training_acceptance']['sha256']=q.SOURCE_ACCEPT;assets['calibration']['sha256']=q.SOURCE_CAL
        docs['calibration_config']['train_receipt_sha256']=assets['training_receipt']['sha256']
        docs['calibration_receipt']=dict(phase='calibrate_source',synthetic_non_scientific_results=False,files={'source_calibration.json':q.SOURCE_CAL,'config.json':assets['calibration_config']['sha256']})
        docs['calibration_acceptance']['assets']=[dict(seed=2025,sha256=assets[k]['sha256']) for k in ('checkpoint','calibration','calibration_receipt')]
    else:
        ident=dict(domain='SMD',entity='synthetic',run='support');support=dict(K=1,events=[dict(ident,event_id='E0',start=6000,end=6010)],normal=[dict(ident,matched_event_id='E0',start=5000,end=5010)],forbidden=[])
        trace=dict(route='F',steps=[dict(attempt=a,updated=True,label=float(a%2==0),event_id='E0',footprint=[5488,6032] if a%2==0 else [4489,5033],normal_ssl_footprint=[4488,5032]) for a in range(300)])
        plan=q.normal.normal_plan_from_f_trace(trace,support,history=512,horizon=32,seed=2025,choice=choice,trace_sha256=assets['reference_trace']['sha256'])
        assets['support']['sha256']=event.SUPPORT[choice];assets['normal_input_manifest']['sha256']=event.ACCEPTED['manifest'];assets['normal_plans']['sha256']=event.ACCEPTED['plans']
        train=dict(base,operation=q.normal.OPERATION,choice=choice,mode='N_NEG300',steps=300,validate_every=50,lr=1e-4,weight_decay=.01,kind='huber',select_count=32,cal_count=512,seed=2025,source_checkpoint_sha256=q.normal.SOURCE)
        for key,name in [('manifest','normal_input_manifest'),('plans','normal_plans'),('support','support'),('normal_plan','normal_plan'),('reference_trace','reference_trace'),('reference_receipt','reference_receipt'),('reference_acceptance','reference_acceptance')]:train[key+'_sha256']=assets[name]['sha256']
        trained=dict(status='COMPLETED',operation=q.normal.OPERATION,phase='target_normal_train',mode='N_NEG300',loaded_roles=['NORMAL_SELECT','SUPPORT'],query_or_labels=False,adapted_checkpoint_sha256=assets['checkpoint']['sha256'],updates=300,positive_forward_exposures=0,selected_attempt=49)
        for key in ('choice','source_checkpoint_sha256','manifest_sha256','plans_sha256','support_sha256','normal_plan_sha256','reference_trace_sha256'):trained[key]=train[key]
        calcfg=dict(train,adapted_checkpoint_sha256=assets['checkpoint']['sha256'],adapted_receipt_sha256=assets['training_receipt']['sha256'])
        calr=dict(trained,phase='target_normal_calibrate',loaded_roles=['NORMAL_CAL'],updates=0,calibrated_checkpoint_sha256=assets['checkpoint']['sha256'],target_calibration_contact=True,calibration_file='event_logit_calibration.json',threshold=.5)
        docs.update(training_config=train,training_receipt=trained,calibration_config=calcfg,calibration_receipt=calr,calibration=dict(score_kind='event_logit',q=.99,method='higher',strict_greater=True,threshold=.5),normal_plan=plan,support=support,reference_trace=trace)
        docs['reference_receipt']=dict(trained,phase='target_event_train',adapted_checkpoint_sha256='f'*64)
        docs['reference_acceptance']=dict(approved_by='root',status='ACCEPTED_SYNTHETIC_SCHEMA',checkpoint_sha256='f'*64)
        docs['training_acceptance'].update(checkpoint_sha256=assets['checkpoint']['sha256'],receipt_sha256=assets['training_receipt']['sha256'])
        docs['calibration_acceptance'].update(calibration_sha256=assets['calibration']['sha256'],receipt_sha256=assets['calibration_receipt']['sha256'])
    seal=dict(status='ACCEPTED_SMD_MATCHED_QUERY_ASSETS',approved_by='root',mode=mode,choice=choice,assets=assets)
    return cfg,docs,seal

class MatchedReadout(unittest.TestCase):
    def test_Z_and_N_are_explicit_not_three_seeds(self):
        for mode in ('Z_SOURCE_R3','N_NEG300'):
            c,d,s=fixture(mode);r=q.binding(c,d,s);self.assertEqual(r['mode'],mode);self.assertEqual(r['geometry']['records'],6053)
            self.assertEqual(r['seed_rule'],'2025 + original_index - 17129')
    def test_F_U_phase_cannot_masquerade_as_N(self):
        for phase in ('target_event_train','target_train','source_train'):
            c,d,s=fixture('N_NEG300');d['training_receipt']['phase']=phase
            with self.assertRaises(ValueError):q.binding(c,d,s)
    def test_source_Z_rejects_targetCAL_or_adapted_model(self):
        c,d,s=fixture();d['calibration']['source_domain_equal_weight']=False
        with self.assertRaises(ValueError):q.binding(c,d,s)
        c,d,s=fixture();c['assets']['checkpoint']['sha256']='b'*64
        with self.assertRaises(ValueError):q.binding(c,d,s)
    def test_N_ownCAL_receipt_or_choice_swap_rejected(self):
        for key,value in [('adapted_checkpoint_sha256','c'*64),('plans_sha256','c'*64),('choice','K1_choice2')]:
            c,d,s=fixture('N_NEG300');d['calibration_config'][key]=value
            with self.assertRaises(ValueError):q.binding(c,d,s)
    def test_root_seal_pending_and_asset_mutation_rejected(self):
        c,d,s=fixture();s['status']='PENDING'
        with self.assertRaises(PermissionError):q.binding(c,d,s)
        c,d,s=fixture();s=copy.deepcopy(s);c['assets']['checkpoint']['path']+='changed'
        with self.assertRaises(PermissionError):q.binding(c,d,s)
    def test_rehashed_N_normal_plan_cannot_change_mask_or_sample(self):
        c,d,s=fixture('N_NEG300');d['normal_plan']['rows'][1]['B']['mask_seed']+=1
        with self.assertRaises(ValueError):q.binding(c,d,s)
    def test_query_seed_or_active_recipe_changed_rejected(self):
        for key,value in [('inference_seed',84551),('active_components',['r_ms','r_pred']),('mode','U')]:
            c,d,s=fixture();c[key]=value
            with self.assertRaises(PermissionError):q.binding(c,d,s)
    def test_missing_root_release_rejects_before_all_asset_io(self):
        with patch.object(q,'read',side_effect=AssertionError('No metadata/body I/O')):
            with self.assertRaises(PermissionError):q.preflight('nonexistent',None,None)
    def test_pending_root_asset_seal_precedes_PT_or_query_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,_,seal=fixture();c.update(output=str(root/'fresh'),source_sha256={},permissions=dict(query=True,labels=False,training=False,threshold_fit=False,target_fit=False))
            sp=root/'seal.json';sp.write_text(json.dumps(dict(seal,status='PENDING')));c['asset_seal']=dict(path=str(sp),sha256=q.sha(sp))
            cp=root/'config.json';cp.write_text(json.dumps(c));rp=root/'release.json'
            rp.write_text(json.dumps(dict(status='APPROVED',approved_by='root',operation=q.OPERATION,allowed_actions=[q.ACTION],allow_labels=False,config_sha256=q.sha(cp),asset_seal_sha256=c['asset_seal']['sha256'])))
            original_sha=q.sha
            def guarded(path):
                if str(path).startswith('/synthetic/'):self.fail('Asset bytes accessed before accepted seal')
                return original_sha(path)
            with patch.object(q,'code_gate'),patch.object(q,'sha',side_effect=guarded):
                with self.assertRaises(PermissionError):q.preflight(cp,rp,original_sha(rp))
    def test_injected_logit_writer_full6053_tail_and_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            run=SimpleNamespace(domain='SMD',entity='machine-2-1',run='machine-2-1_test',role='QUERY',start=17129,values=np.zeros((6565,38)))
            rows=[dict(domain=run.domain,entity=run.entity,run=run.run,index=t,available_index=t,event_features=np.zeros((1,259)),event_logit=1.,replay=dict(actual_mask_seed=2025+t-run.start),prediction=np.array([-0.,np.nan])) for t in range(17641,23694)]
            rows[63]['event_features'][0,0]=np.nan
            cal=dict(score_kind='event_logit',q=.99,method='higher',strict_greater=True,threshold=.5);ident={k:'a'*64 for k in IDENTITY_KEYS}
            q.helper.write_event_bank(None,[run],cal,Path(temp)/'bank',identity=ident,infer_fn=lambda *args,**kwargs:iter(copy.deepcopy(rows)))
            got=list(read_records(Path(temp)/'bank'));m=q.read(Path(temp)/'bank/manifest.json');entries=list(read_entries(Path(temp)/'bank'))
            self.assertEqual((len(got),len(m['chunks']),m['chunks'][-1]['records'],got[-1]['index']),(6053,95,37,23693))
            self.assertFalse(got[63]['valid']);self.assertNotIn('alarm',got[63]);self.assertEqual(got[63]['reason'],'NONFINITE_EVENT_FEATURES')
            for before,after in zip(rows,got):self.assertEqual(_fingerprint(before),_fingerprint({k:after[k] for k in before}))
            self.assertFalse(entries[64]['derived']['confirmed3_current']);self.assertTrue(entries[66]['derived']['confirmed3_current'])
    def test_cold_import_no_Torch(self):self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
