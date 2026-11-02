"""PREPARED, NOT EXECUTED. Root-released real Torch synthetic L0 only."""
import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import traceback
import unittest
from unittest.mock import patch
import numpy as np


@contextmanager
def evidence_directory():
    """Keep every already-written synthetic artifact, including partial failure.

    No TemporaryDirectory cleanup: all case files live under the root-specified
    fresh evidence directory. Export inventory is evidence, never a success fill.
    """
    name=os.environ.get('HYBRID_F_L0_OUTPUT')
    if not name: raise PermissionError('Root-specified HYBRID_F_L0_OUTPUT required')
    output=Path(name)
    if not output.is_absolute(): raise ValueError('Absolute fresh L0 output required')
    output.mkdir(parents=True,exist_ok=False)
    artifacts=output/'artifacts';artifacts.mkdir()
    status='FAILED';error=None
    previous_TERM=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame): raise TimeoutError('L0 received SIGTERM; retain partial evidence')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        yield str(artifacts)
        status='COMPLETED'
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        signal.signal(signal.SIGTERM,previous_TERM)
        inventory={}
        for path in sorted(artifacts.rglob('*')):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
                inventory[str(path.relative_to(output))]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
        expected=['synthetic_source.pt','synthetic.npz','manifest.json','plans.json','source_config.json','source_receipt.json',
                  'source_state_before.json','source_flags_before.json']
        for K in (1,3):
            expected+=['train%d/%s'%(K,p) for p in ['config.json','initial.pt','selected_partial.pt','adapted.pt',
                       'final_training_state.pt','recovery_latest.pt','training.json','training_trace.jsonl',
                       'initial_state_identity.json','adapted_state_identity.json','source_post_state_identity.json','state_difference.json','receipt.json']]
            expected+=['cal%d/%s'%(K,p) for p in ['config.json','normal_event_logit_records.json','event_logit_calibration.json',
                       'CAL_arrived_execution_plan.json','calibration_post_state_identity.json','receipt.json']]
            expected+=['nan_rejection%d.json'%K,'source_state_after_K%d.json'%K,'source_flags_after_K%d.json'%K]
        missing=[name for name in expected if not (artifacts/name).is_file()]
        if status=='COMPLETED' and missing:
            status='INCOMPLETE_EVIDENCE'
            error=dict(type='MissingEvidence',message='Expected synthetic artifacts absent',missing=missing)
        report=dict(status=status,error=error,release_sha256=os.environ.get('HYBRID_F_L0_RELEASE_SHA256'),
                    synthetic_only=True,artifacts=inventory,missing_expected=missing,
                    scope='Actual numeric test if executed; never real32/512 CAL or real dataset',
                    optimizer_RNG_location='trainK/final_training_state.pt and recovery_latest.pt from unchanged kernel')
        (output/'EVIDENCE_EXPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        if status=='INCOMPLETE_EVIDENCE': raise RuntimeError('Incomplete L0 evidence: '+repr(missing))


class ActualFKernelL0(unittest.TestCase):
    def test_K1_K3_actual_updates_frozen_source_and_CAL_nan_gate(self):
        if len(os.environ.get('HYBRID_F_L0_RELEASE_SHA256',''))!=64:
            raise PermissionError('Explicit root Torch L0 job release required; not part of CPU mock suite')
        with evidence_directory() as d:
            import torch
            from hybrid_tsad.hybrid_v1 import HybridV1Config,OfficialSources,event_adaptation as f
            from hybrid_tsad.hybrid_v1 import cli,training
            torch.set_num_threads(1)
            root=Path(d);identity=dict(domain='synthetic',entity='C3',run='original')
            config=dict(scope='synthetic',operation=f.OPERATION,device='cpu',seed=2025,history=128,horizon=4,steps=6,validate_every=2,lr=1e-4,weight_decay=.01,kind='huber',select_count=2,cal_count=3,
                model=json.loads(json.dumps(asdict(HybridV1Config(history=128,horizon=4,local_window=20,width=16,temporal_patch=16,temporal_layers=1,heads=2,tf_hidden=8,tf_repr=16,tf_depth=1,tf_patch=5,lag_window=20,max_lag=8,lag_width=8,lag_top_k=2,dropout=0.,history_mask_probability=.15)))),
                sources=asdict(OfficialSources.from_workspace(os.environ['HYBRID_V1_WORKSPACE'])),active_components=['r_tf','r_ms','r_pred'])
            source=cli.construct(config);before=f.state(source);flags={n:p.requires_grad for n,p in source.named_parameters()}
            checkpoint=root/'synthetic_source.pt';cli.save_model(checkpoint,source)
            (root/'source_state_before.json').write_text(json.dumps(before,indent=2))
            (root/'source_flags_before.json').write_text(json.dumps(flags,indent=2))
            array=np.random.default_rng(2025).normal(size=(1240,3)).astype(np.float32);npz=root/'synthetic.npz';np.savez(npz,values=array,observed=np.ones(array.shape,bool))
            roles={'SUPPORT':[0,560],'NORMAL_SELECT':[720,900],'NORMAL_CAL':[1060,1240]}
            manifest=dict(runs=[dict(identity,role=r,path=npz.name,format='npz',key='values',observed_key='observed',sha256=f.sha(npz),bounds=b,original_start=0,expected_shape=[1240,3]) for r,b in roles.items()])
            plans=dict(history=128,horizon=4,seed=2025,**{r:[dict(identity,role=r,index=t,footprint=[t-128,t+4]) for t in ts] for r,ts in [('NORMAL_SELECT',[848,850]),('NORMAL_CAL',[1188,1190,1192])]})
            for key,value in [('manifest',manifest),('plans',plans),('source_config',config),('source_receipt',{'synthetic':True})]:
                p=root/(key+'.json');p.write_text(json.dumps(value));config[key]=str(p);config[key+'_sha256']=f.sha(p)
            config.update(source_checkpoint=str(checkpoint),source_checkpoint_sha256=f.sha(checkpoint))
            for K in (1,3):
                c=copy.deepcopy(config);c.update(choice='K1_choice0' if K==1 else 'K3_choice3',steps=2*K,validate_every=2,permissions=dict(query=False,labels=False,phase='target_event_train'),output=str(root/('train%d'%K)))
                events=[dict(identity,event_id='E%d'%i,start=t,end=t+2) for i,t in enumerate([180,360,540][:K])]
                normal=[dict(identity,matched_event_id=x['event_id'],start=x['start']-7,end=x['start']-5) for x in events]
                support=dict(K=K,events=events,normal=normal,forbidden=[dict(identity,start=560,end=1240)])
                p=root/('support%d.json'%K);p.write_text(json.dumps(support));c.update(support=str(p),support_sha256=f.sha(p))
                # No mock: real model load/forward/backward/optimizer/validation.
                receipt=f._action(c,'target_event_train',f.preflight(c,'target_event_train'))
                trace=f.read(Path(c['output'])/'training.json');self.assertEqual(receipt['updates'],2*K)
                self.assertEqual(set(trace['selected_seen_event_ids']),{x['event_id'] for x in events});self.assertEqual(set(trace['selected_seen_matched_normal_ids']),{x['event_id'] for x in events})
                adapted=cli.load_model(c,Path(c['output'])/'adapted.pt');after=f.state(adapted)
                self.assertTrue(any(before[n]!=after[n] for n in before if n.startswith('event_head.')))
                self.assertEqual(f.state(source),before);self.assertEqual(flags,{n:p.requires_grad for n,p in source.named_parameters()})
                (root/('source_state_after_K%d.json'%K)).write_text(json.dumps(f.state(source),indent=2))
                (root/('source_flags_after_K%d.json'%K)).write_text(json.dumps({n:p.requires_grad for n,p in source.named_parameters()},indent=2))
                self.assertEqual(f.sha(checkpoint),config['source_checkpoint_sha256'])
                cal=copy.deepcopy(c);cal.update(output=str(root/('cal%d'%K)),permissions=dict(query=False,labels=False,phase='target_event_calibrate'),adapted_checkpoint=str(Path(c['output'])/'adapted.pt'),adapted_checkpoint_sha256=receipt['adapted_checkpoint_sha256'],adapted_receipt=str(Path(c['output'])/'receipt.json'),adapted_receipt_sha256=f.sha(Path(c['output'])/'receipt.json'))
                metadata=f.preflight(cal,'target_event_calibrate');result=f._action(cal,'target_event_calibrate',metadata)
                records=f.read(Path(cal['output'])/'normal_event_logit_records.json');self.assertEqual([r['index'] for r in records],[1188,1190,1192])
                self.assertEqual(result['threshold'],float(np.quantile([r['event_logit'] for r in records],.99,method='higher')))
                self.assertEqual([r['replay']['input_footprint'] for r in records],[[t-128,t+1] for t in [1188,1190,1192]])
                original=training.residual_components
                def nan_features(*args,**kwargs):
                    r=original(*args,**kwargs);r['event_features']=torch.full_like(r['event_features'],float('nan'));return r
                bad=copy.deepcopy(cal);bad['output']=str(root/('invalid_cal%d'%K))
                with patch.object(training,'residual_components',side_effect=nan_features):
                    with self.assertRaisesRegex(ValueError,'continuous valid event logits') as rejected:f._action(bad,'target_event_calibrate',f.preflight(bad,'target_event_calibrate'))
                self.assertFalse((Path(bad['output'])/'normal_event_logit_records.json').exists())
                (root/('nan_rejection%d.json'%K)).write_text(json.dumps(dict(injection='NaN event_features from real residual_components',
                    exception_type=type(rejected.exception).__name__,message=str(rejected.exception),
                    continuous_records_written=False,zero_fill_performed=False),indent=2))

if __name__=='__main__':unittest.main()
