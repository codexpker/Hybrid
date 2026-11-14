"""New readout metadata/NumPy-only mocks; no real input, model or forward."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import stage2_u_readout as r
from hybrid_tsad.hybrid_v1 import stage2_u_matched as u


def write(root, name, value):
    p=root/name;p.write_text(json.dumps(value));return dict(path=str(p),sha256=u.sha(p))


def fixture(root, mode='U'):
    p=root/'QUERY.npz';np.savez(p,values=np.zeros((517,38),np.float32),observed=np.ones((517,38),bool))
    manifest=write(root,'manifest.json',dict(runs=[dict(domain='SMD',entity='machine-3-1',run='machine-3-1_test',role='QUERY',path=str(p),format='npz',key='values',observed_key='observed',sha256=u.sha(p),expected_shape=[517,38],bounds=[0,517],original_start=10110)]))
    pt=root/'parent.pt';pt.write_bytes(b'SYNTHETIC_NON_PT_NOT_LOADED');pa=dict(path=str(pt),sha256=u.sha(pt))
    ap=root/'adapted.pt';ap.write_bytes(b'SYNTHETIC_ADAPTED_NOT_LOADED');ca=dict(path=str(ap),sha256=u.sha(ap)) if mode=='U' else pa
    plan=write(root,'plan.json',dict(history=512,horizon=32,seed=2025,FIT=[],SELECT=[],CAL=[]))
    rec=dict(seed=2025,history=512,horizon=32,active_components=['r_tf','r_ms','r_pred'])
    model=write(root,'model.json',dict(rec,device='cpu',model={},sources={}))
    train=write(root,'train.json',dict(rec,operation=u.OPERATION if mode=='U' else 'HYBRID_V1_SOURCE_STAGE1',action='train_endpoint',method='plain',attempts=200,checkpoint_sha256=pa['sha256'],plan_sha256=plan['sha256'],manifest_sha256=manifest['sha256']))
    tr=write(root,'train_receipt.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',config_sha256=train['sha256'],checkpoint_sha256=ca['sha256'],parent_source_checkpoint_sha256=pa['sha256'],scientific_training_status='VALID_ALL_ATTEMPTS_UPDATED',updates=200,phase='source_train',files={'source.pt':pa['sha256'],'config.json':train['sha256']}))
    cal=dict(scales={k:1. for k in rec['active_components']},threshold=2.,active_components=rec['active_components'],weights={'r_tf':.25,'r_ms':.25,'r_pred':.5},source_domain_equal_weight=False)
    target=write(root,'cal.json',cal)
    cc=write(root,'calconfig.json',dict(rec,operation=u.OPERATION,action='calibrate',checkpoint_sha256=ca['sha256'],parent_source_checkpoint_sha256=pa['sha256'],plan_sha256=plan['sha256'],manifest_sha256=manifest['sha256']))
    cr=write(root,'calreceipt.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',config_sha256=cc['sha256'],checkpoint_sha256=ca['sha256'],scientific_training_status='ZERO_UPDATE_CAL',updates=0))
    assets=dict(checkpoint=ca,parent_source_checkpoint=pa,model_config=model,training_config=train,training_receipt=tr,target_calibration=target,target_calibration_config=cc,target_calibration_receipt=cr,manifest=manifest,plan=plan)
    if mode=='ZS_ZT':assets['source_calibration']=write(root,'sourcecal.json',dict(cal,source_domain_equal_weight=True))
    seal=write(root,'seal.json',dict(approved_by='root',status='ACCEPTED_MATCHED_U_READOUT_ASSETS',mode=mode,condition='A' if mode=='U' else 'COMMON',seed=2025,assets=assets))
    return dict(operation=r.OPERATION,mode=mode,seed=2025,condition='A' if mode=='U' else 'COMMON',history=512,horizon=32,chunk_size=64,resume=False,output=str(root/'fresh_output'),source_sha256={str(Path(r.__file__).resolve()):u.sha(r.__file__)},assets=assets,asset_seal=seal,active_components=rec['active_components'],ranges=[dict(domain='SMD',entity='machine-3-1',run='machine-3-1_test',start=10622,end=10627)],scored_points=5)


class MetadataAndViews(unittest.TestCase):
    def test_u_and_z_metadata_no_arrays_or_torch_needed(self):
        for mode in ['U','ZS_ZT']:
            with tempfile.TemporaryDirectory() as t:
                c=fixture(Path(t),mode)
                with patch.object(np,'load',side_effect=AssertionError('array body forbidden')):
                    m=r.metadata(c)
                self.assertFalse(m['model_forward']);self.assertFalse(m['arrays_opened'])

    def test_chain_seed_axis_resume_fresh_and_no_valid_updates_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c=fixture(root)
            for bad in [dict(c,seed=2026),dict(c,scored_points=4),dict(c,resume=True),dict(c,condition='best')]:
                with self.assertRaises((PermissionError,ValueError)):r.metadata(bad)
            receipt=json.loads(Path(c['assets']['training_receipt']['path']).read_text());receipt.update(updates=0,scientific_training_status='NO_VALID_UPDATES')
            c['assets']['training_receipt']=write(root,'train_receipt.json',receipt)
            seal=json.loads(Path(c['asset_seal']['path']).read_text());seal['assets']=c['assets'];c['asset_seal']=write(root,'seal.json',seal)
            with self.assertRaises(PermissionError):r.metadata(c)

    def test_parent_calibration_mismatch_is_not_papered_over_by_new_seal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c=fixture(root);cc=json.loads(Path(c['assets']['target_calibration_config']['path']).read_text());cc['checkpoint_sha256']='0'*64
            c['assets']['target_calibration_config']=write(root,'calconfig.json',cc);seal=json.loads(Path(c['asset_seal']['path']).read_text());seal['assets']=c['assets'];c['asset_seal']=write(root,'seal.json',seal)
            with self.assertRaises(PermissionError):r.metadata(c)

    def test_pending_release_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cfg=write(root,'cfg.json',fixture(root));rel=write(root,'rel.json',dict(status='PENDING'))
            with self.assertRaises(PermissionError):r.admit(cfg['path'],rel['path'],rel['sha256'])

    def test_z_views_rescore_same_synthetic_raw_without_forward(self):
        from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,read_records
        from hybrid_tsad.hybrid_v1.cli import score_records
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);identity={k:'a'*64 for k in ['input_sha256','config_sha256','checkpoint_sha256','calibration_sha256','source_sha256']}
            cal=dict(scales={'r_tf':1.,'r_ms':1.,'r_pred':1.},threshold=.5,active_components=['r_tf','r_ms','r_pred'],weights={'r_tf':.25,'r_ms':.25,'r_pred':.5})
            ranges=[dict(domain='synthetic',entity='tiny',run='synthetic',start=10,end=13)]
            with ChunkedBank(root/'source',identity=identity,ranges=ranges,threshold=.5,chunk_size=64) as bank:
                for i in range(10,13):bank.append(score_records([dict(domain='synthetic',entity='tiny',run='synthetic',index=i,r_tf=1.,r_ms=2.,r_pred=3.)],cal)[0])
                bank.finalize()
            before=u.sha(root/'source/manifest.json');target=dict(cal,scales={'r_tf':2.,'r_ms':2.,'r_pred':2.},threshold=3.)
            ans=r.rescore_view(root/'source',root/'target',target,dict(identity,calibration_sha256='b'*64))
            self.assertFalse(ans['model_forward']);self.assertEqual(before,u.sha(root/'source/manifest.json'))
            a=list(read_records(root/'source'));b=list(read_records(root/'target'))
            for x,y in zip(a,b):
                for k in ['index','r_tf','r_ms','r_pred']:self.assertEqual(x[k],y[k])
            self.assertNotEqual(a[0]['score'],b[0]['score'])
            with self.assertRaises(PermissionError):r.rescore_view(root/'source',root/'target',target,identity)

    def test_source_writer_is_called_once_and_never_for_each_view(self):
        code=Path(r.__file__).read_text();tree=ast.parse(code)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_model_bank']
        self.assertEqual(len(calls),1)

    def test_update_exposure_and_append_only_forensics_without_forward(self):
        self.assertEqual(u.exposure_status(0,200),'NO_VALID_UPDATES')
        self.assertEqual(u.exposure_status(199,200),'PARTIAL_UPDATE_EXPOSURE')
        self.assertEqual(u.exposure_status(200,200),'VALID_ALL_ATTEMPTS_UPDATED')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'trace.jsonl'
            try:
                u.append_trace(p,dict(attempt=0,status='ATTEMPT_STARTED'));raise RuntimeError('fixture')
            except RuntimeError:pass
            self.assertEqual(json.loads(p.read_text())['status'],'ATTEMPT_STARTED')
            u.append_trace(p,dict(attempt=0,status='FAILED_NO_RETRY'))
            self.assertEqual(len(p.read_text().splitlines()),2)


if __name__=='__main__':unittest.main(verbosity=2)
