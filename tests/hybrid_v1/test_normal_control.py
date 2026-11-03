"""Synthetic NumPy/metadata only. No real input, Torch, training or GPU."""
import copy
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import normal_control as n,event_adaptation as e


def fixture(root,steps=4):
    ident=dict(domain="SYNTHETIC",entity="C3",run="original")
    c=dict(operation=n.OPERATION,mode="N_NEG300",scope="synthetic",choice="K1_choice0",device="cpu",
        output=str(root/"out"),history=4,horizon=2,seed=2025,steps=steps,validate_every=2,lr=1e-4,
        weight_decay=.01,kind="huber",select_count=2,cal_count=3,model={},sources={},active_components=["r_tf","r_ms","r_pred"],
        permissions=dict(phase="target_normal_train",query=False,labels=False))
    def save(key,value):
        p=root/(key+".json");p.write_text(json.dumps(value),encoding="utf-8");c[key]=str(p);c[key+"_sha256"]=e.sha(p)
    npz=root/"synthetic.npz";np.savez(npz,values=np.ones((100,3),np.float32),observed=np.ones((100,3),bool))
    save("manifest",dict(runs=[dict(ident,role=role,path=npz.name,sha256=e.sha(npz),format="npz",key="values",observed_key="observed",expected_shape=[100,3],bounds=b,original_start=0)
         for role,b in [("SUPPORT",[0,30]),("NORMAL_SELECT",[40,60]),("NORMAL_CAL",[70,100])]]))
    def point(role,t):return dict(ident,role=role,index=t,footprint=[t-4,t+2])
    save("plans",dict(history=4,horizon=2,seed=2025,NORMAL_SELECT=[point("NORMAL_SELECT",44),point("NORMAL_SELECT",46)],NORMAL_CAL=[point("NORMAL_CAL",74),point("NORMAL_CAL",76),point("NORMAL_CAL",78)]))
    support=dict(K=1,events=[dict(ident,event_id="E0",start=15,end=17)],normal=[dict(ident,matched_event_id="E0",start=8,end=10)],forbidden=[dict(ident,start=30,end=100)])
    save("support",support);save("source_config",{k:c[k] for k in ("model","sources","history","horizon","seed","active_components")})
    pt=root/"source.pt";pt.write_bytes(b"SYNTHETIC_NOT_TORCH");c.update(source_checkpoint=str(pt),source_checkpoint_sha256=e.sha(pt))
    save("source_receipt",dict(phase="source_train",files={"source.pt":c["source_checkpoint_sha256"],"config.json":c["source_config_sha256"]}))
    trace=dict(route="F",steps=[dict(attempt=a,updated=True,label=float(a%2==0),event_id="E0",footprint=[(15 if a%2==0 else 8)-4,(15 if a%2==0 else 8)+2],normal_ssl_footprint=[4+a%2,10+a%2],actual_mask_seed=2025+a,normal_ssl_mask_seed=12025+a) for a in range(steps)])
    save("reference_trace",trace)
    save("reference_receipt",dict(status="COMPLETED",phase="target_event_train",adapted_checkpoint_sha256="a"*64,**{k:c[k] for k in ("source_checkpoint_sha256","manifest_sha256","plans_sha256","support_sha256","choice")}))
    save("reference_acceptance",dict(status="ACCEPTED_SYNTHETIC_FIXTURE",approved_by="root",adapted_checkpoint_sha256="a"*64))
    save("normal_plan",n.normal_plan_from_f_trace(trace,support,history=4,horizon=2,seed=2025,choice=c["choice"],trace_sha256=c["reference_trace_sha256"]))
    return c,support,trace


class NormalControl(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.c,self.support,self.trace=fixture(self.root)

    def test_300_normal450_positive0_no_resampling(self):
        trace=copy.deepcopy(self.trace);trace["steps"]=[dict(self.trace["steps"][a%4],attempt=a,actual_mask_seed=2025+a,normal_ssl_mask_seed=12025+a) for a in range(300)]
        plan=n.normal_plan_from_f_trace(trace,self.support,history=4,horizon=2,seed=2025,choice="K1_choice0",trace_sha256="b"*64)
        counts=n.validate_normal_plan(plan,self.support,history=4,horizon=2,seed=2025,steps=300,choice="K1_choice0")
        self.assertEqual(counts,dict(attempts=300,normal_SSL=300,normal_negative_BCE=150,normal_forward_exposures=450,positive_forward_exposures=0))
        self.assertEqual([r['B']['footprint'] for r in plan['rows']],[r['normal_ssl_footprint'] for r in trace['steps']]);self.assertTrue(all(r['A'] is None for r in plan['rows'][::2]))

    def test_positive_footprint_even_A_mask_and_U_trace_rejected(self):
        base=e.read(self.c['normal_plan'])
        for kind in ('positive','even','mask','group'):
            plan=copy.deepcopy(base)
            if kind=='positive':plan['rows'][1]['A'].update(index=15,footprint=[11,17])
            elif kind=='even':plan['rows'][0]['A']=copy.deepcopy(plan['rows'][1]['A'])
            elif kind=='mask':plan['rows'][0]['B']['mask_seed']+=1
            else:plan['rows'][0]['group']='WRONG'
            with self.subTest(kind=kind),self.assertRaises(ValueError):n.validate_normal_plan(plan,self.support,history=4,horizon=2,seed=2025,steps=4,choice='K1_choice0')
        with self.assertRaises(ValueError):n.normal_plan_from_f_trace(dict(self.trace,route='U'),self.support,history=4,horizon=2,seed=2025,choice='K1_choice0',trace_sha256='b'*64)

    def test_phase_metadata_exact_trace_not_values(self):
        with patch.object(n,'load_manifest',side_effect=AssertionError('No arrays')):self.assertEqual(len(n.preflight(self.c,'target_normal_train')[3]['rows']),4)
        p=Path(self.c['normal_plan']);v=e.read(p);v['rows'][1]['A']['index']=9;p.write_text(json.dumps(v));self.c['normal_plan_sha256']=e.sha(p)
        with self.assertRaises(ValueError):n.preflight(self.c,'target_normal_train')

    def test_public_rehashed_replacement_pool_and_wrong_mode_rejected(self):
        with self.assertRaises(PermissionError):n.preflight(dict(self.c,scope='public_target'),'target_normal_train')
        with self.assertRaises(PermissionError):n.preflight(dict(self.c,mode='U'),'target_normal_train')

    def normal_cal_config(self):
        c=copy.deepcopy(self.c);pt=self.root/'adapted.pt';pt.write_bytes(b'SYNTHETIC_ADAPTED_NOT_TORCH');c.update(adapted_checkpoint=str(pt),adapted_checkpoint_sha256=e.sha(pt),output=str(self.root/'Ncal'))
        receipt=dict(status='COMPLETED',operation=n.OPERATION,phase='target_normal_train',mode='N_NEG300',**{k:c[k] for k in ('choice','source_checkpoint_sha256','manifest_sha256','plans_sha256','support_sha256','normal_plan_sha256','reference_trace_sha256','adapted_checkpoint_sha256')})
        p=self.root/'Nreceipt.json';p.write_text(json.dumps(receipt));c.update(adapted_receipt=str(p),adapted_receipt_sha256=e.sha(p),permissions=dict(phase='target_normal_calibrate',query=False,labels=False));return c

    def test_CAL_rejects_F_receipt_and_Z_adapted_assets(self):
        c=self.normal_cal_config();p=Path(c['adapted_receipt']);r=e.read(p);r['phase']='target_event_train';p.write_text(json.dumps(r));c['adapted_receipt_sha256']=e.sha(p)
        with self.assertRaises(ValueError):n.preflight(c,'target_normal_calibrate')
        z=dict(c,mode='Z_TCAL_R3',permissions=dict(phase='source_target_calibrate',query=False,labels=False))
        with self.assertRaises(PermissionError):n.preflight(z,'source_target_calibrate')

    def test_both_CAL_use_same_origins_masks_and_Z_source_false(self):
        c=self.normal_cal_config();z=copy.deepcopy(self.c)
        for k in ('normal_plan','normal_plan_sha256'):z.pop(k)
        z.update(mode='Z_TCAL_R3',choice='COMMON_SOURCE',output=str(self.root/'Zcal'),permissions=dict(phase='source_target_calibrate',query=False,labels=False))
        calls=[];model=SimpleNamespace(bits={'weight':0},named_parameters=lambda:[('weight',SimpleNamespace(requires_grad=True))])
        def infer(model,runs,**kw):
            calls.append(kw);r=runs[0]
            return [dict(domain=r.domain,entity=r.entity,run=r.run,index=p['index'],r_tf=float(i+1),r_ms=float(i+2),r_pred=float(i+3),event_logit=float(i),replay=dict(input_footprint=p['footprint'],actual_mask_seed=2025+p['index']-r.start)) for i,p in enumerate(kw['index_plan'])]
        from hybrid_tsad.hybrid_v1 import cli,training
        with patch.object(cli,'load_model',return_value=model),patch.object(cli,'save_runtime'),patch.object(e,'state',side_effect=lambda m:copy.deepcopy(m.bits)),patch.object(training,'iter_infer',side_effect=infer):
            nc=n._action(c,'target_normal_calibrate',n.preflight(c,'target_normal_calibrate'));zc=n._action(z,'source_target_calibrate',n.preflight(z,'source_target_calibrate'))
        self.assertEqual(calls[0]['index_plan'],calls[1]['index_plan']);self.assertTrue(calls[0]['event_logits']);self.assertFalse(calls[1]['event_logits']);self.assertEqual(nc['updates'],zc['updates']);self.assertEqual(nc['threshold'],2.)
        self.assertIs(e.read(self.root/'Zcal/target_r3_calibration.json')['source_domain_equal_weight'],False)

    def test_missing_release_and_preview_no_IO(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('No IO')),contextlib.redirect_stdout(io.StringIO()):n.main(['--config','MISSING','--phase','target_normal_train'])
        with self.assertRaises(PermissionError):n.run_phase('MISSING',None,None,'target_normal_train')

    def test_exact_new_operation_code_release_gate_without_model(self):
        base=Path(n.__file__).resolve().parents[2];c=copy.deepcopy(self.c)
        c['source_sha256']={name:e.sha(base/name) for name in n.RUNTIME_FILES}
        cp=self.root/'config.json';cp.write_text(json.dumps(c));rp=self.root/'FAKE_SYNTHETIC_RELEASE.json'
        r=dict(status='APPROVED',approved_by='root',operation=n.OPERATION,allowed_actions=['target_normal_train'],config_sha256=e.sha(cp),source_sha256=c['source_sha256'])
        def sign():rp.write_text(json.dumps(r));return e.sha(rp)
        def action(cfg,phase,metadata):Path(cfg['output']).mkdir();return {'metadata_mock_only':True}
        with patch.dict(n.os.environ,CUDA_VISIBLE_DEVICES=''),patch.object(n,'_action',side_effect=action),patch.object(n,'load_manifest',side_effect=AssertionError('No arrays')):
            self.assertTrue(n.run_phase(cp,rp,sign(),'target_normal_train')['metadata_mock_only'])
        r['allowed_actions']=['target_event_train']
        with self.assertRaises(PermissionError):n.run_phase(cp,rp,sign(),'target_normal_train')

    def test_cold_closure_no_Torch(self):
        base=Path(n.__file__).resolve().parents[2];cold=self.root/'cold'
        for name in n.RUNTIME_FILES:
            p=cold/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(base/name),str(p))
        code="import sys;sys.path.insert(0,sys.argv[1]);from hybrid_tsad.hybrid_v1 import normal_control;assert 'torch' not in sys.modules;normal_control.main(['--config','MISSING','--phase','source_target_calibrate']);print('COLD_NORMAL_PASS')"
        result=subprocess.run([sys.executable,'-I','-c',code,str(cold)],cwd=self.root,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
