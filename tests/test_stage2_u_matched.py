"""Incremental real-CPU tests only; no public input/weights or CUDA calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import stage2_u_matched as u


def fixture(root):
    entries = []
    for i, role in enumerate(["FIT", "SELECT", "CAL", "QUERY"]):
        p = root / (role + ".npz")
        x = np.random.default_rng(i).normal(size=(140, 3)).astype(np.float32)
        np.savez(p, values=x, observed=np.ones_like(x, bool))
        entries.append(dict(domain="SYNTHETIC", entity="tiny", run=role, role=role,
            path=str(p), format="npz", key="values", observed_key="observed", sha256=u.sha(p),
            expected_shape=[140, 3], bounds=[0, 140], original_start=1000*i))
    p = root / "manifest.json"; p.write_text(json.dumps(dict(runs=entries)))
    return p


def point(run, t=128):
    return dict(domain=run.domain, entity=run.entity, run=run.run, role=run.role,
                index=run.start+t, footprint=[run.start+t-128, run.start+t+2])


class Metadata(unittest.TestCase):
    def test_physical_only_requested_role_opened(self):
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t)); original=np.load
            def restricted(path, **kwargs):
                self.assertEqual(Path(path).stem, "FIT"); return original(path, **kwargs)
            with patch.object(np, "load", side_effect=restricted):
                self.assertEqual([r.role for r in u.load_roles(p, {"FIT"})], ["FIT"])
            # An absent CAL body is irrelevant to FIT loading, not silently opened.
            (Path(t)/"CAL.npz").unlink(); self.assertEqual(len(u.load_roles(p,{"FIT"})),1)

    def test_shared_container_and_extra_member_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t)); d=json.loads(p.read_text()); d['runs'][1]['path']=d['runs'][0]['path'];p.write_text(json.dumps(d))
            with self.assertRaises(PermissionError):u.role_metadata(p)
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t)); d=json.loads(p.read_text()); body=Path(t)/'FIT.npz'
            np.savez(body,values=np.zeros((140,3),np.float32),observed=np.ones((140,3),bool),CAL=np.ones((1,)))
            d['runs'][0]['sha256']=u.sha(body);p.write_text(json.dumps(d))
            with self.assertRaises(PermissionError):u.load_roles(p,{'FIT'})

    def test_plan_labels_role_axis_and_future_guard(self):
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));r=u.load_roles(p,{'FIT'})[0];e=point(r)
            self.assertEqual(u.resolve_u_plan([r],[e],128,2,'FIT')[0][1],128)
            for bad in [dict(e,role='QUERY'),dict(e,index=127),dict(e,label=1),dict(e,footprint=[0,1])]:
                with self.assertRaises((ValueError,PermissionError)):u.resolve_u_plan([r],[bad],128,2,'FIT')
            with self.assertRaises(ValueError):u.resolve_u_plan([r],[dict(e,index=139)],128,2,'FIT')

    def test_distinct_files_same_original_run_overlap_shortguard_and_hardlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));d=json.loads(p.read_text());d['runs']=d['runs'][:2]
            for e in d['runs']:e['run']='same_original_record'
            for start in [0,139,140+129]:
                d['runs'][1]['original_start']=start;p.write_text(json.dumps(d))
                with self.assertRaises(ValueError):u.role_metadata(p,128,2)
            d['runs'][1]['original_start']=270;p.write_text(json.dumps(d));u.role_metadata(p,128,2)
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));d=json.loads(p.read_text());a=Path(d['runs'][0]['path']);b=Path(d['runs'][1]['path']);b.unlink();os.link(a,b)
            with self.assertRaises(PermissionError):u.role_metadata(p)

    def test_pending_release_and_public_attempt_reject_before_model(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c=root/'c.json';rel=root/'r.json'
            c.write_text(json.dumps(dict(action='train_endpoint',source_sha256={})))
            rel.write_text(json.dumps(dict(status='PENDING')))
            with self.assertRaises(PermissionError):u.admit(c,rel,u.sha(rel))
        with self.assertRaises(ValueError):
            u.train_endpoint(None,[],[],{'SELECT':[]},history=512,horizon=32,seed=2025,attempts=2)
        with self.assertRaises(NotImplementedError):
            u.train_endpoint(None,[],[],{},history=128,horizon=2,seed=1,method='robust')


class ActualTinyCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from dataclasses import asdict
        from hybrid_tsad.hybrid_v1 import HybridV1Config, OfficialSources
        from hybrid_tsad.hybrid_v1.cli import construct
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        torch.set_num_threads(1)
        cfg=dict(seed=2025,device='cpu',history=128,horizon=2,
            model=asdict(HybridV1Config(history=128,horizon=2,local_window=20,width=16,
              temporal_layers=1,heads=2,tf_hidden=8,tf_repr=16,tf_depth=1,tf_patch=5,
              lag_window=20,max_lag=8,lag_width=8,lag_top_k=2,history_mask_probability=.15,dropout=0.)),
            sources=asdict(OfficialSources.from_workspace(os.environ['HYBRID_V1_WORKSPACE'])))
        cls.model=construct(cfg)

    def test_actual_modes_and_validation_restores_all_flags_and_cpu_rng(self):
        import torch
        m=copy.deepcopy(self.model);u.adaptation_mode(m)
        self.assertTrue(m.training);self.assertFalse(m.tf_backbone.training)
        self.assertTrue(m.fusion.training)
        self.assertTrue(any(isinstance(x,torch.nn.Dropout) and x.p==.1 and not x.training for x in m.tf_backbone.modules()))
        flags=[x.training for x in m.modules()];rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));r=u.load_roles(p,{'SELECT'});u.validate(m,r,[point(r[0])],128,2,2025)
        self.assertEqual(flags,[x.training for x in m.modules()]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(u.rng_devices(m),[])

    def test_actual_cpu_backward_endpoint_source_frozen_and_same_plan(self):
        from hybrid_tsad.hybrid_v1.stage2_adaptation import state_identity
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));fit=u.load_roles(p,{'FIT'});sel=u.load_roles(p,{'SELECT'})
            plan=dict(history=128,horizon=2,seed=2025,FIT=[point(fit[0],128),point(fit[0],129)],SELECT=[point(sel[0])]);before=state_identity(self.model)
            args=dict(history=128,horizon=2,seed=2025,attempts=2,validate_every=1,scope='synthetic')
            # An improving/worsening selector cannot choose a different endpoint.
            with patch.object(u,'validate',side_effect=[0.,100.]):m,report,opt=u.train_endpoint(self.model,fit,sel,plan,**args)
            with patch.object(u,'validate',side_effect=[100.,0.]):n,other,_=u.train_endpoint(self.model,fit,sel,plan,**args)
            self.assertEqual(state_identity(m),state_identity(n));self.assertEqual(before,state_identity(self.model))
            self.assertEqual(report['endpoint_attempt'],1);self.assertEqual(report['updates'],2)
            self.assertEqual([r['actual_mask_seed'] for r in report['trace']],[2025,2026])
            changed=report['state_difference']['changed_parameters']
            for prefix in u.ALLOWED:self.assertTrue(any(n.startswith(prefix+'.') for n in changed),prefix)
            self.assertFalse(report['state_difference']['changed_buffers']);self.assertTrue(opt.state_dict()['state'])
            self.assertEqual([e['index'] for e in report['trace']],[128,129])

    def test_skip_consumes_attempt_without_redraw_and_no_optimizer_update(self):
        from hybrid_tsad.hybrid_v1.stage2_adaptation import state_identity
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));fit=u.load_roles(p,{'FIT'});sel=u.load_roles(p,{'SELECT'})
            fit[0].observed[:]=False;plan=dict(history=128,horizon=2,seed=1,FIT=[point(fit[0])]*2,SELECT=[point(sel[0])])
            with patch.object(u,'validate',return_value=1.):
                m,r,o=u.train_endpoint(self.model,fit,sel,plan,history=128,horizon=2,seed=1,attempts=2,scope='synthetic')
            self.assertEqual(r['updates'],0);self.assertEqual(len(r['trace']),2)
            self.assertEqual(state_identity(m),state_identity(self.model));self.assertFalse(o.state_dict()['state'])

    def test_cal_zero_update_tail_arrived_and_legacy_entry_unmodified(self):
        from hybrid_tsad.hybrid_v1.stage2_adaptation import state_identity
        from hybrid_tsad.hybrid_v1.data import arrived_window
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));r=u.load_roles(p,{'CAL'});e=point(r[0],139);e['footprint']=[r[0].start+11,r[0].start+140]
            before=state_identity(self.model)
            records,cal=u.calibrate(self.model,r,[e],history=128,horizon=2,seed=2025)
            self.assertEqual(records[0]['index'],2139);self.assertEqual(before,state_identity(self.model))
            self.assertFalse(cal['source_domain_equal_weight'])
            self.assertFalse(arrived_window(r[0],139,128,32)['future_observed'][0,1:].any())

    def test_actual_cpu_exception_restores_modes_rng_and_cuda_device_stub(self):
        import torch
        from contextlib import nullcontext
        m=copy.deepcopy(self.model);u.adaptation_mode(m);flags=[x.training for x in m.modules()]
        rng=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with u.evaluation_state(m,5):
                torch.rand(3);raise RuntimeError('fixture')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(flags,[x.training for x in m.modules()])
        # Logic-only CUDA selection stub. No CUDA initialization, RNG or allocation.
        with patch.object(u,'rng_devices',return_value=[1]),patch.object(torch.random,'fork_rng',return_value=nullcontext()) as fork,patch.object(torch.cuda,'device',return_value=nullcontext()) as dev,patch.object(torch.cuda,'manual_seed') as seed:
            with u.evaluation_state(m,7):pass
            fork.assert_called_once_with(devices=[1]);dev.assert_called_once_with(1);seed.assert_called_once_with(7)

    def test_actual_forecast_does_not_consume_future_or_hidden_gt(self):
        import torch
        from hybrid_tsad.hybrid_v1.data import window
        from hybrid_tsad.hybrid_v1.training import tensor_batch,passes
        with tempfile.TemporaryDirectory() as t:
            p=fixture(Path(t));run=u.load_roles(p,{'FIT'})[0]
            b=tensor_batch(window(run,128,128,2));m=copy.deepcopy(self.model)
            with u.evaluation_state(m,1):
                f,r=passes(m,b,20)
                b['future_targets'][:]=1e6
                f2,_=passes(m,b,20)
                self.assertTrue(torch.equal(f['prediction'],f2['prediction']))
                hidden=r['hidden_mask'];context=b['reconstruction_context'].clone()
                # Explicit group masks retain the same visible input if hidden GT changes.
                h=hidden[:,0];context[h]+=1000
                a=m._encode(b['reconstruction_context'],b['reconstruction_observed'],h,mask_seed=20)
                c=m._encode(context,b['reconstruction_observed'],h,mask_seed=20)
                self.assertTrue(torch.equal(a['student'],c['student']));self.assertTrue(torch.equal(a['stats']['mean'],c['stats']['mean']))


if __name__=='__main__':unittest.main(verbosity=2)
