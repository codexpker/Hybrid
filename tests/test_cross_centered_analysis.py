"""Pure CPU consumer/math fixtures, no production truth or bank access."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1 import cross_centered_analysis as a

class Tests(unittest.TestCase):
    def test_one_policy_explicit_hash_gate(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.py';source.write_text('witness');s=dict(path=str(source),sha256=a.sha(source))
            p=Path(d)/'policy.json';p.write_text(json.dumps(dict(policy=a.POLICY,source_sha256=s['sha256'],new_parameters=0)))
            a.policy_gate(dict(path=str(p),sha256=a.sha(p)),s)
            p.write_text(json.dumps(dict(policy='ORIGINAL_SIGNED',source_sha256=s['sha256'],new_parameters=0)))
            with self.assertRaises(PermissionError):a.policy_gate(dict(path=str(p),sha256=a.sha(p)),s)

    def test_frozen_AR_does_not_fit_or_recalibrate(self):
        cfg=dict(original_controls=dict(path=str(ROOT.parent/'notes/paper_convergence_20261105/R2_bounded_mechanism_v1/run_cpu_controls.py')),
                 frozen_ar_controls=dict(path=str(ROOT.parent/'results/hybrid_v1_R2_matched_learnability_20261105_v1/CPU_FIXED_ANALYSIS/FITTED_AR_CONTROLS.json')))
        for v in cfg.values():v['sha256']=a.sha(v['path'])
        x=np.random.default_rng(919550).standard_normal((2560,3));data={case:x.copy() for case in a.CASES}
        with tempfile.TemporaryDirectory() as d:
            scores,pred,tau=a.frozen_ar(cfg,data,Path(d))
            saved=json.loads(Path(cfg['frozen_ar_controls']['path']).read_text())
            self.assertEqual(tau,saved['thresholds']);self.assertEqual(len(scores),10)
            helper=a.experiment.load_original_controls(cfg['original_controls']['path']);t=np.arange(512,2560)
            for method in ('AR_SELF','ARX12'):
                for channel in (1,2):
                    m=saved['models'][method][str(channel)]
                    expected=helper.predict(m,x,t,channel,None if method=='AR_SELF' else 12)
                    self.assertTrue(np.array_equal(expected,pred[method,'CROSS_N']['prediction'][:,channel]))
                self.assertTrue(np.isnan(pred[method,'CROSS_N']['prediction'][:,0]).all())
            self.assertNotIn('torch',sys.modules)

    def test_actual_index_metadata_preserves_raw_and_mapped(self):
        raw=np.arange(297).reshape(1,3,3,33)/500-.2;mapped=np.abs(raw);center=raw-mapped
        valid=np.broadcast_to((~np.eye(3,dtype=bool))[None,:,:,None],raw.shape).copy()
        src=np.array([[[1]*4,[0]*4,[0]*4]]);lags=np.broadcast_to(np.arange(4),(1,3,4)).copy()
        meta=dict(correlation=raw,overlap_count=np.ones(raw.shape)*64,cross_candidate_valid=valid,
            cross=dict(source=src,lag=lags,weights=np.ones((1,3,4))*.25,selected_valid=np.ones((1,3,4),bool)),
            cross_routing_policy=a.POLICY,cross_routing_score=mapped,cross_centering_term=center)
        entries=[dict(index=t,pass_name=mode,meta=copy.deepcopy(meta)) for t in a.experiment.ROUTE_ANCHORS
                 for mode in ('forecast','reconstruct_group0','reconstruct_group1')]
        s,g,n=a.routes_with_mapping(entries,'CENTERED_MAG','CROSS_N')
        self.assertEqual(len(g),45*297);self.assertEqual(len(n),90)
        self.assertTrue(all(r['routing_policy']==a.POLICY and r['applied_cross_routing_score']>=0 for r in g))
        self.assertTrue(any(r['signed_QK_correlation']<0 for r in g))

if __name__=='__main__':unittest.main(verbosity=2)
