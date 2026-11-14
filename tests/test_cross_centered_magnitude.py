"""Incremental math tests; Torch witness is skipped on a Torch-free host."""
import ast,copy,importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

class MathTests(unittest.TestCase):
    def test_overlap_center_identity_sign_translation_and_zero(self):
        rng=np.random.default_rng(63051)
        for n in (0,1,64):
            x=rng.normal(size=n);y=rng.normal(size=n);wq=rng.normal(size=7);wk=rng.normal(size=7)
            bq=rng.normal(size=7);bk=rng.normal(size=7);d=np.sqrt(7);den=max(n,1)
            q=x[:,None]*wq+bq;k=y[:,None]*wk+bk
            s=np.sum(q*k)/(den*d);center=(q.sum(0)@k.sum(0))/(den*den*d)
            a=wq@wk/d;cov=np.sum(x*y)/den-x.sum()*y.sum()/(den*den)
            self.assertAlmostEqual(s-center,a*cov,places=12)
            qs=(x+3)[:,None]*wq+bq;ks=(y-4)[:,None]*wk+bk
            shifted=np.sum(qs*ks)/(den*d)-(qs.sum(0)@ks.sum(0))/(den*den*d)
            self.assertAlmostEqual(abs(shifted),abs(s-center),places=12)
            flipped=(-y)[:,None]*wk+bk
            neg=np.sum(q*flipped)/(den*d)-(q.sum(0)@flipped.sum(0))/(den*den*d)
            self.assertAlmostEqual(abs(neg),abs(s-center),places=12)
            self.assertTrue(np.isfinite(s-center))

    def test_original_file_and_no_new_parameter_constructor(self):
        p=ROOT/'src/hybrid_tsad/hybrid_v1/cross_centered_magnitude.py'
        tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        self.assertNotIn('__init__',[n.name for n in cls.body if isinstance(n,ast.FunctionDef)])
        self.assertEqual(cls.bases[0].id,'FiniteLag')
        own=ast.parse((p.parent/'lag.py').read_text());old=next(n for n in own.body if isinstance(n,ast.ClassDef))
        old_f=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        new_f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        # Original Self aggregation and projection are bytecode-AST equivalent.
        def self_nodes(f):
            found=[]
            for s in f.body:
                if not isinstance(s,ast.Assign):continue
                t=s.targets[0]
                if isinstance(t,ast.Tuple) and [getattr(z,'id',None) for z in t.elts]==['own','own_meta']:
                    found.append(ast.dump(s,include_attributes=False))
                if isinstance(t,ast.Name) and t.id=='own':found.append(ast.dump(s,include_attributes=False))
            return found
        self.assertEqual(len(self_nodes(old_f)),2)
        self.assertEqual(self_nodes(old_f),self_nodes(new_f))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Actual Torch execution unavailable on this host')
class TorchTests(unittest.TestCase):
    def test_actual_self_state_rng_deepcopy_reload_and_empty_support(self):
        import torch
        from hybrid_tsad.hybrid_v1.config import HybridV1Config
        from hybrid_tsad.hybrid_v1.lag import FiniteLag
        from hybrid_tsad.hybrid_v1.cross_centered_magnitude import install,policy_identity
        torch.set_num_threads(1);torch.manual_seed(63051)
        cfg=HybridV1Config(history=100,lag_window=64,max_lag=16)
        m=SimpleNamespace(config=cfg,lag=FiniteLag(cfg))
        x=torch.randn(2,100,3);visible=torch.ones_like(x,dtype=torch.bool)
        before={k:v.clone() for k,v in m.lag.state_dict().items()};original=m.lag(x,visible)
        rng=torch.get_rng_state().clone();identity=install(m)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(identity['new_parameters'],0)
        for k,v in m.lag.state_dict().items():self.assertTrue(torch.equal(before[k],v))
        own,cross,meta=m.lag(x,visible)
        self.assertTrue(torch.equal(original[0],own))
        for k in ('correlation','overlap_count','self_candidate_valid','cross_candidate_valid'):
            self.assertTrue(torch.equal(original[2][k],meta[k]))
        twin=copy.deepcopy(m);other=twin.lag(x,visible)
        self.assertTrue(torch.equal(own,other[0]));self.assertTrue(torch.equal(cross,other[1]))
        re=SimpleNamespace(config=cfg,lag=FiniteLag(cfg));re.lag.load_state_dict(m.lag.state_dict())
        with self.assertRaises(PermissionError):policy_identity(re)
        install(re);actual=re.lag(x,visible)
        self.assertTrue(torch.equal(cross,actual[1]))
        empty=m.lag(x,torch.zeros_like(visible))
        self.assertTrue(torch.isfinite(empty[0]).all() and torch.isfinite(empty[1]).all())
        self.assertTrue((empty[2]['cross_routing_score']==0).all())
        (own.square().sum()+cross.square().sum()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.lag.parameters()))

if __name__=='__main__':unittest.main(verbosity=2)
