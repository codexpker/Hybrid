"""Shared system-ablation metadata/init/L0 gates; synthetic only, root CPU job.

No N8000 execution, real source/target array, query/label or GPU entry exists.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
import numpy as np


def workspace():return Path(os.environ["HYBRID_V1_WORKSPACE"])
def package():return workspace()/"preparations/hybrid_v1_system_ablation_source_20261103_v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_configs():
    rows=json.loads((package()/"VARIANTS.json").read_text())["variants"]
    return [(row,json.loads((workspace()/row["train_config"]["path"]).read_text())) for row in rows]


def evidence(name,value):
    if os.environ.get("HYBRID_V1_SYSTEM_L0_OUTPUT"):
        root=Path(os.environ["HYBRID_V1_SYSTEM_L0_OUTPUT"]);root.mkdir(parents=True,exist_ok=True)
        with (root/(name+".json")).open("x") as file:json.dump(value,file,indent=2,allow_nan=False)


class ConfigurationOnly(unittest.TestCase):
    def test_fixed_plans_flags_recipes_and_scope(self):
        from hybrid_tsad.hybrid_v1 import HybridV1Config
        from hybrid_tsad.hybrid_v1.scoring import recipe
        base=json.loads((workspace()/"preparations/hybrid_v1_stage1_source_build_20261102_v1/source_N8000_seed2025.BOUND_PENDING.json").read_text())
        for row,cfg in load_configs():
            self.assertEqual(sha(workspace()/row["train_config"]["path"]),row["train_config"]["sha256"])
            self.assertEqual(sha(workspace()/row["CAL_config"]["path"]),row["CAL_config"]["sha256"])
            cal=json.loads((workspace()/row["CAL_config"]["path"]).read_text())
            self.assertIsNone(cal["checkpoint_sha256"]);self.assertIsNone(cal["train_receipt_sha256"])
            self.assertEqual(cal["model"],cfg["model"]);self.assertEqual(cal["active_components"],cfg["active_components"])
            actual={key:value for key,value in cfg["model"].items() if base["model"].get(key)!=value}
            self.assertEqual(actual,row["flags"]);HybridV1Config(**cfg["model"])
            for key in ("seed","history","horizon","steps","validate_every","lr","weight_decay","kind","manifest_sha256","plans_sha256","source_sha256","runtime_src"):
                self.assertEqual(cfg[key],base[key],(row["control_id"],key))
            active,weights=recipe(cfg["active_components"]);self.assertEqual(list(active),row["active_components"]);self.assertEqual(weights,row["score_weights"])
            self.assertEqual(row["SSL_consistency_active"],"r_tf" in active and "r_ms" in active)
            self.assertFalse(row["live_train_or_CAL_release"])


@unittest.skipUnless(importlib.util.find_spec("torch"),"TORCH_UNAVAILABLE_SYSTEM_INIT_L0_NOT_TESTED")
class ActualSourceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise PermissionError("Root CPU job must hide CUDA before Torch import")
        from hybrid_tsad.hybrid_v1 import cli
        closure=json.loads((package()/"SOURCE_INIT_CLOSURE.json").read_text())
        runtime=Path(load_configs()[0][1]["runtime_src"])/"hybrid_tsad/hybrid_v1"
        if Path(cli.__file__).resolve()!= (runtime/"cli.py").resolve():raise PermissionError("Import immutable source V4, not worktree")
        if {p.name:sha(p) for p in runtime.glob("*.py")}!=closure["source_v4_13"]:raise PermissionError("Original thirteen-file source closure changed")
        import torch
        torch.set_num_threads(1)

    def test_fresh_default_state_all_flags_match_original_initial(self):
        import gc,torch
        from hybrid_tsad.hybrid_v1.cli import construct
        reference=package()/"SOURCE_INIT_CLOSURE.json";asset=json.loads(reference.read_text())["reference_initial"]
        self.assertEqual(sha(asset["path"]),asset["sha256"])
        saved=torch.load(asset["path"],map_location="cpu",weights_only=True);reports=[]
        for row,cfg in load_configs():
            model=construct(dict(cfg,device="cpu"));state=model.state_dict();self.assertEqual(set(state),set(saved))
            tensors=[]
            for name,value in state.items():
                actual=value.detach().cpu().contiguous();expected=saved[name]
                same=actual.dtype==expected.dtype and actual.shape==expected.shape and torch.equal(actual,expected)
                self.assertTrue(same,(row["control_id"],name))
                tensors.append(dict(name=name,dtype=str(actual.dtype),shape=list(actual.shape),equal=bool(same),
                    sha256=hashlib.sha256(actual.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()))
            reports.append(dict(control=row["control_id"],state_tensors=len(tensors),stored_parameters=sum(p.numel() for p in model.parameters()),
                no_reference_loaded_into_model=True,tensors=tensors))
            del model,state;gc.collect()
        self.assertEqual(sha(asset["path"]),asset["sha256"])
        evidence("initial_state_comparison",dict(status="PASS_CPU_INIT_ONLY",variants=reports))

    def test_default_C3_real_fit_step_inactive_states_and_active_CAL(self):
        import gc,torch
        from hybrid_tsad.hybrid_v1.cli import construct
        from hybrid_tsad.hybrid_v1.data import Run
        from hybrid_tsad.hybrid_v1.training import fit,infer,calibrate_route
        from hybrid_tsad.hybrid_v1.scoring import apply_calibration
        wave=int(os.environ.get("HYBRID_V1_SYSTEM_L0_WAVE","1"))
        if wave not in (1,2):raise ValueError("One explicitly signed wave only")
        inactive={"tf_off":["tf_backbone.","tf_fusion_delta.","fusion.author.t_from_f.","fusion.author.f_from_t."],
            "causal_transformer_off":["temporal.encoder."],"self_lag_off":["lag.self_out."],"cross_lag_off":["lag.cross_out."],
            "all_lag_off":["lag."],"cross_zero_only":[],"ms_off":["ms_filter.","ms_decoder.","fusion.author.m_from_t."]}
        reports=[]
        for row,cfg in load_configs():
            if row["wave"]!=wave:continue
            name=row["control_id"];model=construct(dict(cfg,device="cpu"))
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            rng=np.random.default_rng(99180);runs=[]
            for role in ("FIT","SELECT","CAL"):
                values=rng.normal(size=(544,3)).astype(np.float32)
                runs.append(Run("SYNTHETIC",role,role,role,values,np.ones_like(values,bool)))
            point=lambda role:dict(domain="SYNTHETIC",entity=role,run=role,role=role,index=512,footprint=[0,544])
            model,trace=fit(model,[runs[0]],[runs[1]],history=512,horizon=32,steps=1,lr=cfg["lr"],kind=cfg["kind"],seed=2025,
                route="STAGE1",active_components=cfg["active_components"],weight_decay=.01,validate_every=1,
                sampling_plan=[point("FIT")],validation_plan=[point("SELECT")])
            self.assertEqual(trace["updates"],1);self.assertTrue(np.isfinite(trace["steps"][0]["loss"]))
            self.assertTrue(np.isfinite(trace["steps"][0]["gradient_norm"]));params=dict(model.named_parameters())
            states=model.state_dict();forbidden=inactive[name]+["fusion.concat_linear.","event_head."]
            for key,value in states.items():
                if any(key.startswith(prefix) for prefix in forbidden):
                    self.assertTrue(torch.equal(value,before[key]),(name,key,"inactive state changed"))
                    if key in params:self.assertIsNone(params[key].grad,(name,key,"inactive has gradient"))
            required=["temporal.embedding.","forecast_head."]
            if cfg["model"].get("temporal_enabled",True):required.append("temporal.encoder.")
            if cfg["model"].get("tf_enabled",True):required.extend(["tf_backbone.","tf_fusion_delta."])
            if cfg["model"].get("ms_enabled",True):required.extend(["ms_filter.","ms_decoder.","fusion.author.m_from_t."])
            if cfg["model"].get("self_lag_enabled",True):required.append("lag.self_out.")
            if cfg["model"].get("cross_lag_enabled",True):required.append("lag.cross_out.")
            blocks={}
            for prefix in required:
                subset=[(k,p) for k,p in params.items() if k.startswith(prefix)]
                self.assertTrue(subset,(name,prefix))
                finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for _,p in subset)
                nonzero=any(p.grad is not None and bool((p.grad!=0).any()) for _,p in subset)
                changed=any(not torch.equal(p.detach(),before[k]) for k,p in subset)
                self.assertTrue(finite and nonzero and changed,(name,prefix,finite,nonzero,changed))
                blocks[prefix]=dict(finite_grad=finite,nonzero_grad=nonzero,changed=changed)
            parts=trace["steps"][0]["parts"];expected=sum(parts[k[2:]] for k in cfg["active_components"])
            if row["SSL_consistency_active"]:expected+=.1*parts["consistency"]
            self.assertAlmostEqual(trace["steps"][0]["loss"],expected,delta=2e-5)
            cal_point=dict(point("CAL"),footprint=[0,513])
            records=infer(model,[runs[2]],history=512,horizon=32,seed=2025,active_components=cfg["active_components"],index_plan=[cal_point])
            calibration=calibrate_route(records,"STAGE1",active_components=cfg["active_components"])
            self.assertEqual(set(calibration["scales"]),set(cfg["active_components"]));self.assertEqual(calibration["weights"],row["score_weights"])
            self.assertTrue(apply_calibration(records[0],calibration)["valid"])
            # Unequal record counts must still give each synthetic source domain
            # half the CAL mass; this checks the actual accepted CAL kernel.
            a=dict(records[0],domain="A");b=dict(records[0],domain="B");c=dict(records[0],domain="B")
            for key in cfg["active_components"]:a[key]=1.;b[key]=3.;c[key]=5.
            balanced=calibrate_route([a,b,c],"STAGE1",active_components=cfg["active_components"])
            self.assertTrue(balanced["source_domain_equal_weight"])
            self.assertEqual(balanced["scales"],{key:1. for key in cfg["active_components"]})
            self.assertAlmostEqual(balanced["threshold"],np.log(6.),delta=1e-15)
            for key in set(("r_tf","r_ms","r_pred"))-set(cfg["active_components"]):self.assertFalse(np.isfinite(records[0][key]),(name,key,"inactive score must not be zero evidence"))
            reports.append(dict(control=name,synthetic_C=3,optimizer_steps=1,history=512,width=128,nominal_stored=sum(p.numel() for p in params.values()),
                grad_not_None_numel=sum(p.numel() for p in params.values() if p.grad is not None),nonzero_grad_numel=sum(p.numel() for p in params.values() if p.grad is not None and bool((p.grad!=0).any())),
                changed_numel=sum(p.numel() for k,p in params.items() if not torch.equal(p.detach(),before[k])),required_blocks=blocks,
                inactive_state_unchanged=True,active_components=cfg["active_components"],calibration_weights=calibration["weights"],
                FLOPs=None,FLOPs_reason="NOT_PROFILED; one synthetic C3 step is not nativeC1/C38 source cost"))
            del model,before,states,params;gc.collect()
        self.assertEqual(len(reports),4 if wave==1 else 3)
        evidence("C3_real_fit_CAL_wave%d"%wave,dict(status="PASS_SYNTHETIC_L0_ONLY",wave=wave,variants=reports))


if __name__=="__main__":unittest.main()
