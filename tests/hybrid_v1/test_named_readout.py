"""Metadata + synthetic CPU only. Execution requires a root-signed CPU job."""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import readout
from hybrid_tsad.hybrid_v1.scoring import recipe


def put(root,name,value):
    p=root/name;p.write_text(json.dumps(value,indent=2))
    return dict(path=str(p),sha256=readout.sha(p))


def fixture(root,route="SOURCE_CONTROL",control_id="none",gpu=3,active=None,**model_updates):
    active=list(recipe(active)[0]);sources=dict(timeradar_file="fixtureTF",left_root="fixtureMS",timesfm_root="fixtureT")
    model=dict(history=512,horizon=32,fusion="none",**model_updates)
    common=dict(model=model,sources=sources,history=512,horizon=32,seed=2025,active_components=active)
    checkpoint=root/"checkpoint.pt";checkpoint.write_bytes(b"SYNTHETIC_NOT_A_TORCH_CHECKPOINT")
    assets=dict(checkpoint=dict(path=str(checkpoint),sha256=readout.sha(checkpoint)))
    ckpt=assets["checkpoint"]["sha256"]
    parent="a"*64
    if route=="U_TCAL":
        p=root/"parent.pt";p.write_bytes(b"SYNTHETIC_PARENT_NOT_TORCH")
        assets["parent_source_checkpoint"]=dict(path=str(p),sha256=readout.sha(p));parent=readout.sha(p)
    cal=dict(scales={k:1. for k in active},weights=recipe(active)[1],active_components=active,
             threshold=.5,source_domain_equal_weight=route=="SOURCE_CONTROL",domains=["SMD","Sunspot"] if route=="SOURCE_CONTROL" else ["PSM"])
    assets["calibration"]=put(root,"cal.json",cal)
    train=dict(common,operation="HYBRID_V1_STAGE2_U" if route=="U_TCAL" else "HYBRID_V1_SOURCE_STAGE1")
    if route=="U_TCAL":train.update(mode="U",scope="public_target",source_checkpoint_sha256=parent,manifest_sha256="b"*64,plans_sha256="c"*64)
    assets["training_config"]=put(root,"train_config.json",train)
    trained=dict(phase="source_train",synthetic_non_scientific_results=False,files={"source.pt":ckpt,"config.json":assets["training_config"]["sha256"]})
    if route=="U_TCAL":
        trained=dict(phase="target_train",status="COMPLETED",mode="U",scope="public_target",target_fit=True,updates=2,
            adapted_checkpoint_sha256=ckpt,source_checkpoint_sha256=parent,manifest_sha256="b"*64,plans_sha256="c"*64)
    assets["training_receipt"]=put(root,"train_receipt.json",trained)
    calcfg=dict(common,operation="HYBRID_V1_SOURCE_STAGE1",checkpoint_sha256=ckpt,
                train_receipt_sha256=assets["training_receipt"]["sha256"])
    if route!="SOURCE_CONTROL":
        calcfg.update(operation="HYBRID_V1_STAGE2_U",scope="public_target",mode="U" if route=="U_TCAL" else "Z_TCAL",
            source_checkpoint_sha256=parent if route=="U_TCAL" else ckpt,manifest_sha256="b"*64,plans_sha256="c"*64,
            adapted_checkpoint_sha256=ckpt,adapted_receipt_sha256=assets["training_receipt"]["sha256"])
    assets["calibration_config"]=put(root,"cal_config.json",calcfg)
    calibrated=dict(phase="calibrate_source",synthetic_non_scientific_results=False,files={"source_calibration.json":assets["calibration"]["sha256"],"config.json":assets["calibration_config"]["sha256"]})
    if route!="SOURCE_CONTROL":
        calibrated=dict(status="COMPLETED",scope="public_target",phase="target_calibrate",mode=calcfg["mode"],updates=0,
            query_or_labels=False,
            target_fit=False,target_calibration_contact=True,calibration_route_kernel="U/source=False",calibrated_checkpoint_sha256=ckpt,
            source_checkpoint_sha256=parent if route=="U_TCAL" else ckpt,manifest_sha256="b"*64,plans_sha256="c"*64)
    assets["calibration_receipt"]=put(root,"cal_receipt.json",calibrated)
    seal=dict(status="ACCEPTED_NAMED_READOUT_ASSETS",approved_by="root",bindings=[dict(control_id=control_id,
        route=route,source_seed=2025,assets=copy.deepcopy(assets))])
    assets["asset_seal"]=put(root,"seal.json",seal)
    query=dict(domain="PSM",entity="PSM",run="test",role="QUERY",bounds=[0,87841],original_start=0,
        expected_shape=[87841,25],format="csv",has_header=True,columns=["feature_%d"%i for i in range(25)],
        sha256="566bc50117d13f58793d830c14975c55f627cab63953265d7d6ce2dbcf2b0f80",path="NEVER_OPEN_QUERY.csv")
    assets["query_manifest"]=put(root,"query_manifest.json",dict(runs=[query]))
    cfg=dict(common,source_seed=2025,inference_seed=2025,physical_gpu=gpu,device="cuda:0",expected_gpu_uuid="GPU-fixture"+str(gpu),
        control_id=control_id,route=route,operation=readout.NAMED_OPERATION,chunk_size=64,assets=assets,output=str(root/"out"),
        permissions=dict(query=True,labels=False,training=False,threshold_fit=False,target_fit=False),
        resources=dict(wall_seconds=21595,kill_grace_seconds=5,evaluation_GiB=30,automatic_retry=False))
    return cfg,seal


class NamedMetadata(unittest.TestCase):
    def test_three_routes_and_independent_gpu_for_same_seed(self):
        for route in ("SOURCE_CONTROL","U_TCAL","Z_TCAL"):
            for gpu in range(4):
                with self.subTest(route=route,gpu=gpu),tempfile.TemporaryDirectory() as d:
                    cfg,seal=fixture(Path(d),route,gpu=gpu)
                    result=readout.named_binding_metadata(cfg,seal)
                    self.assertEqual(result["target_calibration_contact"],route!="SOURCE_CONTROL")

    def test_old_full_binding_stays_seed_to_gpu_strict(self):
        cfg=dict(source_seed=2025,inference_seed=2025,physical_gpu=3,device="cuda:0",
            assets=dict(checkpoint=dict(sha256="a"),calibration=dict(sha256="b")))
        old=dict(status="ACCEPTED_LIMITED_SOURCE_CAL",assets=[dict(seed=2025,sha256=x) for x in ("a","b")])
        with self.assertRaises(PermissionError):readout.model_binding_metadata(cfg,old)

    def test_route_name_seed_and_asset_pair_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,seal=fixture(Path(d))
            for change in (dict(control_id="concat_linear"),dict(source_seed=2026),dict(inference_seed=2026),dict(route="U_TCAL")):
                with self.subTest(change=change),self.assertRaises(PermissionError):readout.named_binding_metadata(dict(cfg,**change),seal)
            wrong=copy.deepcopy(seal);wrong["bindings"][0]["assets"]["calibration"]["sha256"]="f"*64
            with self.assertRaises(PermissionError):readout.named_binding_metadata(cfg,wrong)

    def test_actual_model_config_and_disabled_branch_recipe(self):
        for switch,active in (("tf_enabled",["r_ms","r_pred"]),("ms_enabled",["r_tf","r_pred"])):
            with tempfile.TemporaryDirectory() as d:
                cfg,seal=fixture(Path(d),active=active,**{switch:False})
                readout.named_binding_metadata(cfg,seal)
                bad=copy.deepcopy(cfg);bad["model"]["fusion"]="concat_linear"
                with self.assertRaises(PermissionError):readout.named_binding_metadata(bad,seal)
                with self.assertRaises(PermissionError):readout.named_binding_metadata(dict(cfg,active_components=None),seal)

    def test_source_and_target_phase_or_cal_flags_cannot_masquerade(self):
        for route,key,change in (("SOURCE_CONTROL","training_receipt",dict(phase="target_train")),
                ("U_TCAL","training_receipt",dict(phase="source_train")),
                ("Z_TCAL","calibration_receipt",dict(mode="U")),
                ("U_TCAL","calibration",dict(source_domain_equal_weight=True)),
                ("Z_TCAL","calibration_receipt",dict(updates=1))):
            with self.subTest(route=route,key=key),tempfile.TemporaryDirectory() as d:
                cfg,seal=fixture(Path(d),route)
                p=Path(cfg["assets"][key]["path"]);value=json.loads(p.read_text());value.update(change);p.write_text(json.dumps(value))
                with self.assertRaises(PermissionError):readout.named_binding_metadata(cfg,seal)

    def test_root_guard_exact_seal_gpu_and_byte_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,seal=fixture(root)
            cfg["stream_source_sha256"]={p.name:readout.sha(p) for p in Path(readout.__file__).parent.glob("*.py")}
            config=put(root,"readout.json",cfg)
            binding={k:cfg[k] for k in ("control_id","route","source_seed","inference_seed","physical_gpu","expected_gpu_uuid")}
            release=dict(status="APPROVED",approved_by="root",operation=readout.NAMED_OPERATION,allowed_actions=["named_model_infer"],
                config_sha256=config["sha256"],source_sha256=cfg["stream_source_sha256"],external_source_sha256={},
                asset_seal_sha256=cfg["assets"]["asset_seal"]["sha256"],named_binding=binding)
            signed=put(root,"release.json",release)
            with patch("hybrid_tsad.hybrid_v1.provenance.OfficialSources.manifest",return_value=[]):
                admitted,resume=readout.admit_named(config["path"],signed["path"],signed["sha256"],cfg["output"])
                self.assertFalse(resume);self.assertEqual(admitted["physical_gpu"],3)
                release["named_binding"]["physical_gpu"]=0;signed=put(root,"release.json",release)
                with self.assertRaises(PermissionError):readout.admit_named(config["path"],signed["path"],signed["sha256"],cfg["output"])
            with self.assertRaises(PermissionError):readout.admit_named(config["path"],None,None,cfg["output"])
            self.assertFalse(Path(cfg["output"]).exists())


@unittest.skipUnless(importlib.util.find_spec("torch"),"TORCH_UNAVAILABLE_ACTUAL_WRITER_PARITY_NOT_TESTED")
class ActualWriterCPU(unittest.TestCase):
    def test_original_02844_vs_new_default_and_explicit_full_raw_rng_exact(self):
        import torch
        from hybrid_tsad.hybrid_v1 import HybridV1,HybridV1Config,OfficialSources
        from hybrid_tsad.hybrid_v1.data import Run
        from hybrid_tsad.hybrid_v1.chunked_bank import read_entries,_fingerprint,IDENTITY_KEYS
        workspace=Path(os.environ["HYBRID_V1_WORKSPACE"])
        reference=workspace/"preparations/hybrid_v1_named_readout_build_20261103_v1/readout_02844_before.py"
        self.assertEqual(readout.sha(reference),"02844b36fadb54538f57c258c629d6f40c646080a24e618fb551b08193f941ea")
        spec=importlib.util.spec_from_file_location("hybrid_tsad.hybrid_v1._named_old_02844",reference)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        torch.set_num_threads(1);torch.manual_seed(2025)
        config=HybridV1Config(history=128,horizon=32,local_window=20,width=16,temporal_patch=16,temporal_layers=1,
            heads=2,tf_hidden=8,tf_repr=16,tf_depth=1,tf_patch=5,lag_window=20,max_lag=8,lag_width=8,lag_top_k=2,
            history_mask_probability=.15,dropout=0.)
        model=HybridV1(config,OfficialSources.from_workspace(workspace)).cpu().eval()
        x=np.random.default_rng(72).normal(size=(164,3)).astype(np.float32);obs=np.ones(x.shape,bool)
        x[129,0]=-0.0
        x[132]=np.nan;obs[132]=False
        run=Run("SYNTHETIC","C3","raw","QUERY",x,obs,start=1000)
        cal=dict(scales={k:1. for k in recipe()[0]},weights=recipe()[1],active_components=list(recipe()[0]),threshold=.5)
        state=readout._state(model);initial=readout._rng();results=[]
        with tempfile.TemporaryDirectory() as d,patch("torch.cuda.synchronize",side_effect=AssertionError("CPU only")):
            for name,writer,kw in (("old",old.write_model_bank,{}),("new_default",readout.write_model_bank,{}),
                    ("new_explicit_full",readout.write_model_bank,dict(active_components=list(recipe()[0])))):
                readout._restore(initial);path=Path(d)/name
                result=writer(model,[run],cal,path,identity={k:"a"*64 for k in IDENTITY_KEYS},history=128,horizon=32,
                    seed=2025,chunk_size=8,capture_prefix=2,**kw)
                records=list(read_entries(path))
                results.append(([_fingerprint(r) for r in records],readout._rng_json(readout._rng()),
                    [_fingerprint(r) for r in result["prefix"]],readout._rng_json(result["prefix_rng"])))
                self.assertEqual([r["raw"]["index"] for r in records],list(range(1128,1164)))
                self.assertEqual(readout._state(model),state)
            self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])

    def test_optional_tf_ms_recipe_forwarded_without_changed_score_algorithm(self):
        import torch
        from hybrid_tsad.hybrid_v1.data import Run
        from hybrid_tsad.hybrid_v1.chunked_bank import read_records,IDENTITY_KEYS
        model=torch.nn.Linear(1,1).cpu();run=Run("SYNTHETIC","e","r","QUERY",np.ones((10,2),np.float32),np.ones((10,2),bool))
        for active in (["r_ms","r_pred"],["r_tf","r_pred"]):
            received=[]
            def iterator(*args,**kwargs):
                received.append(kwargs["active_components"])
                for t in range(6,10):
                    row=dict(domain=run.domain,entity=run.entity,run=run.run,index=t,available_index=t)
                    row.update({k:1. for k in active});yield row
            cal=dict(scales={k:1. for k in active},weights=recipe(active)[1],active_components=active,threshold=.5)
            with tempfile.TemporaryDirectory() as d,patch("hybrid_tsad.hybrid_v1.chunked_bank.iter_pointwise_infer",side_effect=iterator):
                readout.write_model_bank(model,[run],cal,d,identity={k:"b"*64 for k in IDENTITY_KEYS},history=6,horizon=32,
                    seed=2025,active_components=active)
                records=list(read_records(d));self.assertEqual(received,[active])
                self.assertTrue(all(r["valid"] and abs(r["score"]-np.log(2.))<1e-15 for r in records))


if __name__=="__main__":unittest.main()
