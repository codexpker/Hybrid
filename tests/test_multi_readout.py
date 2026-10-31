"""Synthetic multi-entity source identities and all-bank label-lock regression."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid_tsad.config import sha256
from hybrid_tsad.engine.readout_manifest import (MODES,METHODS,SEEDS,approve_readout_inputs,entity_profile,
                                               load_readout_operation,record,validate_whitelist)

HAS_TORCH=importlib.util.find_spec("torch") is not None


def source_fixture_catalog(root,profile,official,models):
    """Synthetic source-only evidence; no trained model or real data implied."""
    import numpy as np
    import torch
    from hybrid_tsad.engine.readout_controls import prepare_catalog,simple_forward
    from hybrid_tsad.engine.source import calibrate_source
    root.mkdir(parents=True)
    data=np.stack([np.sin(np.arange(120,dtype="float32")/20),np.cos(np.arange(120,dtype="float32")/19)],axis=1)
    psm=root/"source_psm.csv";np.savetxt(psm,np.c_[np.arange(120),data],delimiter=",",header="time,x,y",comments="")
    smd=root/"source_smd.txt";np.savetxt(smd,data,delimiter=",")
    rows=[dict(dataset=d,source=str(p),entity="synthetic_source",role=r,history=100,horizon=5,columns=[0,1],
               local_start=0,local_end_exclusive=120,forecast_origins=[100,105,110,115])
          for r in ("fit","cal") for d,p in (("PSM",psm),("SMD_SUBSET",smd))]
    manifest=root/"source_manifest.json";manifest.write_text(json.dumps({"records":rows}))
    row_file=root/"source_rows.json";row_file.write_text(json.dumps(rows))
    plan=root/"source_plan.npz";np.savez_compressed(plan,cal_record=np.array([2+(n//64)%2 for n in range(256)],"uint16"),
                                                 cal_origin=np.full(256,100,"uint32"),cal_channel=np.zeros(256,"uint8"))
    inputs=root/"source_inputs.json";inputs.write_text(json.dumps({"manifest":str(manifest),"hashes":{str(p):sha256(p) for p in (psm,smd,manifest)}}))
    definition=dict(official=official,models=[],source_assets={})
    for seed in SEEDS:
        definition["source_assets"][str(seed)]={k:record(p) for k,p in dict(source_inputs=inputs,source_manifest=manifest,rows=row_file,plan=plan).items()}
        for mode in MODES:
            x=torch.tensor(np.repeat(data[None,:100,:1],256,axis=0));y=np.repeat(data[None,100:105,:1],256,axis=0)
            o=simple_forward(x,"history_mean")
            bank=dict(prediction=o["prediction"].numpy()+.01*MODES.index(mode),target=y,history_scale=o["scale"].numpy(),valid_target=np.isfinite(y))
            folder=root/f"source_{seed}_{mode}";folder.mkdir()
            np.savez_compressed(folder/"cal_primary.npz",**bank)
            scores,cal=calibrate_source(bank);np.savez_compressed(folder/"cal_scores.npz",**scores)
            cal.update(history=100,role="source_cal",calibration_schema_version=2,score_space="history_normalized_abs_channelmean_div_source_median_v1",
                       operation_manifest_sha256="a"*64,source_bank_sha256=sha256(folder/"cal_primary.npz"))
            (folder/"calibration.json").write_text(json.dumps(cal))
            old=next(m for m in models if m["seed"]==seed and m["mode"]==mode)
            metadata=json.loads(Path(old["checkpoint_metadata"]["path"]).read_text())
            metadata.update(shared_plan_sha256=sha256(plan),source_manifest_sha256=sha256(manifest))
            (folder/"metadata.json").write_text(json.dumps(metadata))
            definition["models"].append(dict(seed=seed,mode=mode,checkpoint=old["checkpoint"],checkpoint_metadata=record(folder/"metadata.json"),
                calibration=record(folder/"calibration.json"),cal_scores=record(folder/"cal_scores.npz"),cal_bank=record(folder/"cal_primary.npz")))
    path=root/"definition.json";path.write_text(json.dumps(definition))
    prepare_catalog(path,root/"prepared",profile)
    return root/"prepared/catalog.json"


class ReadoutApprovalTests(unittest.TestCase):
    def test_approval_binds_whitelist_scope_and_forbids_label_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ("contract","approval","catalog","query_manifest","whitelist"):
                (root/name).write_text("{}")
            cfg=dict(engine_schema_version=4,operation="readout-infer",contract=record(root/"contract"),approval=record(root/"approval"),
                inputs={name:record(root/name) for name in ("catalog","query_manifest","whitelist")},
                parameters=dict(profile={"name":"SYNTHETIC"},scope="shard",device="cpu"),
                output=str(root/"new_output"),score_space="fixed_native_late_canonical_source_controls_v1")
            path=root/"operation.json";path.write_text(json.dumps(cfg))
            _,_,_,_,digest=load_readout_operation(path)
            (root/"approval").write_text(json.dumps(dict(status="APPROVED",contract_sha256=cfg["contract"]["sha256"],operation_manifest_sha256=digest)))
            cfg["approval"]=record(root/"approval");path.write_text(json.dumps(cfg))
            parsed,paths,out,_,_=load_readout_operation(path)
            approve_readout_inputs(parsed,paths,out,path)
            cfg["parameters"]["scope"]="l0";path.write_text(json.dumps(cfg))
            parsed,paths,out,_,_=load_readout_operation(path)
            with self.assertRaisesRegex(ValueError,"exact operation"):
                approve_readout_inputs(parsed,paths,out,path)
            cfg["inputs"]["labels"]={"path":"must_not_be_opened.npy","sha256":"0"*64}
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,"no label inputs"):
                load_readout_operation(path)


def synthetic_fixture(root):
    import numpy as np
    profile=dict(name="SYNTHETIC",faults=[1,2],runs=[31,32],points=120,channels=52,event_start=110,sample_interval_minutes=3)
    official=root/"official.py";official.write_text("synthetic placeholder; no upstream source")
    models=[];controls=[]
    for seed in SEEDS:
        for mode in MODES:
            checkpoint=root/f"{mode}_{seed}.pt";checkpoint.write_bytes(f"synthetic {mode} {seed}".encode())
            metadata=root/f"{mode}_{seed}.json"
            metadata.write_text(json.dumps(dict(checkpoint_sha256=sha256(checkpoint),history=100,horizon=5,
                mode=mode,copies=2,temporal_depth=4,seed=seed,excluded_target="TEP",source_domains=["PSM","SMD_SUBSET"],step=10000,operation_manifest_sha256="a"*64)))
            calibration=root/f"{mode}_{seed}_cal.json"
            calibration.write_text(json.dumps(dict(calibration_schema_version=2,role="source_cal",history=100,
                score_space="history_normalized_abs_channelmean_div_source_median_v1",source_bank_sha256="a"*64,
                operation_manifest_sha256="a"*64,normal_error_scale=1.+seed*.1,threshold99=1.)))
            models.append(dict(seed=seed,mode=mode,checkpoint=record(checkpoint),checkpoint_metadata=record(metadata),calibration=record(calibration)))
        for method in METHODS:
            controls.append(dict(seed=seed,method=method,scale=1. if method=="late" else 1.+seed*.1,threshold=1.,kind="synthetic"))
    catalog=source_fixture_catalog(root/"synthetic_source",profile,record(official),models)
    records=[]
    for fault in profile["faults"]:
        for run in profile["runs"]:
            entity=f"fault{fault:02}_run{run:03}"
            data=(np.sin(np.arange(120,dtype="float32")[:,None]/20)+np.arange(52,dtype="float32")[None,:]/100).astype("float32")
            array=root/(entity+".npy");np.save(array,data)
            labels=root/(entity+"_labels.npy");np.save(labels,(np.arange(120)>=110).astype("uint8"))
            records.append(dict(dataset="SYNTHETIC",entity=entity,role="query",fault_id=fault,run_id=run,
                history=100,horizon=5,guard=0,local_start=0,local_end_exclusive=120,channels=52,origins=[100,105,110,115],
                array_path=str(array),label_path=str(labels),array_sha256_from_frozen_parent=sha256(array),
                label_sha256_from_frozen_parent=sha256(labels),event_start_0based=110,event_end_exclusive=120))
    query=root/"query.json";query.write_text(json.dumps(dict(dataset="SYNTHETIC",records=records)))
    cells=[dict(seed=s,mode=m,entity=q["entity"],origins=q["origins"]) for s in SEEDS for m in MODES for q in records]
    whitelist=root/"whitelist.json";whitelist.write_text(json.dumps(cells))
    return profile,catalog,query,whitelist


@unittest.skipUnless(HAS_TORCH,"existing torch required; synthetic CPU only")
class MultiReadoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.profile,self.catalog,self.query,self.whitelist=synthetic_fixture(self.root)

    @staticmethod
    def factory(official,*,mode,history,checkpoint,checkpoint_metadata,device):
        import torch
        class Fixture(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1));self.history=history
            def forward(self,x):
                return {"prediction":x[:,-1:]+torch.rand((len(x),5,x.shape[-1]))*.01,
                        "scale":(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt()}
        return Fixture().eval()

    def infer(self,whitelist=None,name="shard",scope="shard"):
        from hybrid_tsad.engine.multi_readout import infer_shard
        return infer_shard(self.catalog,self.query,whitelist or self.whitelist,self.root/name,
                           profile=self.profile,scope=scope,predictor_factory=self.factory)

    def shard_list(self,name="shard"):
        path=self.root/(name+"_list.json");path.write_text(json.dumps([record(self.root/name/"shard_manifest.json")]))
        return path

    def test_complete_group_seals_36_banks_and72_scores_without_labels(self):
        import numpy as np
        from hybrid_tsad.engine.multi_readout import seal_scores,verify_label_unlock
        real_load=np.load
        def locked_load(path,*a,**kw):
            if "_labels" in str(path):raise AssertionError("labels opened before explicit evaluation")
            return real_load(path,*a,**kw)
        with patch.object(np,"load",locked_load):
            status=self.infer();self.assertFalse(status["complete_group"])
            seal=seal_scores(self.shard_list(),self.catalog,self.query,self.root/"seal",profile=self.profile)
            self.assertEqual(seal["counts"],{"native":36,"scores":72})
            result,_=verify_label_unlock(self.root/"seal/group_seal.json",self.catalog,self.query,self.profile)
            self.assertFalse(result["labels_opened"])
            self.assertIsNone(result["overall_Aff_F1"])
        score=next(iter(seal["scores"].values()))["file"]["path"]
        with Path(score).open("ab") as f:f.write(b"tamper")
        with self.assertRaisesRegex(ValueError,"identity mismatch"):
            verify_label_unlock(self.root/"seal/group_seal.json",self.catalog,self.query,self.profile)

    def test_partial_or_l0_success_never_unlocks_group(self):
        from hybrid_tsad.engine.multi_readout import seal_scores
        cells=json.loads(self.whitelist.read_text());self.whitelist.write_text(json.dumps(cells[:-1]))
        status=self.infer();self.assertFalse(status["complete_group"])
        with self.assertRaisesRegex(ValueError,"All-bank label lock"):
            seal_scores(self.shard_list(),self.catalog,self.query,self.root/"partial_seal",profile=self.profile)
        first=min(q["entity"] for q in json.loads(self.query.read_text())["records"])
        l0=[dict(seed=0,mode=m,entity=first,origins=[100,105]) for m in MODES]
        path=self.root/"l0.json";path.write_text(json.dumps(l0))
        status=self.infer(path,name="l0",scope="l0");self.assertEqual(status["status"],"L0_ONLY_LABELS_LOCKED")
        with self.assertRaisesRegex(ValueError,"L0/foreign"):
            seal_scores(self.shard_list("l0"),self.catalog,self.query,self.root/"l0_seal",profile=self.profile)

    def test_exact_whitelist_and_duplicate_shards_are_rejected(self):
        from hybrid_tsad.engine.multi_readout import seal_scores
        entities=entity_profile(json.loads(self.query.read_text()),self.profile)
        cell=json.loads(self.whitelist.read_text())[0]
        with self.assertRaises(ValueError):validate_whitelist([cell,cell],entities,"shard")
        with self.assertRaises(ValueError):validate_whitelist([{**cell,"origins":[100,105]}],entities,"shard")
        self.infer();listing=self.shard_list();ref=json.loads(listing.read_text())[0]
        listing.write_text(json.dumps([ref,ref]))
        with self.assertRaisesRegex(ValueError,"Duplicate bank"):
            seal_scores(listing,self.catalog,self.query,self.root/"duplicate",profile=self.profile)

    def test_catalog_unit_changes_fail_before_infer_seal_and_labelgrant(self):
        from hybrid_tsad.engine.multi_readout import infer_shard,load_catalog,seal_scores,verify_label_unlock
        self.infer();listing=self.shard_list()
        good=seal_scores(listing,self.catalog,self.query,self.root/"unit_good_seal",profile=self.profile)
        baseline=json.loads(self.catalog.read_text())
        for method,field,factor,reason in (("history_mean","scale",11,"Simple catalog units"),
                ("history_mean","threshold",3,"Simple catalog units"),("late","threshold",3,"Late catalog threshold"),
                ("temporal","scale",2,"Native catalog units")):
            changed=json.loads(json.dumps(baseline))
            control=next(c for c in changed["controls"] if c["seed"]==0 and c["method"]==method)
            control[field]*=factor
            path=self.root/(method+field+"_changed_catalog.json");path.write_text(json.dumps(changed))
            with self.subTest(method=method,field=field):
                with self.assertRaisesRegex(ValueError,reason):load_catalog(path,require_full=True)
                with self.assertRaisesRegex(ValueError,reason):
                    infer_shard(path,self.query,self.whitelist,self.root/(method+field+"_infer"),profile=self.profile,scope="shard",predictor_factory=self.factory)
                with self.assertRaisesRegex(ValueError,reason):
                    seal_scores(listing,path,self.query,self.root/(method+field+"_seal"),profile=self.profile)
                forged=json.loads(json.dumps(good));forged["catalog"]=record(path)
                seal=self.root/(method+field+"_forged_seal.json");seal.write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError,reason):verify_label_unlock(seal,path,self.query,self.profile)

    def test_resource_timing_fields_do_not_change_arrays_or_rng(self):
        import itertools
        import numpy as np
        cells=json.loads(self.whitelist.read_text());path=self.root/"one_cell.json";path.write_text(json.dumps(cells[:1]))
        first=self.infer(path,name="timing_original")
        with patch("hybrid_tsad.engine.multi_readout.time.perf_counter",side_effect=itertools.count(1.,.001).__next__):
            second=self.infer(path,name="timing_counter")
        key=next(iter(first["banks"]))
        for kind in ("bank","rng"):
            with np.load(first["banks"][key][kind]["path"]) as a,np.load(second["banks"][key][kind]["path"]) as b:
                for name in a.files:self.assertTrue(np.array_equal(a[name],b[name]),(kind,name))
        metadata=json.loads(Path(first["banks"][key]["metadata"]["path"]).read_text())
        self.assertGreaterEqual(metadata["seconds"],0)
        self.assertIsNone(metadata["peak_allocated_bytes"])
        self.assertEqual(metadata["forward_timing_samples"],4)
        self.assertGreaterEqual(metadata["forward_p50_seconds"],0)
        self.assertGreaterEqual(metadata["forward_p95_seconds"],metadata["forward_p50_seconds"])
        self.assertGreaterEqual(first["banks"][key]["whole_entity_seconds"],metadata["seconds"])

    def test_cold_cuda_setup_precedes_peak_reset_and_model_without_forward_warmup(self):
        import torch
        from hybrid_tsad.engine.multi_readout import infer_shard
        cells=json.loads(self.whitelist.read_text());path=self.root/"cuda_one_cell.json";path.write_text(json.dumps(cells[:1]))
        events=[]
        def reset(device):
            self.assertEqual(events,[("set_device","cuda:0"),("init",None)])
            events.append(("reset",str(device)))
        def factory(*args,**kwargs):
            self.assertEqual(events[:3],[("set_device","cuda:0"),("init",None),("reset","cuda:0")])
            events.append(("model",None))
            # Only a CPU fixture; no CUDA device or random warmup is used.
            return self.factory(*args,**kwargs)
        with patch.object(torch.cuda,"set_device",side_effect=lambda d:events.append(("set_device",str(d)))), \
             patch.object(torch.cuda,"init",side_effect=lambda:events.append(("init",None))), \
             patch.object(torch.cuda,"reset_peak_memory_stats",side_effect=reset), \
             patch.object(torch.cuda,"max_memory_allocated",return_value=42), \
             patch.object(torch.cuda,"manual_seed_all",return_value=None):
            infer_shard(self.catalog,self.query,path,self.root/"mock_cold_cuda",profile=self.profile,scope="shard",device="cuda:0",predictor_factory=factory)
        self.assertEqual(events,[("set_device","cuda:0"),("init",None),("reset","cuda:0"),("model",None)])
        with patch.object(torch.cuda,"set_device",side_effect=AssertionError("CPU branch selected CUDA")), \
             patch.object(torch.cuda,"init",side_effect=AssertionError("CPU branch initialized CUDA")), \
             patch.object(torch.cuda,"reset_peak_memory_stats",side_effect=AssertionError("CPU branch reset CUDA")), \
             patch.object(torch.cuda,"manual_seed_all",return_value=None):
            self.infer(path,name="cpu_no_cuda_setup")

    def test_physical_target_origin_and_history_scale_forgery_refused_before_labels(self):
        import numpy as np
        from hybrid_tsad.engine.multi_readout import seal_scores,verify_label_unlock
        self.infer();listing=self.shard_list()
        good=seal_scores(listing,self.catalog,self.query,self.root/"good_seal",profile=self.profile)
        shard_path=self.root/"shard/shard_manifest.json"
        original_shard=json.loads(shard_path.read_text())
        key=next(iter(original_shard["banks"]));descriptor=original_shard["banks"][key]
        bank_path=Path(descriptor["bank"]["path"]);meta_path=Path(descriptor["metadata"]["path"])
        with np.load(bank_path) as bank:original={k:bank[k].copy() for k in bank.files}
        original_meta=json.loads(meta_path.read_text())
        for attack,reason in (("target","physical target"),("origin","time index/origin"),("scale","history scale")):
            bank={k:v.copy() for k,v in original.items()}
            if attack=="target":bank["target"]+=7
            elif attack=="origin":bank["forecast_origin"][0]+=5
            else:bank["history_scale_by_block"]*=2
            scales=np.repeat(bank["history_scale_by_block"],5,axis=1).reshape(bank["target"].shape)
            bank["normalized_channel_residual"]=np.abs(bank["prediction"]-bank["target"])/scales
            bank["raw_score"]=bank["normalized_channel_residual"].mean(-1)
            bank["standardized_score"]=bank["raw_score"]/original_meta["source_scale"]
            np.savez_compressed(bank_path,**bank)
            metadata={**original_meta,"bank_sha256":sha256(bank_path)};meta_path.write_text(json.dumps(metadata))
            shard=json.loads(json.dumps(original_shard));shard["banks"][key].update(bank=record(bank_path),metadata=record(meta_path))
            shard_path.write_text(json.dumps(shard));listing.write_text(json.dumps([record(shard_path)]))
            with self.subTest(attack=attack),self.assertRaisesRegex(ValueError,reason):
                seal_scores(listing,self.catalog,self.query,self.root/("rejected_"+attack),profile=self.profile)
            forged=json.loads(json.dumps(good));forged["native_banks"][key].update(bank=record(bank_path),metadata=record(meta_path))
            path=self.root/("forged_seal_"+attack+".json");path.write_text(json.dumps(forged))
            with self.subTest(unlock_attack=attack),self.assertRaisesRegex(ValueError,reason):
                verify_label_unlock(path,self.catalog,self.query,self.profile)

    def test_source_control_catalog_uses_fixed_plan_and_recomputed_calibration(self):
        import numpy as np
        import torch
        from hybrid_tsad.engine.readout_controls import prepare_catalog,simple_forward
        from hybrid_tsad.engine.source import calibrate_source
        data=np.stack([np.sin(np.arange(120,dtype="float32")/20),np.cos(np.arange(120,dtype="float32")/19)],axis=1)
        psm=self.root/"source_psm.csv";np.savetxt(psm,np.c_[np.arange(120),data],delimiter=",",header="time,x,y",comments="")
        smd=self.root/"source_smd.txt";np.savetxt(smd,data,delimiter=",")
        rows=[dict(dataset=d,source=str(p),entity="synthetic_source",role=r,history=100,horizon=5,columns=[0,1],
                   local_start=0,local_end_exclusive=120,forecast_origins=[100,105,110,115])
              for r in ("fit","cal") for d,p in (("PSM",psm),("SMD_SUBSET",smd))]
        manifest=self.root/"source_manifest.json";manifest.write_text(json.dumps({"records":rows}))
        row_file=self.root/"source_rows.json";row_file.write_text(json.dumps(rows))
        plan=self.root/"source_plan.npz"
        np.savez_compressed(plan,cal_record=np.array([2+(n//64)%2 for n in range(256)],"uint16"),
                            cal_origin=np.full(256,100,"uint32"),cal_channel=np.zeros(256,"uint8"))
        inputs=self.root/"source_inputs.json";inputs.write_text(json.dumps({"manifest":str(manifest),"hashes":{str(p):sha256(p) for p in (psm,smd,manifest)}}))
        definition={"official":json.loads(self.catalog.read_text())["official"],"models":[],"source_assets":{}}
        for seed in SEEDS:
            definition["source_assets"][str(seed)]={k:record(p) for k,p in dict(source_inputs=inputs,source_manifest=manifest,rows=row_file,plan=plan).items()}
            for mode in MODES:
                x=torch.tensor(np.repeat(data[None,:100,:1],256,axis=0))
                y=np.repeat(data[None,100:105,:1],256,axis=0)
                o=simple_forward(x,"history_mean")
                bank=dict(prediction=o["prediction"].numpy()+.01*MODES.index(mode),target=y,
                          history_scale=o["scale"].numpy(),valid_target=np.isfinite(y))
                folder=self.root/f"source_{seed}_{mode}";folder.mkdir()
                np.savez_compressed(folder/"cal_primary.npz",**bank)
                scores,cal=calibrate_source(bank);np.savez_compressed(folder/"cal_scores.npz",**scores)
                cal.update(history=100,role="source_cal",calibration_schema_version=2,
                           score_space="history_normalized_abs_channelmean_div_source_median_v1",operation_manifest_sha256="a"*64,
                           source_bank_sha256=sha256(folder/"cal_primary.npz"))
                (folder/"calibration.json").write_text(json.dumps(cal))
                old=next(m for m in json.loads(self.catalog.read_text())["models"] if m["seed"]==seed and m["mode"]==mode)
                metadata=json.loads(Path(old["checkpoint_metadata"]["path"]).read_text())
                metadata.update(shared_plan_sha256=sha256(plan),source_manifest_sha256=sha256(manifest))
                (folder/"metadata.json").write_text(json.dumps(metadata))
                definition["models"].append(dict(seed=seed,mode=mode,checkpoint=old["checkpoint"],checkpoint_metadata=record(folder/"metadata.json"),
                    calibration=record(folder/"calibration.json"),cal_scores=record(folder/"cal_scores.npz"),cal_bank=record(folder/"cal_primary.npz")))
        path=self.root/"source_definition.json";path.write_text(json.dumps(definition))
        catalog=prepare_catalog(path,self.root/"prepared_catalog",self.profile)
        self.assertEqual(len(catalog["models"]),9);self.assertEqual(len(catalog["controls"]),18)
        self.assertFalse(catalog["labels_opened"])
        # Original independent counterexample: only rebind rows to another CSV;
        # native weights/cal, source manifest and source plan remain unchanged.
        rogue=self.root/"rogue_source.csv";np.savetxt(rogue,np.c_[np.arange(120),data+50],delimiter=",",header="time,x,y",comments="")
        rogue_rows=[{**r,"source":str(rogue)} if r["dataset"]=="PSM" else r for r in rows]
        rogue_row_file=self.root/"rogue_rows.json";rogue_row_file.write_text(json.dumps(rogue_rows))
        changed=json.loads(json.dumps(definition));changed["source_assets"]["0"]["rows"]=record(rogue_row_file)
        bad=self.root/"rogue_definition.json";bad.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"rows differ"):
            prepare_catalog(bad,self.root/"rejected_rogue",self.profile)
        nohash=json.loads(inputs.read_text());nohash["hashes"].pop(str(psm))
        missing=self.root/"missing_hash.json";missing.write_text(json.dumps(nohash))
        changed=json.loads(json.dumps(definition));changed["source_assets"]["0"]["source_inputs"]=record(missing)
        bad.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"missing frozen input hashes"):
            prepare_catalog(bad,self.root/"rejected_missing_hash",self.profile)
        for attack in ("target","mask"):
            changed=json.loads(json.dumps(definition))
            for native in (m for m in changed["models"] if m["seed"]==0):
                with np.load(native["cal_bank"]["path"]) as original:
                    bank={k:original[k].copy() for k in original.files}
                if attack=="target":bank["target"]+=1
                else:bank["valid_target"][0,0,0]=False
                folder=self.root/(attack+native["mode"]);folder.mkdir()
                np.savez_compressed(folder/"bank.npz",**bank)
                scores,values=calibrate_source(bank);np.savez_compressed(folder/"scores.npz",**scores)
                cal=json.loads(Path(native["calibration"]["path"]).read_text());cal.update(values,source_bank_sha256=sha256(folder/"bank.npz"))
                (folder/"cal.json").write_text(json.dumps(cal))
                native.update(cal_bank=record(folder/"bank.npz"),cal_scores=record(folder/"scores.npz"),calibration=record(folder/"cal.json"))
            bad.write_text(json.dumps(changed))
            with self.subTest(attack=attack),self.assertRaisesRegex(ValueError,"actual frozen source plan"):
                prepare_catalog(bad,self.root/("rejected_"+attack),self.profile)
        for attack,reason in (("footprint","trace role/channel/footprint"),("domain","domain order")):
            with np.load(plan) as original:trace={k:original[k].copy() for k in original.files}
            if attack=="footprint":trace["cal_origin"][0]=99
            else:trace["cal_record"][:]=2
            wrong_plan=self.root/(attack+"_plan.npz");np.savez_compressed(wrong_plan,**trace)
            changed=json.loads(json.dumps(definition));changed["source_assets"]["0"]["plan"]=record(wrong_plan)
            for native in (m for m in changed["models"] if m["seed"]==0):
                meta=json.loads(Path(native["checkpoint_metadata"]["path"]).read_text());meta["shared_plan_sha256"]=sha256(wrong_plan)
                path_meta=self.root/(attack+native["mode"]+"_metadata.json");path_meta.write_text(json.dumps(meta))
                native["checkpoint_metadata"]=record(path_meta)
            bad.write_text(json.dumps(changed))
            with self.subTest(trace_attack=attack),self.assertRaisesRegex(ValueError,reason):
                prepare_catalog(bad,self.root/("rejected_"+attack),self.profile)


if __name__=="__main__":unittest.main()
