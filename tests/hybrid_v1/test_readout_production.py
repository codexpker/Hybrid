"""Only synthetic/metadata PSM orchestration fixtures; no real input/model."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import readout


def calibration():
    return dict(scales=dict(r_tf=1.,r_ms=1.,r_pred=1.),
        weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),
        active_components=["r_tf","r_ms","r_pred"],threshold=.5)


def points():
    return [dict(domain=d,entity=d,run=d,role="CAL",index=512+i)
            for d in ("SMD","Sunspot") for i in range(512)]


class ProductionGlue(unittest.TestCase):
    def test_no_release_before_assets_or_imports(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"config";p.write_text(json.dumps(dict(operation="HYBRID_V1_PSM_Z_MODEL_INFER")))
            with patch("hybrid_tsad.hybrid_v1.readout.sha",side_effect=AssertionError("No asset reads")):
                with self.assertRaises(PermissionError):readout.admit_production(p,None,None,Path(d)/"out")

    def test_pending_signed_object_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cfg=p/"cfg";cfg.write_text('{}');rel=p/"release"
            rel.write_text(json.dumps(dict(status="PENDING",approved_by="root")))
            with self.assertRaises(PermissionError):readout.admit_production(cfg,rel,readout.sha(rel),p/"out")
            self.assertFalse((p/"out").exists())

    def test_exact_common_source_cal_grid_not_three_fake_seeds(self):
        plans={s:dict(CAL=points()) for s in (2025,2026,2027)}
        self.assertEqual(len(readout.common_cal_plan(plans)),1024)
        bad=copy.deepcopy(plans);bad[2027]["CAL"][0]["index"]+=1
        with self.assertRaises(ValueError):readout.common_cal_plan(bad)
        bad=copy.deepcopy(plans)
        for p in bad.values():p["CAL"][0]["role"]="QUERY"
        with self.assertRaises(ValueError):readout.common_cal_plan(bad)

    def test_query_only_exact_columns_no_history_borrow_or_labels(self):
        entry=dict(domain="PSM",entity="PSM",run="test",role="QUERY",bounds=[0,87841],
            original_start=0,expected_shape=[87841,25],format="csv",has_header=True,
            columns=["feature_%d"%i for i in range(25)],
            sha256="566bc50117d13f58793d830c14975c55f627cab63953265d7d6ce2dbcf2b0f80")
        self.assertIs(readout.query_manifest_metadata(dict(runs=[entry])),entry)
        for key,value in (("bounds",[512,87841]),("columns",["timestamp_(min)"]),
                          ("role","FIT"),("label_path","forbidden")):
            with self.assertRaises(ValueError):readout.query_manifest_metadata(dict(runs=[dict(entry,**{key:value})]))
        self.assertEqual(87841-512,87329)
        self.assertEqual(len(list(range(87810,87841))),31)

    def test_seed_GPU_and_accepted_weight_CAL_pairing(self):
        cfg=dict(source_seed=2026,inference_seed=2026,physical_gpu=1,device="cuda:0",
                 assets=dict(checkpoint=dict(sha256="a"*64),calibration=dict(sha256="b"*64)))
        accepted=dict(status="ACCEPTED_LIMITED_SOURCE_CAL",assets=[dict(seed=2026,sha256=x*64) for x in ("a","b")])
        readout.model_binding_metadata(cfg,accepted)
        for key,value in (("source_seed",2028),("inference_seed",2025),("physical_gpu",0),("device","cuda:1")):
            with self.assertRaises(PermissionError):readout.model_binding_metadata(dict(cfg,**{key:value}),accepted)
        with self.assertRaises(PermissionError):
            readout.model_binding_metadata(cfg,dict(accepted,assets=[dict(seed=2025,sha256=x*64) for x in ("a","b")]))

    def test_model_interruption_resume_remaining_only_and_full_raw_parity(self):
        import torch
        from hybrid_tsad.hybrid_v1.chunked_bank import IDENTITY_KEYS,read_entries,_fingerprint
        model=torch.nn.Linear(1,1).cpu()  # Supplies device only, never forward.
        run=readout.synthetic_run(dict(rng_seed=84551,length=640,channels=25,
            distribution="standard_normal_float32",observed="all_true"))
        records=[dict(domain=run.domain,entity=run.entity,run=run.run,index=t,available_index=t,
                      r_tf=1.,r_ms=1.,r_pred=1.,raw_marker=[t,t+.25]) for t in range(512,640)]
        identity={k:"a"*64 for k in IDENTITY_KEYS};seen=[]
        def fake(*args,**kw):
            indices=list(kw["index_plan"]) if kw.get("index_plan") is not None else None
            selected=records if indices is None else [records[e["index"]-512] for e in indices]
            seen.append([r["index"] for r in selected]);yield from selected
        def fail(*args,**kw):
            yield from records[:70]
            raise RuntimeError("Synthetic interruption")
        with tempfile.TemporaryDirectory() as d,patch("torch.cuda.synchronize",side_effect=AssertionError("CPU")):
            a=Path(d)/"fresh";b=Path(d)/"resume"
            with patch("hybrid_tsad.hybrid_v1.chunked_bank.iter_pointwise_infer",side_effect=fake):
                readout.write_model_bank(model,[run],calibration(),a,identity=identity,history=512,horizon=32,seed=2025)
            with patch("hybrid_tsad.hybrid_v1.chunked_bank.iter_pointwise_infer",side_effect=fail):
                with self.assertRaises(RuntimeError):
                    readout.write_model_bank(model,[run],calibration(),b,identity=identity,history=512,horizon=32,seed=2025)
            self.assertEqual(json.loads((b/"manifest.json").read_text())["committed_records"],64)
            with self.assertRaises(PermissionError):
                readout.write_model_bank(model,[run],calibration(),b,identity=identity,history=512,horizon=32,seed=2025)
            with patch("hybrid_tsad.hybrid_v1.chunked_bank.iter_pointwise_infer",side_effect=fake):
                readout.write_model_bank(model,[run],calibration(),b,identity=identity,history=512,horizon=32,seed=2025,resume=True)
            self.assertEqual(seen[-1],list(range(576,640)))
            self.assertEqual([_fingerprint(x) for x in read_entries(a)],[_fingerprint(x) for x in read_entries(b)])
            with self.assertRaises(ValueError):
                readout.write_model_bank(model,[run],calibration(),b,identity=dict(identity,input_sha256="b"*64),
                                         history=512,horizon=32,seed=2025,resume=True)

    def test_baseline_bank_full_tail_same_direct_kernel(self):
        from hybrid_tsad.hybrid_v1.data import Run
        from hybrid_tsad.hybrid_v1.baselines import iter_baselines
        from hybrid_tsad.hybrid_v1.cli import score_records
        from hybrid_tsad.hybrid_v1.chunked_bank import IDENTITY_KEYS,read_records,_fingerprint
        x=np.arange(140,dtype=np.float32).reshape(70,2)
        run=Run("SYNTHETIC","E","R","QUERY",x,np.ones(x.shape,bool))
        cal=dict(scales=dict(r_pred=2.),weights=dict(r_pred=1.),active_components=["r_pred"],threshold=1.)
        with tempfile.TemporaryDirectory() as d:
            for method in ("persistence","history_mean512"):
                dest=Path(d)/method
                readout.write_baseline_bank([run],method,cal,dest,identity={k:"a"*64 for k in IDENTITY_KEYS},history=6)
                actual=list(read_records(dest));expected=score_records(list(iter_baselines([run],method=method,history=6)),cal)
                self.assertEqual([_fingerprint(r) for r in actual],[_fingerprint(r) for r in expected])
                self.assertEqual(actual[-1]["index"],69)
                with self.assertRaises(PermissionError):
                    readout.write_baseline_bank([run],method,cal,dest,identity={k:"a"*64 for k in IDENTITY_KEYS},history=6)


if __name__=="__main__":unittest.main()
