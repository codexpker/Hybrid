"""Synthetic macro/run-cluster estimand and external official callback tests."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid_tsad.engine.official_affiliation import OFFICIAL_FILES,OfficialAffiliationBlocked,load_official_affiliation
from hybrid_tsad.engine.readout_manifest import METHODS,SEEDS,record

HAS_NUMPY=importlib.util.find_spec("numpy") is not None
HAS_TORCH=importlib.util.find_spec("torch") is not None
AFF_DIR=os.environ.get("HYBRID_OFFICIAL_AFF_DIR")


class OfficialDependencyTests(unittest.TestCase):
    def test_missing_official_dependency_is_explicit_block_not_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"identity.json"
            path.write_text(json.dumps(dict(schema_version=1,directory="missing",files=OFFICIAL_FILES,
                license=dict(status="UNVERIFIED_EXISTING_LOCAL_DEPENDENCY",source="existing frozen source audit",redistribute=False))))
            with self.assertRaisesRegex(OfficialAffiliationBlocked,"BLOCKED"):
                load_official_affiliation(path)

    def test_wrong_official_version_or_unrecorded_license_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"identity.json"
            value=dict(schema_version=1,directory="missing",files={**OFFICIAL_FILES,"metrics.py":"0"*64},
                       license=dict(status="UNVERIFIED_EXISTING_LOCAL_DEPENDENCY",source="frozen audit",redistribute=False))
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(OfficialAffiliationBlocked,"six-file identity"):
                load_official_affiliation(path)
            value["files"]=OFFICIAL_FILES;value["license"]={};path.write_text(json.dumps(value))
            with self.assertRaisesRegex(OfficialAffiliationBlocked,"license provenance"):
                load_official_affiliation(path)


@unittest.skipUnless(HAS_NUMPY,"existing NumPy required")
class RunClusterTests(unittest.TestCase):
    def rows(self):
        return [dict(method=m,seed=s,fault_id=f,run_id=r,
                     AP=.3+METHODS.index(m)*.01+f*.001+r*.0001+s*.00001+(f*r*.00001 if m=="hybrid" else 0),
                     ROC=.5+METHODS.index(m)*.01,positive_fraction=800/860)
                for m in METHODS for s in SEEDS for f in range(1,21) for r in range(31,51)]
    def test_macro_first_seed_then_equal_fault_run_and_shared_2000_run_draws(self):
        import numpy as np
        from hybrid_tsad.engine.run_statistics import COMPARATORS,run_cluster_statistics
        rows=self.rows();profile=dict(name="TEP100",faults=list(range(1,21)),runs=list(range(31,51)))
        result=run_cluster_statistics(rows,profile)
        macro=next(r for r in result["macro"] if r["method"]=="hybrid" and r["metric"]=="AP")
        expected=sum(r["AP"] for r in rows if r["method"]=="hybrid")/(3*20*20)
        self.assertAlmostEqual(macro["mean"],expected,places=14)
        lookup={(r["method"],r["seed"],r["fault_id"],r["run_id"]):r["AP"] for r in rows}
        paired=np.array([[sum(lookup["hybrid",s,f,r]-lookup[c,s,f,r] for f in range(1,21) for s in SEEDS)/(20*3)
                          for c in COMPARATORS] for r in range(31,51)])
        draws=np.random.default_rng(99174).integers(0,20,size=(2000,20))
        self.assertTrue(np.array_equal(result["bootstrap_arrays"]["draws"],draws))
        expected_boot=np.stack([paired[draw].mean(0) for draw in draws])
        self.assertTrue(np.allclose(result["bootstrap_arrays"]["AP_difference"],expected_boot,atol=1e-15,rtol=0))
        self.assertIsNone(result["overall_Aff_F1"])
        self.assertEqual(len(result["per_fault"]),6*2*20)
        self.assertEqual(len(result["per_run"]),6*2*20)
        worst=next(row for row in result["worst"] if row["method"]=="hybrid" and row["metric"]=="AP")
        self.assertEqual((worst["worst_entity_fault_id"],worst["worst_entity_run_id"],worst["worst_seed_id"]),(1,31,0))
        self.assertAlmostEqual(macro["sample_std"],.00001,places=14)
    def test_missing_cells_refused_and_invalid_run_has_common_na_without_redraw(self):
        import numpy as np
        from hybrid_tsad.engine.run_statistics import run_cluster_statistics
        rows=self.rows();profile=dict(name="TEP100",faults=list(range(1,21)),runs=list(range(31,51)))
        with self.assertRaisesRegex(ValueError,"Missing"):
            run_cluster_statistics(rows[:-1],profile)
        bad=next(r for r in rows if r["method"]=="persistence" and r["seed"]==2 and r["fault_id"]==1 and r["run_id"]==31)
        bad["AP"]=float("nan")
        result=run_cluster_statistics(rows,profile);arrays=result["bootstrap_arrays"]
        invalid=(arrays["draws"]==0).any(1)
        self.assertTrue(np.isnan(arrays["AP_difference"][invalid]).all())
        self.assertTrue(np.isfinite(arrays["AP_difference"][~invalid]).all())
        self.assertEqual(arrays["draws"].shape,(2000,20));self.assertEqual(len(result["failures"]),1)
        self.assertEqual(result["status"],"INCOMPLETE")
        bad["AP"]=2.
        with self.assertRaisesRegex(ValueError,"outside"):
            run_cluster_statistics(rows,profile)
        with self.assertRaisesRegex(ValueError,"unique nonempty"):
            run_cluster_statistics(rows,{**profile,"runs":[31,31]})


@unittest.skipUnless(HAS_TORCH and AFF_DIR,"existing torch and external frozen official Affiliation required")
class MultiEntityEvaluationTests(unittest.TestCase):
    def test_complete_synthetic_group_uses_official_per_entity_and_run_clusters(self):
        import numpy as np
        from test_multi_readout import MultiReadoutTests,synthetic_fixture
        from hybrid_tsad.engine.multi_readout import infer_shard,seal_scores
        from hybrid_tsad.engine.multi_evaluation import evaluate_group
        from hybrid_tsad.engine.multi_readout import run_readout,SCORE_SPACE
        from hybrid_tsad.engine.readout_manifest import load_readout_operation
        import csv
        with tempfile.TemporaryDirectory() as temp:
            archive=os.environ.get("HYBRID_SYNTHETIC_EVALUATION_ARCHIVE")
            root=Path(archive) if archive else Path(temp)
            if archive:root.mkdir(parents=True,exist_ok=False)
            profile,catalog,query,whitelist=synthetic_fixture(root)
            infer_shard(catalog,query,whitelist,root/"shard",profile=profile,scope="shard",predictor_factory=MultiReadoutTests.factory)
            shards=root/"shards.json";shards.write_text(json.dumps([record(root/"shard/shard_manifest.json")]))
            seal_scores(shards,catalog,query,root/"seal",profile=profile)
            identity=root/"affiliation.json";identity.write_text(json.dumps(dict(schema_version=1,directory=AFF_DIR,files=OFFICIAL_FILES,
                license=dict(status="UNVERIFIED_EXISTING_LOCAL_DEPENDENCY",source="HY10K-PSM-READOUT/aff_source.json; no redistribution",redistribute=False))))
            callback,provenance=load_official_affiliation(identity)
            self.assertEqual(provenance["files"],OFFICIAL_FILES)
            contract=root/"contract.md";contract.write_text("SYNTHETIC evaluation fixture only; no scientific release")
            approval=root/"approval.json";approval.write_text("{}")
            config=root/"evaluate.json"
            cfg=dict(engine_schema_version=4,operation="readout-evaluate",contract=record(contract),approval=record(approval),
                inputs=dict(seal=record(root/"seal/group_seal.json"),catalog=record(catalog),query_manifest=record(query),affiliation=record(identity)),
                parameters=dict(profile=profile),output=str(root/"evaluation"),score_space=SCORE_SPACE)
            def approve():
                config.write_text(json.dumps(cfg))
                _,_,_,_,digest=load_readout_operation(config)
                approval.write_text(json.dumps(dict(status="APPROVED",contract_sha256=cfg["contract"]["sha256"],
                    operation_manifest_sha256=digest,scope="SYNTHETIC_UNIT_TEST_ONLY")))
                cfg["approval"]=record(approval);config.write_text(json.dumps(cfg))
            approve()
            self.assertFalse(run_readout(config)["executed"])
            with self.assertRaisesRegex(ValueError,"allow-labels"):
                run_readout(config,execute=True)
            result=run_readout(config,execute=True,allow_labels=True)
            good_config,good_approval=config.read_bytes(),approval.read_bytes()
            self.assertEqual(len(result["metrics"]),72)
            self.assertEqual(len(result["per_entity_Aff"]),72)
            self.assertIsNone(result["overall_Aff_F1"])
            self.assertEqual(result["statistics"]["intervals"][0]["rng_seed"],99174)
            with (root/"evaluation/entity_metrics.csv").open(newline="") as stream:csv_rows=list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows),72)
            self.assertEqual(csv_rows[0]["operating_condition_reason"],"NOT_PROVIDED_BY_FROZEN_QUERY_METADATA")
            for filename,count in (("macro_summary.csv",12),("seed_summary.csv",36),("fault_summary.csv",24),("run_summary.csv",24),("paired_run_intervals.csv",3)):
                with (root/"evaluation"/filename).open(newline="") as stream:self.assertEqual(len(list(csv.DictReader(stream))),count)
            manifest=json.loads((root/"evaluation/evaluation_manifest.json").read_text())
            for name,digest in manifest["files"].items():
                from hybrid_tsad.config import sha256
                self.assertEqual(sha256(root/"evaluation"/name),digest)
            coordinate=next((root/"evaluation/coordinates").rglob("*.npz"))
            with np.load(coordinate) as bank:
                self.assertTrue(np.array_equal(bank["time_index"],np.arange(100,120)))
                self.assertTrue(np.array_equal(bank["history_start"],np.array([0,5,10,15])))
                self.assertTrue(np.array_equal(bank["target_end_exclusive"],np.array([105,110,115,120])))
            with np.load(root/"evaluation/paired_run_bootstrap.npz") as bank:
                self.assertEqual(bank["draws"].shape,(2000,2))
            # Loss of any sealed bank blocks evaluation before binary labels.
            broken=json.loads((root/"seal/group_seal.json").read_text());broken["native_banks"].pop(next(iter(broken["native_banks"])))
            path=root/"broken_seal.json";path.write_text(json.dumps(broken))
            real_load=np.load
            def locked_load(path,*a,**kw):
                if "_labels" in str(path):raise AssertionError("label opened without all-bank completeness")
                return real_load(path,*a,**kw)
            with patch.object(np,"load",locked_load),self.assertRaisesRegex(ValueError,"All-bank label lock"):
                evaluate_group(path,catalog,query,identity,root/"blocked",profile=profile)
            missing=json.loads(identity.read_text());missing["directory"]=str(root/"missing_official")
            missing_path=root/"missing_affiliation.json";missing_path.write_text(json.dumps(missing))
            cfg["inputs"]["affiliation"]=record(missing_path);cfg["output"]=str(root/"blocked_dependency");approve()
            with patch.object(np,"load",locked_load),self.assertRaisesRegex(OfficialAffiliationBlocked,"BLOCKED"):
                run_readout(config,execute=True,allow_labels=True)
            self.assertFalse((root/"blocked_dependency").exists())
            # Wrong original labels fail explicitly; no repair and no normal
            # evaluation report can be mistaken for a complete archive.
            label_path=Path(json.loads(query.read_text())["records"][0]["label_path"])
            original_label=label_path.read_bytes()
            label_path.write_bytes(original_label+b"tamper")
            cfg["inputs"]["affiliation"]=record(identity);cfg["output"]=str(root/"failed_label_identity");approve()
            with self.assertRaisesRegex(ValueError,"label identity mismatch"):
                run_readout(config,execute=True,allow_labels=True)
            self.assertTrue((root/"failed_label_identity/evaluation_failure.json").is_file())
            self.assertFalse((root/"failed_label_identity/evaluation_manifest.json").exists())
            # Optional engineering archive retains a reproducible valid operation,
            # plus separate negative-test evidence. Never use a real query here.
            label_path.write_bytes(original_label)
            config.write_bytes(good_config);approval.write_bytes(good_approval)
