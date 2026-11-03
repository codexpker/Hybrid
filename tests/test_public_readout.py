"""New glue metadata tests only; fake asset bytes are never models or arrays."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hybrid_tsad.hybrid_v1 import public_readout as public
from hybrid_tsad.hybrid_v1.config import HybridV1Config
from hybrid_tsad.hybrid_v1 import readout


class PublicReadoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Synthetic metadata authority only; no claim these fake bytes are PT/NPZ.
        for name in ("FULL_SOURCE", "BASELINE_CAL"):
            patch = mock.patch.object(public, name, copy.deepcopy(getattr(public, name)))
            patch.start(); self.addCleanup(patch.stop)
        for name in ("MSL_MANIFEST", "MSL_NPZ", "REFERENCE"):
            patch = mock.patch.object(public, name, getattr(public, name))
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(public, "external_identity", return_value={"synthetic_external": "0"*64})
        patch.start(); self.addCleanup(patch.stop)

    def asset(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return dict(path=str(path), sha256=public.sha(path))

    def packet(self, operation=public.PROFILE, bad_train_phase=False):
        model = HybridV1Config().to_dict()
        cfg = dict(operation=operation, profile="SYNTHETIC_C1_COST128_V1" if operation == public.PROFILE else "MSL_PROCESSED_COL0_SOURCE_Z_V1",
            permissions=public.permissions(operation), resume=False, history=512, horizon=32, chunk_size=64,
            source_seed=2025, inference_seed=2025, control_id="synthetic_metadata_full2025", route="SOURCE_CONTROL",
            physical_gpu=0, expected_gpu_uuid="GPU-SYNTHETIC", device="cuda:0", model=model,
            sources={"fixture": "no_external_import"}, active_components=["r_tf", "r_ms", "r_pred"],
            output=str(self.root/"output"), stream_source_sha256=public.source_identity(), assets={},
            resources=dict(wall_seconds=175, kill_grace_seconds=5, evaluation_GiB=30, max_GPUh=.05, automatic_retry=False))
        assets = cfg["assets"]
        if operation != public.BASELINE:
            assets["checkpoint"] = self.asset("fake_state.pt", "SYNTHETIC_NOT_A_CHECKPOINT")
            cal = dict(active_components=cfg["active_components"], weights=dict(r_tf=.25, r_ms=.25, r_pred=.5),
                       scales=dict(r_tf=1., r_ms=1., r_pred=1.), threshold=1., source_domain_equal_weight=True,
                       domains=["SMD", "Sunspot"])
            assets["calibration"] = self.asset("calibration.json", cal)
            public.FULL_SOURCE[2025] = (assets["checkpoint"]["sha256"], assets["calibration"]["sha256"])
            train = dict(operation="HYBRID_V1_SOURCE_STAGE1", model=model, sources=cfg["sources"], seed=2025,
                         history=512, horizon=32, active_components=cfg["active_components"])
            assets["training_config"] = self.asset("train.json", train)
            assets["training_receipt"] = self.asset("trained.json", dict(
                phase="WRONG" if bad_train_phase else "source_train", synthetic_non_scientific_results=False,
                files={"source.pt": assets["checkpoint"]["sha256"], "config.json": assets["training_config"]["sha256"]}))
            cal_config = dict(train, checkpoint_sha256=assets["checkpoint"]["sha256"],
                              train_receipt_sha256=assets["training_receipt"]["sha256"])
            assets["calibration_config"] = self.asset("cal_config.json", cal_config)
            assets["calibration_receipt"] = self.asset("calibrated.json", dict(
                phase="calibrate_source", synthetic_non_scientific_results=False,
                files={"source_calibration.json": assets["calibration"]["sha256"],
                       "config.json": assets["calibration_config"]["sha256"]}))
            assets["asset_seal"] = self.asset("seal.json", dict(status="ACCEPTED_NAMED_READOUT_ASSETS", approved_by="root",
                bindings=[dict(control_id=cfg["control_id"], route="SOURCE_CONTROL", source_seed=2025,
                               assets={k: assets[k] for k in public.SCIENCE})]))
        else:
            cfg.update(device="cpu", methods=public.METHODS)
            cfg["resources"] = dict(wall_seconds=600, kill_grace_seconds=5, AS_GiB=8, automatic_retry=False)
            methods = []
            for method in public.METHODS:
                assets[method+"_calibration"] = self.asset(method+".json", dict(synthetic_metadata_only=True))
                public.BASELINE_CAL[method] = assets[method+"_calibration"]["sha256"]
                methods.append(dict(method=method, calibration_sha256=public.BASELINE_CAL[method]))
            assets["baseline_acceptance"] = self.asset("baseline_acceptance.json", dict(
                status="ACCEPTED_LIMITED_BASELINE_SOURCE_CAL", approved_by="root", methods=methods))
        if operation == public.PROFILE:
            cfg.update(synthetic=public.SYNTHETIC.copy(), reference_points=3)
            assets["reference_training"] = self.asset("reference.py", "SYNTHETIC_REFERENCE_METADATA_ONLY")
            public.REFERENCE = assets["reference_training"]["sha256"]
        else:
            cfg["expected_records"] = 73217
            assets["input_data"] = self.asset("query_inputs.npz", "NOT_AN_ARRAY_NO_VALUE_IO")
            public.MSL_NPZ = assets["input_data"]["sha256"]
            entry = dict(domain="MSL_AUTHOR_PROCESSED_COL0", entity="processed_anonymous_stream", run="processed_test_axis",
                role="QUERY", path="query_inputs.npz", format="npz", key="values", observed_key="observed",
                bounds=[0,73729], expected_shape=[73729,1], sha256=public.MSL_NPZ, original_start=0)
            assets["query_manifest"] = self.asset("INPUT_MANIFEST.json", dict(runs=[entry]))
            public.MSL_MANIFEST = assets["query_manifest"]["sha256"]
            assets["query_acceptance"] = self.asset("query_acceptance.json", dict(
                status="ACCEPTED_LIMITED_PROCESSED_QUERY_INPUT", approved_by="root", dataset="MSL_AUTHOR_PROCESSED_COL0",
                manifest_sha256=public.MSL_MANIFEST, products={"query_inputs.npz": public.MSL_NPZ},
                shape=[73729,1], Run_original_start=0, primary_feature_indices=[0]))
        return cfg

    def authority(self, cfg):
        config = self.root/"config.json"; config.write_text(json.dumps(cfg))
        release = dict(status="APPROVED", approved_by="root", operation=cfg["operation"], allowed_actions=[cfg["operation"]],
            config_sha256=public.sha(config), resources=cfg["resources"], allow_resume=False,
            source_sha256=cfg["stream_source_sha256"], external_source_sha256={"synthetic_external":"0"*64})
        if cfg["operation"] != public.BASELINE:
            release.update(asset_seal_sha256=cfg["assets"]["asset_seal"]["sha256"], named_binding={k:cfg[k] for k in
                ("control_id","route","source_seed","inference_seed","physical_gpu","expected_gpu_uuid")})
        path=self.root/"release.json"; path.write_text(json.dumps(release))
        return config,path,public.sha(path)

    def test_cost_admit_without_torch_or_arrays(self):
        cfg=self.packet(); config,release,root=self.authority(cfg)
        self.assertEqual(public.admit(config,release,root)["operation"],public.PROFILE)
        self.assertFalse(Path(cfg["output"]).exists())

    def test_real_model_metadata_only_full_chain(self):
        cfg=self.packet(public.MODEL);config,release,root=self.authority(cfg)
        admitted=public.admit(config,release,root)
        self.assertEqual(admitted["expected_records"],73217)
        self.assertEqual(Path(cfg["assets"]["input_data"]["path"]).read_text(),"NOT_AN_ARRAY_NO_VALUE_IO")

    def test_two_deterministic_baselines_admit_once(self):
        cfg=self.packet(public.BASELINE);config,release,root=self.authority(cfg)
        self.assertEqual(public.admit(config,release,root)["methods"],public.METHODS)

    def test_real_cannot_claim_synthetic_exemption(self):
        cfg=self.packet(public.PROFILE)
        cfg["assets"]["query_manifest"]={"path":"REAL_MUST_NOT_OPEN","sha256":"0"*64}
        with self.assertRaisesRegex(PermissionError,"assets required"):public.validate_config(cfg)
        del cfg["assets"]["query_manifest"]
        cfg["operation"]=public.MODEL;cfg["permissions"]=public.permissions(public.MODEL)
        with self.assertRaises(PermissionError):public.validate_config(cfg)

    def test_frozen_profile_shape_and_hard_cost_limit(self):
        cfg=self.packet()
        for key,value in [("channels",25),("length",641),("rng_seed",17)]:
            changed=copy.deepcopy(cfg);changed["synthetic"][key]=value
            with self.assertRaises(PermissionError):public.validate_config(changed)
        changed=copy.deepcopy(cfg);changed["resources"]["wall_seconds"]=180
        with self.assertRaises(PermissionError):public.validate_config(changed)

    def test_wrong_seed_gpu_recipe_resume_labels_rejected(self):
        cfg=self.packet()
        for key,value in [("source_seed",2026),("physical_gpu",1),("inference_seed",17),("resume",True),
                          ("active_components",["r_pred"]),("labels","FORBIDDEN")]:
            changed=copy.deepcopy(cfg);changed[key]=value
            with self.assertRaises(PermissionError):public.validate_config(changed)

    def test_no_h5_checkpoint_or_target_cal_substitution(self):
        cfg=self.packet(public.MODEL)
        for key in ("checkpoint","calibration"):
            changed=copy.deepcopy(cfg);changed["assets"][key]["sha256"]="e"*64
            with self.assertRaisesRegex(PermissionError,"original full"):public.validate_config(changed)

    def test_wrong_source_receipt_phase_is_not_a_valid_seal(self):
        cfg=self.packet(bad_train_phase=True);config,release,root=self.authority(cfg)
        with self.assertRaisesRegex(PermissionError,"lineage mismatch"):public.admit(config,release,root)

    def test_missing_or_wrong_root_and_changed_source(self):
        cfg=self.packet();config,release,root=self.authority(cfg)
        with self.assertRaises(PermissionError):public.admit(config,None,None)
        with self.assertRaises(PermissionError):public.admit(config,release,"0"*64)
        cfg["stream_source_sha256"]["public_readout.py"]="0"*64
        config,release,root=self.authority(cfg)
        with self.assertRaisesRegex(PermissionError,"source required"):public.admit(config,release,root)

    def test_fresh_output_refuses_existing_prefix(self):
        cfg=self.packet();Path(cfg["output"]).mkdir();config,release,root=self.authority(cfg)
        with self.assertRaisesRegex(PermissionError,"Fresh output"):public.admit(config,release,root)

    def test_public_query_axis_rejects_train_shift_shape_or_labels(self):
        cfg=self.packet(public.MODEL)
        manifest=json.loads(Path(cfg["assets"]["query_manifest"]["path"]).read_text())
        acceptance=json.loads(Path(cfg["assets"]["query_acceptance"]["path"]).read_text())
        for key,value in [("role","FIT"),("original_start",58317),("expected_shape",[73729,55]),("label_path","FORBIDDEN")]:
            changed=copy.deepcopy(manifest);changed["runs"][0][key]=value
            with self.assertRaises(PermissionError):public.query_metadata(changed,acceptance,cfg["assets"])

    def test_shared_kernel_identity_and_full_tail_coverage_gate(self):
        self.assertIs(public.write_model_bank,readout.write_model_bank)
        self.assertIs(public.write_baseline_bank,readout.write_baseline_bank)
        bank=self.root/"bank";bank.mkdir()
        path=bank/"manifest.json";path.write_text(json.dumps(dict(complete=True,committed_records=73217)))
        self.assertEqual(public._complete(bank,73217),public.sha(path))
        path.write_text(json.dumps(dict(complete=True,committed_records=73186)))
        with self.assertRaises(ValueError):public._complete(bank,73217)

    def test_default_preview_never_hashes_asset_or_constructs_model(self):
        cfg=self.packet();path=self.root/"preview.json";path.write_text(json.dumps(cfg))
        with mock.patch.object(public,"sha",side_effect=AssertionError("asset opened")):
            with contextlib.redirect_stdout(io.StringIO()) as stream:public.main(["--config",str(path)])
        self.assertFalse(json.loads(stream.getvalue())["model_constructed"])


if __name__=="__main__":unittest.main()
