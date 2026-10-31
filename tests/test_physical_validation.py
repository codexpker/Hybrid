"""CPU-only provenance rejection tests; CUDA integration remains separately gated."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from hybrid_tsad.engine.physical_validation import (CPU_BACKEND, CUDA_BACKEND, CUDA_RUNTIME,
    physical_validation_spec, validate_physical_runtime, bind_execution_evidence)
from hybrid_tsad.engine.readout_manifest import record, load_readout_operation, approve_readout_inputs

CUDA_SPEC={"backend":CUDA_BACKEND,"device":"cuda:0"}
CPU_SPEC={"backend":CPU_BACKEND,"device":"cpu"}
PROFILE=dict(name="SYNTHETIC",faults=[1],runs=[31],points=105,channels=52,event_start=100,sample_interval_minutes=3)
HAS_TORCH=importlib.util.find_spec("torch") is not None


def json_file(path,value):
    path.write_text(json.dumps(value),encoding="utf-8")
    return path


def evidence_fixture(root):
    """Dummy hashed files/CPU metadata only; never executable CUDA bank evidence.

    Tests patch ONLY the frozen original-code identity pin for this toy graph.
    Production accepts the exact audited A identity, not a self-signed fixture.
    """
    source=root/"original/src";source.mkdir(parents=True)
    source_file=source/"model.txt";source_file.write_text("synthetic original code bytes")
    identity=json_file(root/"identity.json",{"files":{"src/model.txt":record(source_file)["sha256"]}})
    environment=json_file(root/"env.json",dict(torch=CUDA_RUNTIME["torch_version"],cuda=CUDA_RUNTIME["cuda_version"]))
    interpreter="/existing/synthetic/python"
    commands=json_file(root/"commands.json",dict(prepare=interpreter+" prepare",verify=interpreter+" verify"))
    catalog=json_file(root/"catalog.json",{})
    q=dict(entity="fault01_run031",fault_id=1,run_id=31,role="query",history=100,horizon=5,guard=0,
           local_start=0,local_end_exclusive=105,channels=52,origins=[100])
    query=json_file(root/"query.json",dict(dataset="SYNTHETIC",records=[q]))
    cells=[dict(seed=0,mode="temporal",entity=q["entity"],origins=[100])]
    whitelist=json_file(root/"whitelist.json",cells);contract=json_file(root/"contract.json",{})
    approval=json_file(root/"approval.json",{})
    output=root/"shard";output.mkdir()
    cfg=dict(engine_schema_version=4,operation="readout-infer",contract=record(contract),approval=record(approval),
        inputs=dict(catalog=record(catalog),query_manifest=record(query),whitelist=record(whitelist)),
        parameters=dict(profile=PROFILE,scope="shard",device="cuda:0"),output=str(output),
        score_space="fixed_native_late_canonical_source_controls_v1")
    operation=json_file(root/"operation.json",cfg)
    *_,digest=load_readout_operation(operation)
    json_file(approval,dict(status="APPROVED",operation_manifest_sha256=digest,contract_sha256=cfg["contract"]["sha256"],
                            approved_at_utc="2026-11-01T00:00:00+00:00"))
    cfg["approval"]=record(approval);json_file(operation,cfg)
    bank={part:record(json_file(root/(part+".json"),{})) for part in ("bank","rng","metadata")}
    key="s0/temporal/fault01_run031"
    shard=dict(status="SHARD_ONLY_LABELS_LOCKED",profile=PROFILE,scope="shard",labels_opened=False,
               catalog_sha256=record(catalog)["sha256"],query_manifest_sha256=record(query)["sha256"],whitelist=cells,banks={key:bank})
    shard_path=json_file(output/"shard_manifest.json",shard)
    log=json_file(root/"stdout.json",shard)
    listing=json_file(root/"shards.json",[record(shard_path)])
    command=["timeout","--kill-after=5s","60s","env","CUDA_VISIBLE_DEVICES=0","PYTHONPATH="+str(source),
             "OMP_NUM_THREADS=2","OPENBLAS_NUM_THREADS=2",interpreter,"-m","hybrid_tsad","readout","--config",str(operation),"--execute"]
    launch=json_file(root/"launch.json",dict(command=command,GPU=0,launched_utc="2026-11-01T00:01:00+00:00"))
    row=dict(name="pilot",shard_manifest=record(shard_path),infer_operation=record(operation),approval=record(approval),
        release=record(approval),launch=record(launch),launch_selector="top-level",stdout_log=record(log),whitelist=record(whitelist),
        runtime=record(identity),runtime_source_root=str(source),physical_gpu_index_from_original_CUDA_VISIBLE_DEVICES=0,
        physical_gpu_uuid="UNVERIFIED",physical_gpu_model="UNVERIFIED",logical_device_from_original_operation="cuda:0",
        backend_type="cuda",recorded_input_contract=dict(shape=[1,100,52],dtype="float32",history=100),
        bank_count=1,bank_keys=[key],bank_metadata_bindings=[dict(key=key,metadata=bank["metadata"])])
    evidence=dict(schema_version=1,status="CANDIDATE_NOT_PRIMARY_APPROVED_METADATA_DERIVATION_ONLY",profile=PROFILE,
        evidence_not_approval=True,original_code_identity=record(identity),runtime_source_root=str(source),
        actual_runtime_files=[record(source_file)],original_core_files={"model":record(source_file)},
        historical_versions=dict(job_time_Torch="UNVERIFIED",job_time_CUDA="UNVERIFIED",
            prior_frozen_record=dict(environment=record(environment),commands=record(commands))),shards=[row])
    path=json_file(root/"evidence.json",evidence)
    return path,listing,{key:bank},catalog,query,identity


class PhysicalValidationMetadataTests(unittest.TestCase):
    def test_default_cpu_only_synthetic_and_unapproved_backend_refused(self):
        self.assertEqual(physical_validation_spec(None,PROFILE),CPU_SPEC)
        for value,profile in ((None,{"name":"TEP100"}),(CPU_SPEC,{"name":"TEP100"}),
                ({"backend":"float64","device":"cpu"},PROFILE),({**CUDA_SPEC,"device":"cuda:1"},PROFILE),
                (CUDA_SPEC,{**PROFILE,"channels":1})):
            with self.subTest(value=value),self.assertRaises(ValueError):physical_validation_spec(value,profile)
        self.assertEqual(physical_validation_spec(CUDA_SPEC,PROFILE),CUDA_SPEC)

    def test_no_cuda_fallback_and_runtime_is_explicit_not_bank_self_report(self):
        cuda=SimpleNamespace(is_available=lambda:False,get_device_name=lambda _:self.fail("no fallback device probe"))
        fake=SimpleNamespace(cuda=cuda)
        with patch.dict(sys.modules,{"torch":fake}),self.assertRaisesRegex(ValueError,"no CPU fallback"):
            validate_physical_runtime(CUDA_SPEC)
        with patch.dict(sys.modules,{"torch":fake}):validate_physical_runtime(CPU_SPEC)
        cuda.is_available=lambda:True;cuda.get_device_name=lambda _:"NVIDIA A40";cuda.get_device_capability=lambda _:(8,6)
        fake.__version__="2.8.0+cu128";fake.version=SimpleNamespace(cuda="12.8");fake.is_autocast_enabled=lambda _:False
        with patch.dict(sys.modules,{"torch":fake}):validate_physical_runtime(CUDA_SPEC)  # mock only, no GPU claim
        fake.__version__="wrong"
        with patch.dict(sys.modules,{"torch":fake}),self.assertRaisesRegex(ValueError,"runtime differs"):
            validate_physical_runtime(CUDA_SPEC)
        fake.__version__="2.8.0+cu128";cuda.get_device_name=lambda _:"wrong GPU"
        with patch.dict(sys.modules,{"torch":fake}),self.assertRaisesRegex(ValueError,"runtime differs"):
            validate_physical_runtime(CUDA_SPEC)

    def test_evidence_binds_actual_launch_operation_log_source_and_bank_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            args=evidence_fixture(Path(temp));path,listing,banks,catalog,query,identity=args
            with patch("hybrid_tsad.engine.physical_validation.ORIGINAL_A_CODE_IDENTITY_SHA",record(identity)["sha256"]):
                self.assertEqual(bind_execution_evidence(path,listing,banks,catalog,query,PROFILE,CUDA_SPEC),record(path))
            # A production pin cannot be replaced by a fresh self-described hash.
            with self.assertRaisesRegex(ValueError,"audited A snapshot"):
                bind_execution_evidence(path,listing,banks,catalog,query,PROFILE,CUDA_SPEC)
            with self.assertRaisesRegex(ValueError,"requires frozen execution evidence"):
                bind_execution_evidence(None,listing,banks,catalog,query,PROFILE,CUDA_SPEC)

    def test_separate_original_release_digest_must_cover_the_launched_operation(self):
        with tempfile.TemporaryDirectory() as temp:
            path,listing,banks,catalog,query,identity=evidence_fixture(Path(temp))
            value=json.loads(path.read_text());row=value["shards"][0]
            digest=json.loads(Path(row["approval"]["path"]).read_text())["operation_manifest_sha256"]
            release_path=Path(temp)/"release.json"
            release=dict(status="REMAINING3597_RELEASED",operation_digests={row["name"]:digest},labels_released=False,
                         released_at_utc="2026-11-01T00:00:30+00:00")
            json_file(release_path,release);row["release"]=record(release_path);json_file(path,value)
            with patch("hybrid_tsad.engine.physical_validation.ORIGINAL_A_CODE_IDENTITY_SHA",record(identity)["sha256"]):
                bind_execution_evidence(path,listing,banks,catalog,query,PROFILE,CUDA_SPEC)
                release["operation_digests"][row["name"]]="0"*64;json_file(release_path,release)
                row["release"]=record(release_path);json_file(path,value)
                with self.assertRaisesRegex(ValueError,"release does not cover"):
                    bind_execution_evidence(path,listing,banks,catalog,query,PROFILE,CUDA_SPEC)

    def test_rehashed_metadata_forgery_still_rejected(self):
        for attack,reason in (("launch","device/interpreter/source"),("operation","exact shard/input/output"),
                              ("log","completion JSON/log"),("bank","bank hashes"),("source","identity mismatch"),
                              ("time","precede launch"),("missing","every original shard")):
            with self.subTest(attack=attack),tempfile.TemporaryDirectory() as temp:
                path,listing,banks,catalog,query,identity=evidence_fixture(Path(temp))
                value=json.loads(path.read_text());row=value["shards"][0]
                if attack=="launch":
                    p=Path(row["launch"]["path"]);v=json.loads(p.read_text());v["GPU"]=1;json_file(p,v);row["launch"]=record(p)
                elif attack=="operation":
                    p=Path(row["infer_operation"]["path"]);v=json.loads(p.read_text());v["parameters"]["device"]="cpu";json_file(p,v);row["infer_operation"]=record(p)
                elif attack=="log":
                    p=Path(row["stdout_log"]["path"]);v=json.loads(p.read_text());v["labels_opened"]=True;json_file(p,v);row["stdout_log"]=record(p)
                elif attack=="bank":next(iter(banks.values()))["rng"]={"path":"forged","sha256":"0"*64}
                elif attack=="source":Path(value["actual_runtime_files"][0]["path"]).write_text("changed code")
                elif attack=="time":
                    p=Path(row["launch"]["path"]);v=json.loads(p.read_text());v["launched_utc"]="2025-01-01T00:00:00+00:00";json_file(p,v);row["launch"]=record(p)
                else:value["shards"]=[]
                json_file(path,value)
                with patch("hybrid_tsad.engine.physical_validation.ORIGINAL_A_CODE_IDENTITY_SHA",record(identity)["sha256"]),self.assertRaisesRegex(ValueError,reason):
                    bind_execution_evidence(path,listing,banks,catalog,query,PROFILE,CUDA_SPEC)

    def test_schema4_approval_binds_backend_and_evidence_without_gpu_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ("catalog","query_manifest","shards","contract","approval","execution_evidence"):json_file(root/name,{})
            cfg=dict(engine_schema_version=4,operation="readout-seal",contract=record(root/"contract"),approval=record(root/"approval"),
                inputs={n:record(root/n) for n in ("catalog","query_manifest","shards")},parameters=dict(profile=PROFILE),
                output=str(root/"new"),score_space="fixed_native_late_canonical_source_controls_v1")
            path=json_file(root/"operation.json",cfg);*_,digest=load_readout_operation(path)
            json_file(root/"approval",dict(status="APPROVED",contract_sha256=cfg["contract"]["sha256"],operation_manifest_sha256=digest))
            cfg["approval"]=record(root/"approval");cfg["parameters"]["physical_validation"]=CUDA_SPEC
            cfg["inputs"]["execution_evidence"]=record(root/"execution_evidence");json_file(path,cfg)
            parsed,paths,out,_,newdigest=load_readout_operation(path);self.assertNotEqual(digest,newdigest)
            with self.assertRaisesRegex(ValueError,"exact operation"):approve_readout_inputs(parsed,paths,out,path)
            cfg["parameters"]=dict(profile={**PROFILE,"name":"TEP100"});json_file(path,cfg)
            with self.assertRaisesRegex(ValueError,"explicit approved backend"):load_readout_operation(path)

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"existing NumPy required to import evaluator wrapper")
    def test_evaluate_passes_backend_to_unlock_before_external_callback_or_labels(self):
        from hybrid_tsad.engine.multi_evaluation import evaluate_group
        with patch("hybrid_tsad.engine.multi_evaluation.verify_label_unlock",side_effect=ValueError("locked backend")) as unlock, \
             patch("hybrid_tsad.engine.multi_evaluation.load_official_affiliation",side_effect=AssertionError("callback loaded before unlock")):
            with self.assertRaisesRegex(ValueError,"locked backend"):
                evaluate_group("seal","catalog","query","aff","new",profile=PROFILE,physical_validation=CUDA_SPEC)
        unlock.assert_called_once_with("seal","catalog","query",PROFILE,physical_validation=CUDA_SPEC)


@unittest.skipUnless(HAS_TORCH,"existing torch required for synthetic CPU preprocess; CUDA not executed")
class PhysicalValidationCPUReferenceTests(unittest.TestCase):
    def test_constant_cpu_reference_exact_rng_and_scale_forgery_rejected(self):
        import numpy as np
        import torch
        from hybrid_tsad.model import HybridPredictor
        from hybrid_tsad.engine.multi_readout import physical_reference,verify_physical_bank,HISTORY_SCALE_RTOL,HISTORY_SCALE_ATOL
        self.assertEqual((HISTORY_SCALE_RTOL,HISTORY_SCALE_ATOL),(2e-6,2e-6))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=np.full((105,52),np.float32(3000.10009765625),dtype="float32")
            array=root/"history.npy";np.save(array,data)
            q=dict(array_path=str(array),array_sha256_from_frozen_parent=record(array)["sha256"],origins=[100],local_start=0,local_end_exclusive=105)
            before=torch.get_rng_state().clone()
            with patch.object(torch.cuda,"is_available",side_effect=AssertionError("CPU must not probe CUDA")):
                reference=physical_reference(q,PROFILE)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            scale=HybridPredictor.preprocess(SimpleNamespace(history=100),torch.tensor(data[None,:100].copy()))[3].numpy()[0]
            self.assertTrue(np.array_equal(reference[2],scale[None]))
            bank=dict(time_index=np.arange(100,105),forecast_origin=np.array([100]),target=data[100:],history_scale_by_block=reference[2])
            path=root/"bank.npz";np.savez(path,**bank);verify_physical_bank(path,q,PROFILE,reference)
            bank["history_scale_by_block"]=reference[2]*2;np.savez(path,**bank)
            with self.assertRaisesRegex(ValueError,"history scale"):verify_physical_bank(path,q,PROFILE,reference)
            # Frozen diagnostic GPU literal: its rejection by CPU is not a CUDA test.
            bank["history_scale_by_block"]=np.full_like(reference[2],.0031997528858482838);np.savez(path,**bank)
            with self.assertRaisesRegex(ValueError,"history scale"):verify_physical_bank(path,q,PROFILE,reference)
