"""Six synthetic named banks only; no production bank/label/model/GPU access."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import named_evaluate_cli as cli
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,IDENTITY_KEYS


def reseal(cfg):
    path=Path(cfg["raw_seal"]["path"])
    path.write_text(json.dumps(dict(status="ACCEPTED_NAMED_FIXED_BANKS",approved_by="root",profile=cfg["profile"],banks=cfg["banks"])))
    cfg["raw_seal"]["sha256"]=cli.sha(path)


def fixture(root,missing=None,gap=False):
    y=np.zeros(18,np.int8);y[1:3]=1;y[5:7]=1;y[11:13]=1;y[16:18]=1
    labels=root/"synthetic_labels.csv";labels.write_text("timestamp_(min),label\n"+"".join("%d,%d\n"%(i,v) for i,v in enumerate(y)))
    cfg=dict(profile="SYNTHETIC_NAMED",banks=[],comparisons=[],history=2,original_length=18,original_start=0,stream=["SYNTHETIC","E","R"],
        labels=dict(path=str(labels),sha256=cli.sha(labels),shape=[18],format="csv",drop_columns=["timestamp_(min)"],value_columns=["label"]),
        output=str(root/"output"),affiliation_identity=None,statistics=dict(cli.STATISTICS_NA),raw_seal=dict(path=str(root/"seal.json"),sha256=None))
    methods=list(cli.A1_MODELS.items())+[(name,name) for name in sorted(cli.BASELINES)]
    for name,control in methods:
        kind="deterministic_baseline" if name in cli.BASELINES else "source_model"
        cal=root/(name+"_cal.json");cal.write_text(json.dumps(dict(threshold=1.,source_domain_equal_weight=True)))
        identity={k:"a"*64 for k in IDENTITY_KEYS};identity["calibration_sha256"]=cli.sha(cal)
        ranges=[dict(domain="SYNTHETIC",entity="E",run="R",start=a,end=b) for a,b in ([(2,8),(10,18)] if gap else [(2,18)])]
        directory=root/name
        with ChunkedBank(directory,identity=identity,ranges=ranges,threshold=1.,chunk_size=64) as bank:
            for span in ranges:
                for i in range(span["start"],span["end"]):
                    alarm=(5<=i<14) if control=="A1_fusion_none" else i in (2,5,6,11,12,16,17)
                    valid=i not in (missing or {}).get(name,())
                    bank.append(dict(domain="SYNTHETIC",entity="E",run="R",index=i,available_index=i,
                        score=2. if alarm else 0.,valid=valid,reason=None if valid else "SYNTHETIC_MISSING"))
            bank.finalize()
        cfg["banks"].append(dict(id=name,control_id=control,kind=kind,training_seed=None if kind=="deterministic_baseline" else 2025,
            inference_seed=None if kind=="deterministic_baseline" else 2025,calibration_origin="source",directory=str(directory),
            manifest_sha256=cli.sha(directory/"manifest.json"),identity=identity,calibration=dict(path=str(cal),sha256=cli.sha(cal)),ranges=ranges))
        if name!="full_seed2025":cfg["comparisons"].append(dict(reference="full_seed2025",candidate=name,pairing="model_vs_deterministic" if kind=="deterministic_baseline" else "same_training_seed"))
    reseal(cfg);return cfg,y


class NamedEvaluation(unittest.TestCase):
    def test_preview_and_pending_never_open_labels_or_banks(self):
        with patch("pathlib.Path.open",side_effect=AssertionError("No I/O")),contextlib.redirect_stdout(io.StringIO()) as log:cli.main(["--config","NOT_FOUND"])
        self.assertIn('"label_or_bank_opened": false',log.getvalue())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"config.json";path.write_text('{}')
            with patch.object(cli,"load_labels",side_effect=AssertionError("No labels")):
                with self.assertRaises(PermissionError):cli.admit(path,None,None)

    def test_missing_or_changed_root_seal_before_label_SHA_or_loader(self):
        for change in ("missing","wrong_sha","unaccepted","wrong_named_descriptor"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_=fixture(Path(d));p=Path(cfg["raw_seal"]["path"])
                if change=="missing":cfg["raw_seal"]["sha256"]=None
                elif change=="wrong_sha":p.write_text('{}')
                elif change=="unaccepted":
                    v=json.loads(p.read_text());v["status"]="PENDING";p.write_text(json.dumps(v));cfg["raw_seal"]["sha256"]=cli.sha(p)
                else:cfg["banks"][1]["identity"]["checkpoint_sha256"]="b"*64
                with patch.object(cli,"load_labels",side_effect=AssertionError("No label opener")) as loader:
                    with self.assertRaises(PermissionError):cli.run_evaluation(cfg)
                    loader.assert_not_called()
                self.assertFalse(Path(cfg["output"]).exists())

    def test_changed_CAL_manifest_chunk_or_clock_refused_before_labels(self):
        for change in ("CAL","manifest","chunk","available_clock"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_=fixture(Path(d));e=cfg["banks"][-1];mp=Path(e["directory"])/"manifest.json";m=json.loads(mp.read_text())
                if change=="CAL":Path(e["calibration"]["path"]).write_text('{}')
                elif change=="manifest":m["complete"]=False;mp.write_text(json.dumps(m));e["manifest_sha256"]=cli.sha(mp);reseal(cfg)
                elif change=="chunk":(Path(e["directory"])/m["chunks"][0]["file"]).write_bytes(b"CORRUPT_SYNTHETIC")
                else:
                    from hybrid_tsad.hybrid_v1.chunked_bank import read_records
                    original=read_records
                    def wrong(directory):
                        for row in original(directory):yield dict(row,available_index=row["index"]+1)
                    with patch("hybrid_tsad.hybrid_v1.chunked_bank.read_records",side_effect=wrong),patch.object(cli,"load_labels",side_effect=AssertionError("No labels")):
                        with self.assertRaises(ValueError):cli.run_evaluation(cfg)
                    continue
                with patch.object(cli,"load_labels",side_effect=AssertionError("No labels")) as loader:
                    with self.assertRaises(ValueError):cli.run_evaluation(cfg)
                    loader.assert_not_called()

    def test_distinct_controls_not_fake_seeds_baselines_or_ensemble(self):
        for change in ("seed","kind","mean","ensemble","alias"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_=fixture(Path(d))
                if change=="seed":cfg["banks"][1]["training_seed"]=2026;cfg["banks"][1]["inference_seed"]=2026
                elif change=="kind":cfg["banks"][1].update(kind="deterministic_baseline",training_seed=None,inference_seed=None)
                elif change=="mean":cfg["model_mean_SD"]={}
                elif change=="ensemble":cfg["ensemble"]=True
                else:cfg["banks"][1]["directory"]=cfg["banks"][0]["directory"]
                with patch.object(cli,"load_labels",side_effect=AssertionError("No labels")),self.assertRaises(ValueError):cli.run_evaluation(cfg)

    def test_target_CAL_cannot_relabel_old_source_derived_scores(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,_=fixture(Path(d));cfg["banks"][1]["calibration_origin"]="target"
            with patch.object(cli,"load_labels",side_effect=AssertionError("No labels")),self.assertRaises(ValueError):cli.run_evaluation(cfg)

    def test_label_path_cannot_be_a_prelabel_seal_or_calibration_alias(self):
        for which in ('seal','CAL'):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as d:
                cfg,_=fixture(Path(d));labels=cfg['labels']['path']
                if which=='seal':cfg['raw_seal'].update(path=labels,sha256=cfg['labels']['sha256'])
                else:cfg['banks'][0]['calibration'].update(path=labels,sha256=cfg['labels']['sha256']);reseal(cfg)
                original=cli.sha
                def no_label_hash(path):
                    self.assertNotEqual(Path(path).resolve(),Path(labels).resolve(),'Label hash happened before bank gates')
                    return original(path)
                with patch.object(cli,'sha',side_effect=no_label_hash),self.assertRaises(PermissionError):cli.bank_gates(cfg)

    def test_signed_synthetic_metadata_preflight_and_source_change_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,_=fixture(root);cfg.update(operation=cli.OPERATION,
                permissions=dict(labels=True,evaluation=True,training=False,target_fit=False,GPU=False))
            base=Path(cli.__file__).resolve().parents[2]
            required=cli.REQUIRED_SOURCE|{'hybrid_tsad/hybrid_v1/named_evaluate_cli.py','hybrid_tsad/hybrid_v1/scoring.py'}
            cfg['source_sha256']={name:cli.sha(base/name) for name in required}
            cp=root/'config.json';cp.write_text(json.dumps(cfg))
            signed=dict(status='APPROVED',approved_by='root',operation=cli.OPERATION,allowed_actions=['evaluate_named_fixed_banks'],
                allow_labels=True,config_sha256=cli.sha(cp),source_sha256=cfg['source_sha256'],raw_seal_sha256=cfg['raw_seal']['sha256'])
            rp=root/'release.json';rp.write_text(json.dumps(signed))
            with patch.object(cli,'load_labels',side_effect=AssertionError('No labels in preflight')),contextlib.redirect_stdout(io.StringIO()) as log:
                cli.main(['--config',str(cp),'--release',str(rp),'--trusted-release-sha256',cli.sha(rp),'--execute','--preflight-only'])
            self.assertIn('"labels_opened": false',log.getvalue());self.assertFalse(Path(cfg['output']).exists())
            signed['source_sha256']['hybrid_tsad/hybrid_v1/statistics.py']='0'*64;rp.write_text(json.dumps(signed))
            with self.assertRaises(PermissionError):cli.admit(cp,rp,cli.sha(rp))

    def test_core_result_parity_all_metrics_inputs_and_event_pairs(self):
        from hybrid_tsad.hybrid_v1.evaluation import evaluate_bank_arrays
        with tempfile.TemporaryDirectory() as d:
            cfg,y=fixture(Path(d));expected={e["id"]:evaluate_bank_arrays(e["directory"],y,1.,expected_manifest_sha256=e["manifest_sha256"],original_length=18,
                warmup_points=2,stream=cfg["stream"]) for e in cfg["banks"]}
            summary=cli.run_evaluation(cfg);output=Path(cfg["output"])
            self.assertEqual(len(summary["methods"]),6);self.assertIsNone(summary["model_seed_mean"]);self.assertIsNone(summary["statistics"]["CI"])
            for e in cfg["banks"]:
                actual=json.loads((output/e["id"]/"result.json").read_text());self.assertEqual(actual,expected[e["id"]])
                self.assertTrue((output/e["id"]/"strict_point_aff_blocks.csv").exists())
                with np.load(output/e["id"]/"calculation_inputs.npz",allow_pickle=False) as a:np.testing.assert_array_equal(a["full_labels"],y)
            folder=output/"pairs/full_seed2025__vs__A1_fusion_none_seed2025"
            with (folder/"paired_events.csv").open(newline='') as f:rows=list(csv.DictReader(f))
            # One long alarm hits2 original events but can match only the first.
            row=next(x for x in rows if x['event_id']=='E000003' and x['stream']=='strict_point' and x['view']=='main_events')
            self.assertEqual(row['candidate_any_hit'],'True');self.assertEqual(row['candidate_one_to_one_matched'],'False');self.assertEqual(row['match_difference'],'-1')
            warmup=next(x for x in rows if x['event_id']=='E000001' and x['stream']=='strict_point' and x['view']=='main_events')
            self.assertEqual(warmup['joint_eligible'],'False');self.assertEqual(warmup['any_hit_difference'],'')
            with self.assertRaises(FileExistsError):cli.run_evaluation(cfg)

    def test_gapped_raw_ranges_and_missing_event_never_stitch(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,_=fixture(Path(d),missing={'A1_fusion_none_seed2025':(6,)},gap=True);cli.run_evaluation(cfg)
            result=json.loads((Path(cfg['output'])/'A1_fusion_none_seed2025/result.json').read_text())
            self.assertEqual(result['coverage']['missing_arrived_raw_records'],2)
            self.assertEqual(result['point_coverage'][8]['reason'],'NOT_IN_DECLARED_BANK_COVERAGE')
            self.assertEqual(result['point_coverage'][6]['reason'],'SYNTHETIC_MISSING')
            self.assertFalse(result['event_coverage'][1]['complete'])
            blocks=result['streams']['strict_point']['main_events']['blocks']
            self.assertTrue(all(not (b['start']<8 and b['end_exclusive']>10) for b in blocks))
            self.assertIsNone(result['streams']['strict_point']['affiliation']['global_result']['Aff_F1'])

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'SKLEARN_UNAVAILABLE_COMMON_AP_NOT_TESTED')
    def test_common_original_point_effect_is_not_score_ensemble(self):
        index=np.arange(4,dtype=np.int64);y=np.array([0,1,0,1],np.int8)
        a=dict(index=index,score=np.array([.1,.9,.2,.8]),valid=np.ones(4,bool));b=dict(index=index,score=np.array([.8,.2,.7,.1]),valid=np.ones(4,bool))
        result,common=cli.point_pair(a,b,y,0)
        self.assertEqual(common.tolist(),[0,1,2,3]);self.assertEqual(result['reference_AP'],1.)
        self.assertAlmostEqual(result['candidate_AP'],5/12,delta=1e-15);self.assertAlmostEqual(result['AP_difference_candidate_minus_reference'],-7/12,delta=1e-15)
        self.assertIsNone(result['CI']);self.assertFalse(result['primary_full_coverage_replaced'])

    def test_real_official_Aff_passes_through_and_gap_has_no_global(self):
        identity=os.environ.get('HYBRID_V1_AFFILIATION_IDENTITY')
        if not identity or not Path(identity).exists():self.skipTest('EXISTING_OFFICIAL_IDENTITY_NOT_BOUND')
        with tempfile.TemporaryDirectory() as d:
            cfg,_=fixture(Path(d),gap=True);cfg.update(affiliation_identity=identity,affiliation_identity_sha256=cli.sha(identity));summary=cli.run_evaluation(cfg)
            result=json.loads((Path(cfg['output'])/'full_seed2025/result.json').read_text())
            self.assertIsNotNone(result['affiliation_source'])
            aff=result['streams']['strict_point']['affiliation'];self.assertIsNone(aff['global_result']['Aff_F1'])
            self.assertTrue(any(b['Aff_F1'] is not None for b in aff['blocks']))


if __name__=='__main__':unittest.main()
