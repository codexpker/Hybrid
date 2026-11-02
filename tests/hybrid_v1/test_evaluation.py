"""Counterexample fixtures for independent fixed-bank CPU evaluation."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hybrid_tsad.hybrid_v1.evaluation import evaluate_arrays, evaluate_bank_arrays, save_evaluation
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank, IDENTITY_KEYS


def evaluate(y,s,index=None,**kwargs):
    y=np.asarray(y,dtype=np.int8); s=np.asarray(s,dtype=np.float64)
    index=np.arange(len(s),dtype=np.int64) if index is None else np.array(index,np.int64)
    return evaluate_arrays(index,s,y,1.,bank_complete=True,original_length=len(y),
                           warmup_points=kwargs.pop("warmup_points",0),**kwargs)


class Evaluation(unittest.TestCase):
    def test_warmup_through_event_does_not_create_new_event_or_normal_points(self):
        result=evaluate([0,1,1,1,0,0,1,1,0],[2,2,2,2,2,2],index=range(3,9),warmup_points=3)
        events=result["event_coverage"]
        self.assertEqual([(e["start"],e["end_exclusive"]) for e in events],[(1,4),(6,8)])
        self.assertEqual(events[0]["scored_points"],1)
        self.assertFalse(events[0]["main_eligible"])
        self.assertTrue(events[1]["main_eligible"])
        strict=result["streams"]["strict_point"]
        self.assertEqual(strict["point"]["TP"],3)
        self.assertEqual(strict["normal"]["normal_exposure_points"],3)
        self.assertEqual(strict["main_events"]["metrics"]["Ntrue"],1)
        self.assertEqual(strict["main_events"]["blocks"][0]["start"],4)
        self.assertEqual(result["point_coverage"][1]["reason"],"WARMUP")

    def test_adjacent_score_ranges_do_not_repair_missing_positive_endpoint(self):
        result=evaluate([0,0,1,1,1,0,0],[2,2,2],index=[2,3,5])
        event=result["event_coverage"][0]
        self.assertEqual(event["coverage_ratio"],2/3)
        self.assertFalse(event["complete"])
        strict=result["streams"]["strict_point"]
        self.assertEqual(strict["main_events"]["metrics"]["Ntrue"],0)
        self.assertEqual(strict["normal"]["normal_exposure_points"],1)
        self.assertEqual(strict["normal"]["normal_false_alarm_points"],1)

    def test_confirm3_resets_at_original_gap_not_compressed_third_point(self):
        result=evaluate(np.zeros(8),[2]*5,index=[1,2,4,5,6])
        confirmed=[r["index"] for r in result["point_coverage"] if r["confirmed3_current"] is True]
        self.assertEqual(confirmed,[6])
        self.assertIsNone(result["point_coverage"][3]["confirmed3_current"])

    def test_invalid_score_is_NA_not_TN_and_resets_confirmation(self):
        result=evaluate(np.zeros(7),[2,2,np.nan,2,2,2],index=[1,2,3,4,5,6],
                        valid=np.array([True,True,False,True,True,True]),
                        invalid_reasons=[None,None,"NO_CHANNELS",None,None,None])
        self.assertEqual(result["streams"]["strict_point"]["point"]["TN"],0)
        self.assertEqual(result["streams"]["strict_point"]["normal"]["normal_exposure_points"],5)
        self.assertEqual(result["point_coverage"][3]["reason"],"NO_CHANNELS")
        self.assertEqual([r["index"] for r in result["point_coverage"] if r["confirmed3_current"] is True],[6])

    def test_boundaries_are_censored_separately_from_complete_coverage(self):
        result=evaluate([1,1,0,0,1,1,0,0,1,1],[2]*10)
        main=result["streams"]["strict_point"]["main_events"]
        legacy=result["streams"]["strict_point"]["legacy_complete_only_events"]
        self.assertEqual(result["coverage"]["boundary_censored_events"],2)
        self.assertEqual(main["metrics"]["Ntrue"],1)
        self.assertEqual(legacy["metrics"]["Ntrue"],3)
        self.assertIsNone(legacy["events"][0]["delay_points"])
        self.assertEqual(legacy["events"][0]["delay_NA_reason"],"UNKNOWN_ORIGINAL_EVENT_BOUNDARY")

    def test_one_long_alarm_covers_two_events_but_matches_only_one(self):
        result=evaluate([0,0,1,1,0,0,1,1,0,0],[0,2,2,2,2,2,2,2,2,0])
        m=result["streams"]["strict_point"]["main_events"]["metrics"]
        self.assertEqual((m["M"],m["Npred"],m["Ntrue"]),(1,1,2))
        self.assertEqual(m["auxiliary_any_hit_recall"],1.)
        self.assertEqual(m["one_to_one_eventRecall"],.5)
        self.assertEqual(m["one_to_one_eventPrecision"],1.)
        self.assertAlmostEqual(m["one_to_one_eventF1"],2/3)

    def test_repeated_alarms_reduce_matching_precision_not_truth_count(self):
        y=np.zeros(12); y[3:8]=1
        score=np.zeros(12); score[[3,6]]=2
        m=evaluate(y,score)["streams"]["strict_point"]["main_events"]["metrics"]
        self.assertEqual((m["M"],m["Npred"],m["Ntrue"]),(1,2,1))
        self.assertEqual(m["one_to_one_eventPrecision"],.5)
        self.assertEqual(m["one_to_one_eventRecall"],1.)
        self.assertAlmostEqual(m["one_to_one_eventF1"],2/3)

    def test_confirmation_delay_current_third_point_and_miss_censor(self):
        y=np.zeros(15);y[5:10]=1;y[12:14]=1
        score=np.zeros(15);score[4:8]=2
        result=evaluate(y,score)
        a=result["streams"]["strict_point"]["main_events"]["events"]
        b=result["streams"]["confirmed3_current"]["main_events"]["events"]
        self.assertEqual(a[0]["delay_points"],0)
        self.assertEqual(b[0]["delay_points"],1)  # 4,5,6 confirm at6, event begins5
        self.assertEqual(b[0]["first_available_alarm_index"],6)
        self.assertTrue(b[1]["delay_right_censored"])
        self.assertEqual(b[1]["censor_after_points"],2)
        self.assertIsNone(b[1]["delay_points"])

    def test_normal_fragments_clipped_by_anomaly_are_not_new_onsets(self):
        y=np.zeros(10); y[3:6]=1
        score=np.zeros(10);score[1:9]=2
        n=evaluate(y,score)["streams"]["strict_point"]["normal"]
        self.assertEqual(n["normal_exposure_points"],7)
        self.assertEqual(n["normal_false_alarm_points"],5)
        self.assertEqual(n["normal_alarm_fragments"],2)
        self.assertEqual(n["anomaly_boundary_clipped_normal_alarm_fragments"],2)
        self.assertAlmostEqual(n["normal_alarm_fragments_per_1000_points"],2000/7)
        self.assertEqual(n["normal_block_any_alarm_fraction"],1.)
        self.assertEqual(n["normal_block_lengths"],[3,4])

    def test_single_class_no_prediction_zero_denominators_and_fixed_stats_NA(self):
        r=evaluate(np.zeros(6),np.zeros(6))
        self.assertIsNone(r["ranking"]["AP"])
        self.assertEqual(r["ranking"]["ranking_metric_NA_reason"],"SINGLE_CLASS_SCORED_LABELS")
        p=r["streams"]["strict_point"]["point"]
        self.assertIsNone(p["Precision"]);self.assertIsNone(p["Recall"]);self.assertIsNone(p["F1"])
        y=[0,1,1,0]
        s=evaluate(y,[0]*4)["streams"]["strict_point"]
        self.assertIsNone(s["main_events"]["metrics"]["one_to_one_eventPrecision"])
        self.assertEqual(s["main_events"]["metrics"]["one_to_one_eventRecall"],0.)
        self.assertEqual(s["main_events"]["metrics"]["one_to_one_eventF1"],0.)
        self.assertTrue(s["main_events"]["events"][0]["delay_right_censored"])
        self.assertEqual(r["statistics"]["status"],"NOT_COMPUTED")
        self.assertEqual((r["statistics"]["n_replicates"],r["statistics"]["block_points"],r["statistics"]["rng_seed"]),(2000,4096,99173))

    def test_aff_and_VUS_keep_real_blocks_no_cross_gap_global(self):
        y=np.zeros(12);y[1:5]=1;y[7:9]=1
        result=evaluate(y,np.full(11,2.),index=[0,1,2,4,5,6,7,8,9,10,11])
        aff=result["streams"]["strict_point"]["affiliation"]
        self.assertEqual([(b["start"],b["end_exclusive"]) for b in aff["blocks"]],[(0,1),(5,12)])
        self.assertIsNone(aff["global_result"]["Aff_F1"])
        self.assertEqual(aff["global_result"]["reason"],"MULTIPLE_REAL_BLOCKS_NO_GLOBAL_AGGREGATION")
        vus=result["streams"]["strict_point"]["Range_VUS"]
        self.assertEqual([(b["start"],b["end_exclusive"]) for b in vus["blocks"]],[(0,1),(5,12)])
        self.assertEqual(vus["status"],"NOT_EVALUATED")
        self.assertIsNone(vus["global_result"])

    def test_reject_incomplete_bank_bad_labels_indices_clock_and_NA_reason(self):
        args=(np.array([0,1]),np.array([0.,1.]),np.array([0,1]),1.)
        base=dict(bank_complete=True,original_length=2,warmup_points=0)
        for change in (dict(bank_complete=False),dict(original_length=3),dict(available_index=np.array([0,2]))):
            with self.assertRaises(ValueError):evaluate_arrays(*args,**{**base,**change})
        with self.assertRaises(ValueError):evaluate([0,1],[0,1],index=[1,1])
        with self.assertRaises(ValueError):evaluate([0,2],[0,1])
        with self.assertRaises(ValueError):evaluate([0,1],[0,np.nan])

    def test_empty_valid_stream_retains_original_events_and_no_fake_normal(self):
        r=evaluate([0,1,1,0],[],index=[])
        self.assertEqual(r["ranking"]["ranking_metric_NA_reason"],"NO_VALID_SCORES")
        self.assertEqual(r["event_coverage"][0]["coverage_ratio"],0.)
        self.assertEqual(r["streams"]["strict_point"]["normal"]["normal_exposure_points"],0)
        self.assertIsNone(r["streams"]["strict_point"]["normal"]["normal_FPR"])

    def test_long_missing_reason_preserved_and_missing_tail_disclosed(self):
        reason="NO_CHANNELS_"+"x"*400
        r=evaluate([0,1,1,0,0],[np.nan,2],index=[1,2],valid=np.array([False,True]),
                   invalid_reasons=[reason,None],warmup_points=1)
        self.assertEqual(r["point_coverage"][1]["reason"],reason)
        self.assertFalse(r["coverage"]["complete_arrived_raw_coverage"])
        self.assertEqual(r["coverage"]["missing_arrived_raw_records"],2)
        self.assertFalse(r["coverage"]["final_original_point_has_raw_record"])

    def test_original_start_is_explicit_not_offset_repaired(self):
        r=evaluate_arrays(np.array([101,102]),[2,2],np.array([0,1,1,0]),1.,
            bank_complete=True,original_length=4,original_start=100,warmup_points=1)
        self.assertEqual(r["event_coverage"][0]["start"],101)
        self.assertEqual(r["point_coverage"][0]["index"],100)
        self.assertEqual(r["streams"]["strict_point"]["main_events"]["events"][0]["delay_points"],0)

    def test_complete_bank_manifest_indices_threshold_and_saved_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"bank"
            ident={k:"a"*64 for k in IDENTITY_KEYS}
            ranges=[dict(domain="synthetic",entity="fixture",run="r",start=1,end=7)]
            with ChunkedBank(path,identity=ident,ranges=ranges,threshold=1.,chunk_size=2) as bank:
                bank.write(dict(domain="synthetic",entity="fixture",run="r",index=i,
                    available_index=i,score=2. if i in (3,4) else 0.,valid=True) for i in range(1,7))
                with self.assertRaises(ValueError):
                    evaluate_bank_arrays(path,[0,0,0,1,1,0,0],1.,expected_manifest_sha256=hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest(),original_length=7,warmup_points=1)
                bank.finalize()
            sha=hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest()
            y=np.array([0,0,0,1,1,0,0],np.int8)
            r=evaluate_bank_arrays(path,y,1.,expected_manifest_sha256=sha,original_length=7,warmup_points=1)
            self.assertTrue(r["bank_identity"]["complete"])
            self.assertEqual(r["streams"]["strict_point"]["main_events"]["metrics"]["M"],1)
            index=np.arange(1,7,dtype=np.int64);score=np.array([0,0,2,2,0,0],float);valid=np.ones(6,bool)
            receipt=save_evaluation(Path(temporary)/"evaluation",r,index=index,score=score,valid=valid,
                available_index=index,full_labels=y,provenance={"scope":"synthetic_fixture_only"})
            self.assertEqual(len(receipt["code"]),6)
            self.assertTrue((Path(temporary)/"evaluation/strict_point_main_events_events.csv").exists())
            with np.load(Path(temporary)/"evaluation/calculation_inputs.npz",allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["full_labels"],y)
            with self.assertRaises(ValueError):
                evaluate_bank_arrays(path,y,2.,expected_manifest_sha256=sha,original_length=7,warmup_points=1)
            with self.assertRaises(ValueError):
                evaluate_bank_arrays(path,y,1.,expected_manifest_sha256="b"*64,original_length=7,warmup_points=1)
            with self.assertRaises(ValueError):
                save_evaluation(Path(temporary)/"wrong",r,index=index,score=score+1,valid=valid,
                    available_index=index,full_labels=y,provenance={})
            with self.assertRaises(ValueError):
                save_evaluation(Path(temporary)/"wrong_clock",r,index=index,score=score,valid=valid,
                    available_index=index+1,full_labels=y,provenance={})
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["committed_records"]=7
            (path/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            wrong_sha=hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                evaluate_bank_arrays(path,y,1.,expected_manifest_sha256=wrong_sha,original_length=7,warmup_points=1)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"SKLEARN_UNAVAILABLE_RANKING_PARITY_NOT_TESTED")
    def test_real_shared_sklearn_AP_trapezoid_and_ROC_are_separate(self):
        r=evaluate([0,1],[2,0])
        self.assertEqual(r["ranking"]["AP"],.5)
        self.assertEqual(r["ranking"]["trapezoid_PR_AUC"],.25)
        self.assertEqual(r["ranking"]["ROC_AUC"],0.)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"SKLEARN_UNAVAILABLE_OFFICIAL_AFF_NOT_TESTED")
    def test_real_official_affiliation_exact_hit_and_no_global_gap_merge(self):
        root=Path(__file__).resolve().parents[3]
        identity=Path(os.environ.get("HYBRID_V1_AFFILIATION_IDENTITY",
            str(root/"preparations/candidate_psm_fixed_affiliation_20261102_v1/LOCAL_IDENTITY.json")))
        if not identity.exists():self.skipTest("EXISTING_OFFICIAL_IDENTITY_NOT_PRESENT")
        r=evaluate([0,0,1,1,0,0],[0,0,2,2,0,0],affiliation_identity=identity)
        a=r["streams"]["strict_point"]["affiliation"]
        self.assertEqual(a["global_result"]["Aff_F1"],1.)
        self.assertIsNotNone(r["affiliation_source"])
        y=np.zeros(10);y[1:3]=1;y[7:9]=1
        index=np.array([0,1,2,3,5,6,7,8,9])
        score=np.where(y[index]==1,2.,0.)
        gap=evaluate(y,score,index=index,affiliation_identity=identity)
        a=gap["streams"]["strict_point"]["affiliation"]
        self.assertEqual([(b["start"],b["end_exclusive"]) for b in a["blocks"]],[(0,4),(5,10)])
        self.assertEqual([b["Aff_F1"] for b in a["blocks"]],[1.,1.])
        self.assertIsNone(a["global_result"]["Aff_F1"])


if __name__=="__main__":unittest.main()
