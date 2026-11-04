"""Small independent array fixtures; no formal inputs, labels, model or Torch."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hybrid_tsad.hybrid_v1 import controlled_wave_evaluation as entry
from hybrid_tsad.hybrid_v1 import evaluation


def cal(active=("r_pred",)):
    _,weights=entry.scoring.recipe(active)
    return dict(active_components=list(active),weights=weights,scales={k:1. for k in active},
                threshold=1.,source_domain_equal_weight=True,domains=["SMD","Sunspot"])


def raw(i=2):
    row=dict(domain=entry.wave.DOMAIN,entity="fixture",run="case",index=i,available_index=i,r_pred=2.)
    row.update(entry.scoring.apply_calibration(row,cal()))
    return row


class RecordTests(unittest.TestCase):
    def test_valid_and_reason_tampering_rejected(self):
        for key,value in (("valid",False),("valid",1),("reason","WRONG"),("alarm",1),("score",12.)):
            r=raw();r[key]=value
            with self.assertRaises(ValueError):entry.scalar_record(r,cal())

    def test_inactive_TF_nan_not_imputed(self):
        c=cal(("r_ms","r_pred"));r=raw();r.update(r_ms=2.,r_tf=float("nan"),r_var=float("nan"))
        r.update(entry.scoring.apply_calibration(r,c))
        self.assertTrue(entry.scalar_record(r,c)[2]);self.assertTrue(np.isnan(r["r_tf"]))

    def test_invalid_preserved_and_never_false_alarm_decision(self):
        r=raw();r["r_pred"]=float("nan");r.update(entry.scoring.apply_calibration(r,cal()));r.pop("alarm")
        self.assertFalse(entry.scalar_record(r,cal())[2]);r["alarm"]=False
        with self.assertRaises(ValueError):entry.scalar_record(r,cal())

    def test_full_axis_truth_rejects_shift_or_trim(self):
        p=dict(N=10,cases=[dict(id="a",event=[3,6]),dict(id="n")])
        labels=dict(a=np.array([0,0,0,1,1,1,0,0,0,0],np.int8),n=np.zeros(10,np.int8))
        entry.validate_truth(labels,p)
        for bad in (dict(labels,a=labels["a"][2:]),dict(labels,a=np.roll(labels["a"],1))):
            with self.assertRaises(ValueError):entry.validate_truth(bad,p)


class CoverageTests(unittest.TestCase):
    def evaluate(self,index,score,labels):
        return evaluation.evaluate_arrays(np.asarray(index,np.int64),np.asarray(score),np.asarray(labels,np.int8),1.,
            bank_complete=True,original_length=len(labels),warmup_points=2)

    def test_all_normal_NA_and_FP_zero_rules(self):
        no=self.evaluate(range(2,10),np.zeros(8),np.zeros(10))
        yes=self.evaluate(range(2,10),np.full(8,2.),np.zeros(10))
        self.assertIsNone(no["ranking"]["AP"])
        self.assertIsNone(no["streams"]["strict_point"]["point"]["F1"])
        self.assertEqual(yes["streams"]["strict_point"]["point"]["F1"],0.)
        self.assertEqual(no["streams"]["strict_point"]["normal"]["normal_FPR"],0.)
        self.assertEqual(yes["event_coverage"],[])

    def test_real_gap_resets_confirmation_not_concatenated(self):
        result=self.evaluate([2,3,5,6,7],[2.]*5,np.zeros(10))
        got={r["index"]:r["confirmed3_current"] for r in result["point_coverage"]}
        self.assertFalse(got[3]);self.assertFalse(got[5]);self.assertTrue(got[7]);self.assertIsNone(got[4])

    def test_phase_diagnostics_do_not_split_original_event(self):
        labels=np.array([0,0,0,1,1,1,1,0,0,0],np.int8)
        result=self.evaluate(range(2,10),[0,2,2,2,2,0,0,0],labels)
        self.assertEqual(len(result["event_coverage"]),1)
        phases=entry.phase_counts(result,dict(onset=[3,5],steady=[5,7],empty=None))
        self.assertEqual(phases["steady"]["confirmed3_alarm_points"],2)
        self.assertIsNone(phases["empty"])
        self.assertEqual(len(result["streams"]["confirmed3_current"]["main_events"]["events"]),1)

    def test_multibank_prelabel_gate_missing_record(self):
        p=dict(N=5,history=2,cases=[dict(id="case",family="fixture",models=["full2025"],matched_interval=[3,4])],
            shams=[],semantic_identical_input_counterexamples=[])
        ds=[dict(case="case",method=m,directory=m,manifest_sha256="fixture",identity=dict(calibration_sha256="cal"),
                 calibration=dict(path="cal",sha256="cal"),active_components=["r_pred"])
            for m in ["full2025"]+list(entry.wave.METHODS)]
        cfg=dict(banks=ds,raw_seal=dict(path="seal",sha256="seal"))
        def bound(spec):
            if spec["path"]=="seal":return dict(status="ACCEPTED_CONTROLLED_WAVE1_FIXED_BANKS",approved_by="root",banks=ds)
            if spec["path"]=="cal":return cal()
            return dict(complete=True,committed_records=3,contract=dict(threshold=1.,identity=ds[0]["identity"],
                ranges=[dict(domain=entry.wave.DOMAIN,entity="fixture",run="case",start=2,end=5)]))
        with patch.object(entry.wave,"bound",side_effect=bound),patch.object(entry.chunked_bank,"read_records",return_value=iter([raw(2),raw(3)])):
            with self.assertRaises(ValueError):entry.prelabel_banks(cfg,p)


if __name__=="__main__":unittest.main()
