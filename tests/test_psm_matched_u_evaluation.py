import sys
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1 import psm_matched_u_evaluation as glue
from hybrid_tsad.hybrid_v1 import evaluation

class PSMEvaluationTests(unittest.TestCase):
    def test_unknown_only_and_partial_steady_are_not_missed_events(self):
        base=dict(eligible=True,scored_points=8349,unknown_points=0,detected_points=0)
        rows=[base,dict(base,eligible=False,scored_points=0,unknown_points=9000),
              dict(base,eligible=False,scored_points=20,unknown_points=30),
              dict(base,scored_points=0,unknown_points=0),dict(base,detected_points=1)]
        self.assertEqual(glue.complete_steady_all_missed(rows),1)

    def test_actual_typed_scalar_bank_gap_and_invalid_dtype(self):
        from hybrid_tsad.hybrid_v1.chunked_bank import _pack
        c=dict(active_components=['r_tf','r_ms','r_pred'],weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),
               scales=dict(r_tf=1.,r_ms=2.,r_pred=4.),threshold=1.,valid_windows=512,source_domain_equal_weight=False)
        records=[]
        for i in (512,1536):
            raw=dict(domain='PSM',entity='anonymous_PSM',run='test_original',index=i,available_index=i,
                     r_tf=1.,r_ms=2. if i==512 else None,r_pred=4.)
            raw.update(glue.scoring.apply_calibration(raw,c));records.append(dict(raw=raw))
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);chunk=folder/'chunk.npz'
            with chunk.open('wb') as f:np.savez_compressed(f,**_pack(records))
            manifest=folder/'manifest.json';manifest.write_text(json.dumps(dict(chunks=[dict(file='chunk.npz',sha256=glue.sha(chunk),start_position=0,records=2)])))
            d=dict(manifest=dict(path=str(manifest),sha256=glue.sha(manifest)))
            a=glue.raw_bank_scalars(d,np.array([512,1536]),c)
            self.assertTrue(a['valid'][0]);self.assertFalse(a['valid'][1]);self.assertTrue(np.isnan(a['score'][1]))
            self.assertEqual(a['score'].dtype,np.dtype('float64'));self.assertEqual(a['index'].dtype,np.dtype('int64'))

    def test_fixed_full_index_geometry_and_no_shift(self):
        index=np.r_[np.arange(512,1024),np.arange(2097,4956),np.arange(5314,5826),
                    np.arange(8149,8661),np.arange(10524,11036),np.arange(17479,27364)]
        p=dict(history=512,horizon=32,seed=2025,QUERY=[dict(domain='PSM',entity='anonymous_PSM',run='test_original',role='QUERY',index=int(i),footprint=[int(i)-512,int(i)+1]) for i in index])
        np.testing.assert_array_equal(glue.query_index(p),index)
        self.assertEqual(len(glue.index_ranges(index)),6)
        p['QUERY'][3]['footprint'][1]+=31
        with self.assertRaises(PermissionError):glue.query_index(p)

    def test_fixed_ZT_scales_and_tau_not_own_tau(self):
        c=dict(active_components=['r_tf','r_ms','r_pred'],weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),
               scales=dict(r_tf=1.,r_ms=2.,r_pred=4.),threshold=99.,valid_windows=512,source_domain_equal_weight=False)
        glue.validate_calibration(c)
        score,valid=glue.score_components(np.array([[1.,2.,4.],[1.,np.nan,4.]]),c)
        self.assertTrue(valid[0]);self.assertFalse(valid[1]);self.assertTrue(np.isnan(score[1]))
        self.assertAlmostEqual(score[0],np.log(2.))
        c['valid_windows']=1018
        with self.assertRaises(PermissionError):glue.validate_calibration(c)

    def test_original_axis_gap_and_partial_event_not_normal(self):
        ix=np.array([0,3,4,8,9,10]);labels=np.zeros(14,int);labels[3:5]=1;labels[8:11]=1
        r=evaluation.evaluate_arrays(index=ix,score=np.ones(len(ix))*2,valid=np.ones(len(ix),bool),
            available_index=ix,invalid_reasons=[None]*len(ix),full_labels=labels,threshold=1.,
            bank_complete=True,original_length=14,warmup_points=0)
        self.assertEqual(r['streams']['confirmed3_current']['point']['TP'],1)
        self.assertFalse(r['point_coverage'][8]['confirmed3_current'])
        labels[4:9]=1
        r=evaluation.evaluate_arrays(index=ix,score=np.ones(len(ix))*2,valid=np.ones(len(ix),bool),
            available_index=ix,invalid_reasons=[None]*len(ix),full_labels=labels,threshold=1.,
            bank_complete=True,original_length=14,warmup_points=0)
        self.assertFalse(r['event_coverage'][0]['main_eligible'])
        self.assertGreater(r['event_coverage'][0]['total_points'],r['event_coverage'][0]['scored_points'])
        self.assertFalse(r['point_coverage'][6]['valid'])

    def test_pending_rejected_before_array_or_truth(self):
        with patch.object(np,'load',side_effect=AssertionError('No early arrays')):
            with self.assertRaises(PermissionError):glue.run(dict(status='ROOT_PENDING',seed=2025))

if __name__=='__main__':unittest.main(verbosity=2)
