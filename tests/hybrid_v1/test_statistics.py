"""Original-grid paired sensitivity fixtures; genuine sklearn checks may skip."""
import copy
import importlib.util
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from hybrid_tsad.hybrid_v1.statistics import (
    BLOCK_POINTS,REPLICATES,RNG_SEED,DEFAULT_MODELS,DEFAULT_BASELINES,
    prepare_paired_inputs,fixed_block_draws,point_weights_from_draw,
    paired_ap_bootstrap,save_statistics,
)


def fixture(index=(0,1,4096,4097),length=4098):
    index=np.array(index,np.int64);labels=np.zeros(length,np.int8);labels[index[1::2]]=1
    vectors=[[0,3,1,2],[3,0,2,1],[1,1,1,1],[3,0,2,1],[1,1,1,1]]
    methods={name:dict(index=index.copy(),score=np.array(v,float),valid=np.ones(4,bool))
             for name,v in zip(DEFAULT_MODELS+DEFAULT_BASELINES,vectors)}
    return methods,labels


class Plans(unittest.TestCase):
    def test_original_grid_blocks_keep_gaps_and_empty_blocks(self):
        m,y=fixture(index=(100,101,8200,8201),length=9000)
        p=prepare_paired_inputs(m,y,original_length=9000)
        np.testing.assert_array_equal(p["common_index"],[100,101,8200,8201])
        np.testing.assert_array_equal(p["common_block_id"],[0,0,2,2])
        np.testing.assert_array_equal(p["block_bounds"],[[0,4096],[4096,8192],[8192,9000]])
        np.testing.assert_array_equal(p["per_block_common_count"],[2,0,2])
        weights=point_weights_from_draw(p["common_block_id"],np.array([1,1,1]),3)
        np.testing.assert_array_equal(weights,[0,0,0,0])

    def test_common_intersection_counts_and_exclusion_reasons(self):
        m,y=fixture()
        m[DEFAULT_MODELS[1]]["valid"][1]=False
        m[DEFAULT_MODELS[1]]["invalid_reasons"]=[None,"MISSING_CURRENT",None,None]
        row=m[DEFAULT_MODELS[2]]
        for key in ("index","score","valid"):row[key]=np.delete(row[key],2)
        m[DEFAULT_BASELINES[1]]["valid"][3]=False
        m[DEFAULT_BASELINES[1]]["score"][3]=np.nan
        m[DEFAULT_BASELINES[1]]["invalid_reasons"]=[None,None,None,"T_MINUS_1_MISSING"]
        p=prepare_paired_inputs(m,y,original_length=len(y))
        self.assertEqual(p["common_index"].tolist(),[0])
        c=p["coverage"]
        self.assertEqual((c["common_points"],c["common_positive_points"],c["common_negative_points"]),(1,0,1))
        self.assertEqual(sum(c["exclusion_patterns"].values()),4097)
        self.assertEqual(c["per_method"][DEFAULT_MODELS[1]]["invalid_raw_records"],1)
        self.assertEqual(c["per_method"][DEFAULT_MODELS[2]]["absent_original_points"],4095)
        self.assertTrue(c["does_not_replace_method_full_coverage_metrics"])

    def test_exact_fixed2000_draws_local_rng_and_shared_weight_example(self):
        before_py=random.getstate();before_np=np.random.get_state()
        a=fixed_block_draws(3);b=fixed_block_draws(3)
        self.assertEqual((BLOCK_POINTS,REPLICATES,RNG_SEED),(4096,2000,99173))
        self.assertEqual(a.shape,(2000,3));np.testing.assert_array_equal(a,b)
        w=point_weights_from_draw(np.array([0,0,2,2]),np.array([0,2,0]),3)
        np.testing.assert_array_equal(w,[2,2,1,1])
        self.assertEqual(before_py,random.getstate())
        after=np.random.get_state();self.assertEqual(before_np[0],after[0]);np.testing.assert_array_equal(before_np[1],after[1]);self.assertEqual(before_np[2:],after[2:])

    def test_no_input_mutation_or_aliasing_during_preparation(self):
        m,y=fixture();before=copy.deepcopy(m);y_before=y.copy()
        p=prepare_paired_inputs(m,y,original_length=len(y))
        for name in m:
            for key in ("index","score","valid"):np.testing.assert_array_equal(m[name][key],before[name][key])
        np.testing.assert_array_equal(y,y_before)
        p["normalized_inputs"][DEFAULT_MODELS[0]]["score"][0]=900
        self.assertEqual(m[DEFAULT_MODELS[0]]["score"][0],0.)

    def test_reject_wrong_label_axis_duplicate_unsorted_unsigned_and_valid_NaN(self):
        m,y=fixture()
        with self.assertRaises(ValueError):prepare_paired_inputs(m,y[[0,1,4096,4097]],original_length=len(y))
        for index in (np.array([0,1,1,4097]),np.array([0,4096,1,4097]),np.array([2,1,4096,4097],dtype=np.uint64)):
            wrong=copy.deepcopy(m);wrong[DEFAULT_MODELS[0]]["index"]=index
            with self.assertRaises(ValueError):prepare_paired_inputs(wrong,y,original_length=len(y))
        wrong=copy.deepcopy(m);wrong[DEFAULT_MODELS[0]]["score"][0]=np.nan
        with self.assertRaises(ValueError):prepare_paired_inputs(wrong,y,original_length=len(y))
        wrong=copy.deepcopy(m);wrong[DEFAULT_MODELS[0]]["valid"]=np.ones(4,np.int8)
        with self.assertRaises(ValueError):prepare_paired_inputs(wrong,y,original_length=len(y))

    def test_explicit_original_start_and_no_missing_method_or_reason(self):
        m,y=fixture()
        for row in m.values():row["index"]+=1000
        p=prepare_paired_inputs(m,y,original_length=len(y),original_start=1000)
        self.assertEqual(p["block_bounds"][0].tolist(),[1000,5096])
        self.assertEqual(p["common_block_id"].tolist(),[0,0,1,1])
        with self.assertRaises(ValueError):prepare_paired_inputs(m,y,original_length=len(y))
        m,y=fixture();del m[DEFAULT_MODELS[0]]
        with self.assertRaises(ValueError):prepare_paired_inputs(m,y,original_length=len(y))
        m,y=fixture();m[DEFAULT_MODELS[0]]["valid"][0]=False
        with self.assertRaises(ValueError):prepare_paired_inputs(m,y,original_length=len(y))


@unittest.skipUnless(importlib.util.find_spec("sklearn"),"SKLEARN_UNAVAILABLE_REAL_AP_BOOTSTRAP_NOT_TESTED")
class RealAP(unittest.TestCase):
    def test_hand_expected_AP_effects_no_score_ensemble_and_saved_draws(self):
        m,y=fixture();before=copy.deepcopy(m)
        result=paired_ap_bootstrap(m,y,original_length=len(y));s=result["summary"];a=result["arrays"]
        expected=[1.,5/12,.5,5/12,.5]
        np.testing.assert_allclose(a["original_AP"],expected,rtol=0,atol=1e-15)
        self.assertAlmostEqual(s["model_seed_AP"]["mean"],23/36)
        self.assertAlmostEqual(s["model_seed_AP"]["sample_SD"],math.sqrt(129)/36)
        self.assertAlmostEqual(s["model_seed_AP"]["minimum"],5/12,delta=1e-15)
        self.assertAlmostEqual(s["original_paired_AP_difference"]["persistence"]["mean"],2/9)
        self.assertAlmostEqual(s["original_paired_AP_difference"]["history_mean512"]["mean"],5/36)
        # Averaging three score vectors would be all ties/AP=.5, which differs.
        self.assertNotEqual(s["model_seed_AP"]["mean"],.5)
        np.testing.assert_array_equal(a["replicate_paired_difference"][:,1,0],np.zeros(2000))
        np.testing.assert_array_equal(a["replicate_paired_difference"][:,2,1],np.zeros(2000))
        same=a["block_draws"][:,0]==a["block_draws"][:,1]
        np.testing.assert_allclose(a["replicate_AP"][:,1],np.where(same,.5,5/12),rtol=0,atol=1e-15)
        self.assertEqual(s["draw_counts"]["VALID_TWO_CLASS_DRAW"],2000)
        interval=s["paired_difference_intervals"]["persistence"]["mean"]
        self.assertAlmostEqual(interval["lower"],1/6)
        self.assertAlmostEqual(interval["upper"],2/9)
        for name in m:
            for key in ("index","score","valid"):np.testing.assert_array_equal(m[name][key],before[name][key])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/"stats"
            receipt=save_statistics(out,result,provenance={"scope":"SYNTHETIC_ONLY"})
            self.assertTrue((out/"calculation_inputs.npz").exists())
            with np.load(out/"bootstrap_arrays.npz",allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["block_draws"],a["block_draws"])
                np.testing.assert_array_equal(arrays["replicate_AP"],a["replicate_AP"])
            self.assertEqual(receipt["parameters"],dict(block_points=4096,n_replicates=2000,rng_seed=99173))

    def test_single_class_and_empty_draws_remain_NA_without_redraw(self):
        index=np.array([0,4096]);y=np.zeros(8193,np.int8);y[4096]=1
        m={name:dict(index=index.copy(),score=np.array([0.,1.]),valid=np.ones(2,bool))
           for name in DEFAULT_MODELS+DEFAULT_BASELINES}
        r=paired_ap_bootstrap(m,y,original_length=len(y));a=r["arrays"]
        expected=[]
        for draw in a["block_draws"]:
            if not np.any(draw<2):expected.append(1)
            elif not np.any(draw==0) or not np.any(draw==1):expected.append(2)
            else:expected.append(0)
        np.testing.assert_array_equal(a["NA_reason_code"],expected)
        valid=np.array(expected)==0
        self.assertEqual(a["block_draws"].shape,(2000,3))
        self.assertTrue(np.isnan(a["replicate_AP"][~valid]).all())
        np.testing.assert_array_equal(a["replicate_AP"][valid],np.ones((valid.sum(),5)))
        self.assertGreater(r["summary"]["draw_counts"]["SINGLE_CLASS_DRAW"],0)
        self.assertGreater(r["summary"]["draw_counts"]["NO_COMMON_POINTS_IN_DRAW"],0)
        self.assertEqual(sum(r["summary"]["draw_counts"].values()),2000)


if __name__=="__main__":unittest.main()
