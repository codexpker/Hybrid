"""Pure synthetic NumPy controls; no Torch, files, source/query data or labels."""
import copy
import unittest

import numpy as np

from hybrid_tsad.hybrid_v1.baselines import (
    METHODS, baseline_record, iter_baselines, fit_source_baseline_calibration,
)
from hybrid_tsad.hybrid_v1.chunked_bank import _fingerprint
from hybrid_tsad.hybrid_v1.data import Run
from hybrid_tsad.hybrid_v1.scoring import fit_calibration, apply_calibration


def run(values, observed=None, *, domain="A", role="QUERY", start=1000):
    values = np.array(values,dtype=np.float32)
    if values.ndim == 1:
        values=values[:,None]
    observed = np.isfinite(values) if observed is None else np.array(observed,bool)
    return Run(domain,"fixture","r",role,values,observed,start=start)


class Baselines(unittest.TestCase):
    def test_hand_computed_population_variance_and_residual(self):
        source=run([1,2,3,4,5])
        for method, prediction, error in (("persistence",4.,1.),("history_mean512",2.5,6.25)):
            row=baseline_record(source,4,method=method,history=4)
            self.assertEqual(row["prediction"],[[prediction]])
            self.assertEqual(row["history_variance"],[1.25])
            self.assertAlmostEqual(row["r_pred"],error/(1.25+1e-5),places=12)
            self.assertEqual(row["index"],1004)
            self.assertEqual(row["available_index"],1004)
            self.assertEqual(row["forecast_issued_index"],1003)
            self.assertEqual(row["forecast_history_bounds"],[1000,1004])

    def test_future_mutation_no_effect_and_history_boundary_strict(self):
        original=run(np.arange(10,dtype=np.float32))
        changed=copy.deepcopy(original); changed.values[5:]=np.nan
        for method in METHODS:
            before=baseline_record(original,4,method=method,history=4)
            after=baseline_record(changed,4,method=method,history=4)
            self.assertEqual(_fingerprint(before),_fingerprint(after))
        first=run([900,1,2,3,4,5]); second=run([-900,1,2,3,4,5])
        for method in METHODS:
            self.assertEqual(_fingerprint(baseline_record(first,5,method=method,history=4)),
                             _fingerprint(baseline_record(second,5,method=method,history=4)))

    def test_current_not_in_prediction_statistics(self):
        first=run([1,2,3,4,5]); second=run([1,2,3,4,500])
        for method in METHODS:
            a=baseline_record(first,4,method=method,history=4)
            b=baseline_record(second,4,method=method,history=4)
            for key in ("prediction","history_mean","history_variance","history_scale"):
                self.assertEqual(a[key],b[key])
            self.assertNotEqual(a["r_pred"],b["r_pred"])

    def test_missing_tminus1_not_backfilled_and_valid_intersection(self):
        source=run([[1,np.nan,1],[2,np.nan,np.nan],[3,4,3],[np.nan,np.nan,4],[5,6,np.nan]])
        p=baseline_record(source,4,method="persistence",history=4)
        m=baseline_record(source,4,method="history_mean512",history=4)
        self.assertEqual(p["history_observed_count"],[3,1,3])
        self.assertFalse(p["prediction_valid"][0])
        self.assertTrue(np.isnan(p["prediction"][0][0]))
        self.assertEqual(p["valid_channel_count"],0)
        self.assertTrue(np.isnan(p["r_pred"]))
        self.assertEqual(m["valid_channels"],[True,False,False])
        self.assertEqual(m["valid_channel_count"],1)
        self.assertAlmostEqual(m["r_pred"],9/(2/3+1e-5))
        self.assertTrue(np.isnan(m["per_channel"]["r_pred"][2]))

    def test_missing_placeholders_ignored_no_imputed_GT(self):
        first=run([1,np.nan,3,4,5])
        second=copy.deepcopy(first); second.values[1]=1e20
        for method in METHODS:
            self.assertEqual(_fingerprint(baseline_record(first,4,method=method,history=4)),
                             _fingerprint(baseline_record(second,4,method=method,history=4)))
            missing=copy.deepcopy(first); missing.values[4]=np.nan; missing.observed[4]=False
            row=baseline_record(missing,4,method=method,history=4)
            self.assertEqual(row["valid_channel_count"],0)
            self.assertTrue(np.isnan(row["r_pred"]))
            self.assertFalse(row["valid"])

    def test_constant_history_zero_variance_epsilon_finite(self):
        for method in METHODS:
            row=baseline_record(run([2,2,2,2,3]),4,method=method,history=4)
            self.assertEqual(row["history_variance"],[0.])
            self.assertAlmostEqual(row["r_pred"],100000.,places=8)
            zero=baseline_record(run([2,2,2,2,2]),4,method=method,history=4)
            self.assertEqual(zero["r_pred"],0.)

    def test_variable_channels_no_channel_weight_by_missing_count(self):
        for channels in (1,3,25):
            source=run(np.tile(np.arange(5)[:,None],(1,channels)))
            for method in METHODS:
                row=baseline_record(source,4,method=method,history=4)
                self.assertEqual(row["valid_channel_count"],channels)
                self.assertEqual(len(row["per_channel"]["r_pred"]),channels)
                self.assertEqual(row["r_pred"],np.mean(row["per_channel"]["r_pred"]))

    def test_unequal_history_counts_do_not_weight_channel_pool(self):
        source=run([[1,1],[2,np.nan],[3,3],[4,4],[5,6]])
        row=baseline_record(source,4,method="persistence",history=4)
        expected=[1/(1.25+1e-5),4/(14/9+1e-5)]
        np.testing.assert_allclose(row["per_channel"]["r_pred"],expected,rtol=0,atol=1e-12)
        self.assertAlmostEqual(row["r_pred"],sum(expected)/2,places=12)
        self.assertNotAlmostEqual(row["r_pred"],(4*expected[0]+3*expected[1])/7)

    def test_full_H512_coverage_preserves_last31(self):
        for method in METHODS:
            rows=list(iter_baselines([run(np.arange(552))],method=method))
            self.assertEqual([r["index"] for r in rows],list(range(1512,1552)))
            self.assertEqual([r["index"] for r in rows[-31:]],list(range(1521,1552)))
            self.assertEqual(rows[-1]["footprint"],[1039,1552])
            self.assertEqual(rows[-1]["scored_horizon"],1)

    def test_exact_fixed_CAL_plan_and_existing_domain_weighted_recipe(self):
        runs=[run(np.arange(10),domain=d,role="CAL") for d in ("A","B")]
        plan=[dict(domain=d,entity="fixture",run="r",role="CAL",index=i)
              for d in ("A","B") for i in (1004,1008)]
        for method in METHODS:
            records=list(iter_baselines(runs,method=method,history=4,index_plan=plan))
            self.assertEqual([(r["domain"],r["index"]) for r in records],[(e["domain"],e["index"]) for e in plan])
            actual=fit_source_baseline_calibration(records,expected_domains=["A","B"])
            expected=fit_calibration(records,source=True,active_components=["r_pred"])
            self.assertEqual(actual,expected)
            self.assertEqual(actual["weights"],{"r_pred":1.})
            self.assertTrue(actual["source_domain_equal_weight"])
            scored=apply_calibration(records[0],actual)
            self.assertAlmostEqual(scored["score"],np.log1p(records[0]["r_pred"]/actual["scales"]["r_pred"]))
            self.assertFalse(scored["alarm"])  # all tied; strict > not >=

    def test_source_calibration_rejects_query_domains_methods_and_all_invalid(self):
        valid=baseline_record(run([1,2,3,4,5],role="CAL"),4,method="persistence",history=4)
        with self.assertRaises(ValueError):
            fit_source_baseline_calibration([{**valid,"role":"QUERY"}],expected_domains=["A"])
        with self.assertRaises(ValueError):
            fit_source_baseline_calibration([valid],expected_domains=["A","B"])
        with self.assertRaises(ValueError):
            fit_source_baseline_calibration([valid,{**valid,"method":"history_mean512"}],expected_domains=["A"])
        with self.assertRaises(ValueError):
            fit_source_baseline_calibration([{**valid,"r_pred":np.nan}],expected_domains=["A"])

    def test_invalid_bounds_short_run_and_observed_nonfinite_refused(self):
        source=run([1,2,3,4,5])
        self.assertEqual(list(iter_baselines([source],method="persistence")),[])
        for target in (3,5):
            with self.assertRaises(ValueError):
                baseline_record(source,target,method="persistence",history=4)
        bad=copy.deepcopy(source); bad.values[1]=np.nan
        with self.assertRaises(ValueError):
            baseline_record(bad,4,method="persistence",history=4)
        with self.assertRaises(ValueError):
            baseline_record(source,4,method="backfill",history=4)


if __name__=="__main__":
    unittest.main()
