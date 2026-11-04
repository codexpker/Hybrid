"""E1-only import/NA fixtures. No real metric imports, labels or old test reruns."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from hybrid_tsad.hybrid_v1 import controlled_wave_evaluation as entry


class DependencyTests(unittest.TestCase):
    def test_top_package_findable_metrics_missing_blocks_before_truth(self):
        with patch.object(entry.importlib.util,"find_spec",return_value=object()) as spec, \
                patch.object(entry.importlib,"import_module",side_effect=ModuleNotFoundError("metrics missing")) as imp, \
                patch.object(entry.np,"load") as labels,patch.object(entry,"prelabel_banks") as banks:
            self.assertIsNotNone(entry.importlib.util.find_spec("sklearn"))
            with self.assertRaises(PermissionError):entry.run({}, {})
            imp.assert_called_once_with("sklearn.metrics");labels.assert_not_called();banks.assert_not_called()

    def test_subdependency_import_failure_blocks_before_truth(self):
        exc=ModuleNotFoundError("No module named sklearn._missing_dependency")
        with patch.object(entry.importlib,"import_module",side_effect=exc),patch.object(entry.np,"load") as labels:
            with self.assertRaises(PermissionError):entry.run({}, {})
            labels.assert_not_called()

    def test_four_callables_imported_but_never_called(self):
        callbacks={name:Mock() for name in ("average_precision_score","roc_auc_score","precision_recall_curve","auc")}
        with patch.object(entry.importlib,"import_module",return_value=SimpleNamespace(**callbacks)) as imp:
            entry.require_metric_dependencies();imp.assert_called_once_with("sklearn.metrics")
        for callback in callbacks.values():callback.assert_not_called()

    def test_missing_or_noncallable_metric_rejected(self):
        with patch.object(entry.importlib,"import_module",return_value=SimpleNamespace(average_precision_score=1)):
            with self.assertRaises(PermissionError):entry.require_metric_dependencies()

    def test_legitimate_singleclass_and_zero_denominator_NA_preserved(self):
        value=dict(ranking=dict(AP=None,ranking_metric_NA_reason="SINGLE_CLASS_SCORED_LABELS"),
            pairs=[dict(reference_AP=None,NA_reason="SINGLE_CLASS_COMMON_VALID_LABELS")],
            streams=dict(point=dict(Recall=None,Recall_NA_reason="NO_VALID_POSITIVE_POINTS"),
                         affiliation=dict(reason="NO_GROUND_TRUTH_EVENTS",Aff_F1=None)),
            another_official_NA="OFFICIAL_UNDEFINED")
        entry.reject_dependency_na(value)
        self.assertIsNone(value["ranking"]["AP"])

    def test_backend_dependency_NA_cannot_be_completed(self):
        for reason in ("SKLEARN_UNAVAILABLE_NOT_COMPUTED", "RANKING_DEPENDENCY_UNAVAILABLE_NOT_COMPUTED:scipy",
                       "RANKING_DEPENDENCY_UNAVAILABLE:sklearn", "BLOCKED: official Affiliation dependency unavailable: scipy"):
            with self.assertRaises(PermissionError):entry.reject_dependency_na(dict(nested=[dict(reason=reason)]))


if __name__=="__main__":unittest.main()
