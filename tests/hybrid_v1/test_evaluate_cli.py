"""Five tiny synthetic banks and fake CSV only; never production labels/data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import evaluate_cli as cli
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,IDENTITY_KEYS


def fixture(root,invalid_by_name=None):
    labels=root/"fake_labels.csv";labels.write_text("timestamp_(min),label\n"+"".join("%d,%d\n"%(i,int(i in (3,4,5))) for i in range(8)))
    cfg=dict(banks=[],history=2,original_length=8,original_start=0,expected_records=6,stream=["SYNTHETIC","E","R"],
        labels=dict(path=str(labels),sha256=cli.sha(labels),shape=[8],format="csv",drop_columns=["timestamp_(min)"],value_columns=["label"]),
        output=str(root/"output"),affiliation_identity=None,statistics=dict(cli.STATISTICS_RECIPE))
    for name in sorted(cli.IDS):
        directory=root/name;cal=root/(name+"_cal.json");cal.write_text(json.dumps(dict(threshold=1.)))
        identity={k:"a"*64 for k in IDENTITY_KEYS};identity["calibration_sha256"]=cli.sha(cal)
        with ChunkedBank(directory,identity=identity,ranges=[dict(domain="SYNTHETIC",entity="E",run="R",start=2,end=8)],threshold=1.,chunk_size=64) as bank:
            bank.write(dict(domain="SYNTHETIC",entity="E",run="R",index=i,available_index=i,
                score=2. if i in (3,4,5) else 0.,valid=i not in (invalid_by_name or {}).get(name,()),
                reason="SYNTHETIC_MISSING_GT" if i in (invalid_by_name or {}).get(name,()) else None) for i in range(2,8));bank.finalize()
        cfg["banks"].append(dict(id=name,kind="model" if name.startswith("model_") else "deterministic_baseline",
            seed=int(name[-4:]) if name.startswith("model_") else None,directory=str(directory),manifest_sha256=cli.sha(directory/"manifest.json"),
            identity=identity,calibration=dict(path=str(cal),sha256=cli.sha(cal))))
    return cfg


def mutate_manifest(entry,update):
    p=Path(entry["directory"])/"manifest.json";v=json.loads(p.read_text());update(v);p.write_text(json.dumps(v));entry["manifest_sha256"]=cli.sha(p)


class EvaluationGlue(unittest.TestCase):
    def test_preview_does_not_even_open_config(self):
        with patch("pathlib.Path.open",side_effect=AssertionError("No I/O")),contextlib.redirect_stdout(io.StringIO()) as log:
            cli.main(["--config","nonexistent.PENDING.json"])
        self.assertIn('"label_or_bank_opened": false',log.getvalue())

    def test_pending_release_before_bank_or_label(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"cfg";p.write_text('{}')
            with patch.object(cli,"load_labels",side_effect=AssertionError("No label I/O")):
                with self.assertRaises(PermissionError):cli.admit(p,None,None)

    def test_all_five_complete_gate_precedes_label_I_O(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(Path(d));mutate_manifest(cfg['banks'][-1],lambda m:m.update(complete=False))
            with patch.object(cli,"load_labels",side_effect=AssertionError("Label loader called")) as loader:
                with self.assertRaises(ValueError):cli.run_evaluation(cfg)
                loader.assert_not_called()
            self.assertFalse(Path(cfg['output']).exists())

    def test_identity_count_clock_threshold_reject_before_labels(self):
        for kind in ('identity','count','clock','threshold'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                cfg=fixture(Path(d));entry=cfg['banks'][0]
                def change(m):
                    if kind=='identity':m['contract']['identity']['input_sha256']='b'*64
                    elif kind=='count':m['committed_records']-=1
                    elif kind=='clock':m['contract']['ranges'][0]['start']+=1
                    else:m['contract']['threshold']=2.
                mutate_manifest(entry,change)
                with patch.object(cli,'load_labels',side_effect=AssertionError('labels')) as loader:
                    with self.assertRaises(ValueError):cli.run_evaluation(cfg)
                    loader.assert_not_called()

    def test_chunk_corruption_and_missing_seed_reject(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(Path(d));entry=cfg['banks'][0];m=json.loads((Path(entry['directory'])/'manifest.json').read_text())
            (Path(entry['directory'])/m['chunks'][0]['file']).write_bytes(b'corrupt synthetic chunk')
            with patch.object(cli,'load_labels',side_effect=AssertionError('labels')):
                with self.assertRaises(ValueError):cli.run_evaluation(cfg)
            cfg['banks'].pop()
            with self.assertRaises(ValueError):cli.bank_gates(cfg)

    def test_five_synthetic_banks_reuse_accepted_evaluator_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(Path(d));summary=cli.run_evaluation(cfg)
            self.assertEqual(summary['model_seeds'],[2025,2026,2027])
            self.assertEqual(summary['model_mean_SD']['/ranking/AP']['mean'],1.)
            self.assertEqual(summary['model_mean_SD']['/ranking/AP']['SD'],0.)
            self.assertEqual(set(summary['deterministic_controls']),{'persistence','history_mean512'})
            for name in cli.IDS:
                output=Path(cfg['output'])/name
                self.assertTrue((output/'result.json').exists())
                with np.load(output/'calculation_inputs.npz',allow_pickle=False) as a:
                    self.assertEqual(a['index'].tolist(),list(range(2,8)));self.assertEqual(a['full_labels'].shape,(8,))
            with self.assertRaises(FileExistsError):cli.run_evaluation(cfg)

    def test_label_file_wrong_identity_or_nonbinary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(Path(d));spec=cfg['labels'];spec['sha256']='b'*64
            with self.assertRaises(ValueError):cli.load_labels(spec)
            p=Path(spec['path']);p.write_text('timestamp_(min),label\n0,2\n');spec.update(sha256=cli.sha(p),shape=[1])
            with self.assertRaises(ValueError):cli.load_labels(spec)

    def test_explicit_worst_directions_and_no_count_or_threshold_ranking(self):
        results={}
        for i,seed in enumerate((2025,2026,2027)):
            results['model_seed%d'%seed]=dict(ranking=dict(AP=[.4,.7,.6][i]),streams=dict(strict_point=dict(
                point=dict(F1=[.3,.5,.4][i],threshold=[1.,2.,3.][i]),
                main_events=dict(metrics=dict(one_to_one_eventRecall=[.3,.2,.6][i],one_to_one_eventF1=[.1,.2,.3][i],
                    auxiliary_any_hit_recall=[.9,.7,.8][i],delay_eligible_events=[20,10,30][i])),
                normal=dict(normal_FPR=[.02,.06,.04][i],normal_alarm_fragments_per_1000_points=[1.,2.,3.][i],
                    normal_block_any_alarm_fraction=[.5,.4,.3][i]))))
        for name in ('persistence','history_mean512'):results[name]=dict(ranking={},streams={})
        s=cli.aggregate(results)['model_mean_SD'];prefix='/streams/strict_point/'
        expected={'main_events/metrics/one_to_one_eventRecall':2026,'main_events/metrics/one_to_one_eventF1':2025,
            'main_events/metrics/auxiliary_any_hit_recall':2026,'normal/normal_FPR':2026,
            'normal/normal_alarm_fragments_per_1000_points':2027,'normal/normal_block_any_alarm_fraction':2025}
        for key,seed in expected.items():self.assertEqual(s[prefix+key]['worst_seed'],seed,key)
        for key in ('main_events/metrics/delay_eligible_events','point/threshold'):
            self.assertIsNone(s[prefix+key]['worst_seed']);self.assertEqual(s[prefix+key]['worst_direction'],'NA_DENOMINATOR_OR_NONPERFORMANCE_FIELD')

    def test_fixed_statistics_bridge_reuses_scalar_inputs_and_keeps_full_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=fixture(Path(d),invalid_by_name={'model_seed2027':(2,)})
            summary=cli.run_evaluation(cfg);output=Path(cfg['output']);stats=output/'paired_AP_time_block_sensitivity'
            s=json.loads((stats/'summary.json').read_text())
            self.assertEqual(s['models'],['model_seed2025','model_seed2026','model_seed2027'])
            self.assertEqual(s['baselines'],['persistence','history_mean512']);self.assertFalse(s['score_ensemble'])
            self.assertEqual(s['parameters'],dict(block_points=4096,n_replicates=2000,rng_seed=99173))
            with np.load(stats/'bootstrap_arrays.npz',allow_pickle=False) as a:
                self.assertEqual(a['block_draws'].shape,(2000,1));self.assertEqual(a['replicate_AP'].shape,(2000,5))
                self.assertEqual(a['common_index'].tolist(),[3,4,5,6,7])
            # Per-method original coverage stays distinct from the common 5 points.
            primary=json.loads((output/'model_seed2025/result.json').read_text())
            self.assertEqual(primary['coverage']['valid_scored_points'],6)
            with np.load(stats/'calculation_inputs.npz',allow_pickle=False) as a:
                self.assertEqual(a['m0_index'].tolist(),list(range(2,8)));self.assertEqual(a['m2_valid'].sum(),5)
            self.assertFalse(summary['paired_AP_statistics']['full_coverage_metrics_replaced'])
            bad=dict(cfg,statistics=dict(cli.STATISTICS_RECIPE,enabled=False))
            with patch.object(cli,'load_labels',side_effect=AssertionError('No labels after unfreezing recipe')):
                with self.assertRaises(ValueError):cli.run_evaluation(bad)


if __name__=='__main__':unittest.main()
