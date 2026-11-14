"""Native-axis, event-phase, cross-entity and recovery tests; no real truth."""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1.evaluation import evaluate_arrays
from hybrid_tsad.hybrid_v1.public_native_evaluation import event_diagnostics,duration_bin,aggregate_dataset
from hybrid_tsad.hybrid_v1.data import Run
from hybrid_tsad.hybrid_v1.baselines import iter_baselines
from hybrid_tsad.hybrid_v1.cli import score_records
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,read_entries
from hybrid_tsad.hybrid_v1.public_native_zs import ranges


class PublicEvaluationTests(unittest.TestCase):
    def test_aggregate_all_invalid_retains_unknown_geometry_and_no_ranking(self):
        ix=np.arange(512,518,dtype=np.int64);s=np.full(6,np.nan);y=np.zeros(518,np.int8);y[513:515]=1
        valid=np.zeros(6,bool)
        r=evaluate_arrays(ix,s,y,1.,bank_complete=True,original_length=518,warmup_points=512,
            valid=valid,invalid_reasons=['FIXTURE_INVALID']*6)
        mk=('persistence','SMAP','a','a_run')
        d=aggregate_dataset({mk:r},{mk:dict(index=ix,score=s,valid=valid)},{mk[1:]:y},'SMAP','persistence')
        rank=d['micro_point_ranking']
        self.assertEqual((rank['original_length_total'],rank['declared_raw_points'],rank['evaluated_points']),(518,6,0))
        self.assertEqual((rank['unscored_points'],rank['invalid_raw_points'],rank['non_raw_original_points']),(518,6,512))
        self.assertIsNone(rank['AP']);self.assertEqual(rank['ranking_metric_NA_reason'],'NO_VALID_SCORES')
        self.assertEqual(d['streams']['strict_point']['events']['Ntrue'],0)

    @unittest.skipUnless(importlib.util.find_spec('sklearn') is not None,'Real sklearn required; run in authorenv CPU')
    def test_aggregate_mixed_invalid_point_pool_preserves_native_missing_counts(self):
        ix=np.arange(512,518,dtype=np.int64);s=np.array([0.,2.,np.nan,0.,0.,0.]);y=np.zeros(518,np.int8);y[513:515]=1
        valid=np.isfinite(s)
        r=evaluate_arrays(ix,s,y,1.,bank_complete=True,original_length=518,warmup_points=512,
            valid=valid,invalid_reasons=[None,None,'FIXTURE_INVALID',None,None,None])
        mk=('persistence','SMAP','a','a_run')
        d=aggregate_dataset({mk:r},{mk:dict(index=ix,score=s,valid=valid)},{mk[1:]:y},'SMAP','persistence')
        rank=d['micro_point_ranking']
        self.assertEqual((rank['original_length_total'],rank['declared_raw_points'],rank['evaluated_points']),(518,6,5))
        self.assertEqual((rank['unscored_points'],rank['invalid_raw_points'],rank['non_raw_original_points']),(513,1,512))
        self.assertEqual((rank['positive_points'],rank['negative_points']),(1,4))
        self.assertEqual(d['streams']['strict_point']['events']['Ntrue'],0)  # incomplete event excluded, not normal
        self.assertEqual(d['streams']['strict_point']['normal']['normal_exposure_points'],4)

    @unittest.skipUnless(importlib.util.find_spec('sklearn') is not None,'Real sklearn required; run in authorenv CPU')
    def test_dataset_aggregate_pools_only_point_ranking_and_sums_native_events(self):
        results={};arrays={};labels={};scores=[[0.,2.,2.,0.,0.,0.],[2.,0.,0.,2.,0.,0.]]
        for entity,score in zip(('a','b'),scores):
            ix=np.arange(512,518,dtype=np.int64);s=np.asarray(score);y=np.zeros(518,np.int8);y[513:515]=1
            r=evaluate_arrays(ix,s,y,1.,bank_complete=True,original_length=518,warmup_points=512)
            # Synthetic supplied official-result values exercise macro/NA bookkeeping,
            # not the external affiliation implementation itself.
            for stream in r['streams'].values():
                stream['affiliation']['global_result']['Aff_F1']=.25 if entity=='a' else None
            mk=('persistence','SMAP',entity,entity+'_run')
            results[mk]=r;arrays[mk]=dict(index=ix,score=s,valid=np.ones(6,bool));labels[mk[1:]]=y
        summary=aggregate_dataset(results,arrays,labels,'SMAP','persistence')
        point=summary['streams']['strict_point']['point']
        self.assertEqual({k:point[k] for k in ('TP','FP','TN','FN')},dict(TP=2,FP=2,TN=6,FN=2))
        events=summary['streams']['strict_point']['events']
        self.assertEqual({k:events[k] for k in ('M','Npred','Ntrue')},dict(M=1,Npred=3,Ntrue=2))
        self.assertEqual(summary['streams']['strict_point']['legacy_complete_only_events']['Ntrue'],2)
        self.assertEqual(summary['streams']['confirmed3_current']['point']['TP'],0)
        self.assertFalse(any('event' in k for k in summary['micro_point_ranking']))
        from sklearn.metrics import average_precision_score
        self.assertAlmostEqual(summary['micro_point_ranking']['AP'],average_precision_score([0,1,1,0,0,0]*2,np.array(scores).ravel()))
        aff=summary['streams']['strict_point']['official_Aff_F1_entity_macro']
        self.assertEqual(aff,dict(mean=.25,entities=1,total_entities=2))
        self.assertIsNone(summary['streams']['strict_point']['global_Aff'])

    def test_original_unknown_prefix_and_512_phase_no_reconfirmation(self):
        y=np.zeros(1400,np.int8);y[10:20]=1;y[600:1200]=1
        ix=np.arange(512,1400,dtype=np.int64);score=np.ones(len(ix))*2
        r=evaluate_arrays(ix,score,y,1.,bank_complete=True,original_length=len(y),warmup_points=512)
        self.assertEqual(r['coverage']['original_events'],2)
        self.assertEqual(r['coverage']['main_eligible_events'],1)
        self.assertFalse(r['point_coverage'][15]['valid'])
        self.assertIsNone(r['point_coverage'][15]['strict_point'])
        d=event_diagnostics(r)
        steady=[x for x in d if x['start']==600 and x['stream']=='confirmed3_current' and x['phase']=='steady_after512'][0]
        self.assertEqual(steady['total_points'],88)
        self.assertEqual(steady['detected_points'],88)
        self.assertEqual(steady['duration_bin'],'>512')
        self.assertEqual([duration_bin(n) for n in [32,33,512,513]],['<=32','33..512','33..512','>512'])

    def test_confirmed_streak_and_events_never_cross_entity(self):
        results={};arrays={};labels={}
        for entity in ('a','b'):
            ix=np.array([512,513],np.int64);s=np.array([2.,2.]);y=np.zeros(514,np.int8);y[512:]=1
            r=evaluate_arrays(ix,s,y,1.,bank_complete=True,original_length=514,warmup_points=512)
            self.assertEqual(r['streams']['confirmed3_current']['point']['TP'],0)
            self.assertFalse(r['event_coverage'][0]['main_eligible'])  # run-end censoring retained
            self.assertEqual(r['streams']['strict_point']['main_events']['metrics']['Ntrue'],0)
            mk=('persistence','SMAP',entity,entity+'_run')
            results[mk]=r;arrays[mk]=dict(index=ix,score=s,valid=np.ones(2,bool));labels[mk[1:]]=y
        # Counts are checked without requiring optional ranking dependencies.
        total=sum(r['streams']['confirmed3_current']['point']['TP'] for r in results.values())
        self.assertEqual(total,0)  # concatenating scores would incorrectly yield two confirmed points

    def test_role_injected_resume_is_bit_exact_and_never_recomputes_prefix(self):
        r=Run('SMAP','fixture','fixture_native','QUERY',np.linspace(-1,2,585,dtype=np.float32)[:,None],np.ones((585,1),bool),start=100)
        cal=dict(scales={'r_pred':.2},threshold=1.,active_components=['r_pred'],weights={'r_pred':1.})
        identity={k:'a'*64 for k in ('input_sha256','config_sha256','source_sha256','checkpoint_sha256','calibration_sha256')}
        records=list(score_records(list(iter_baselines([r],method='persistence')),cal))
        self.assertEqual(len(records),73)
        with tempfile.TemporaryDirectory() as temp:
            full=Path(temp)/'full';resumed=Path(temp)/'resumed'
            with ChunkedBank(full,identity=identity,ranges=ranges(r),threshold=1.,chunk_size=64) as b:
                for v in records:b.append(v)
                b.finalize()
            with ChunkedBank(resumed,identity=identity,ranges=ranges(r),threshold=1.,chunk_size=64) as b:
                for v in records[:64]:b.append(v)
            with ChunkedBank(resumed,identity=identity,ranges=ranges(r),threshold=1.,chunk_size=64) as b:
                plan=[dict(e,role=r.role) for e in b.remaining_plan()]
                self.assertEqual([e['index'] for e in plan],list(range(676,685)))
                computed=list(score_records(list(iter_baselines([r],method='persistence',index_plan=plan)),cal))
                self.assertEqual(len(computed),9)
                for v in computed:b.append(v)
                b.finalize()
            a=list(read_entries(full));b=list(read_entries(resumed))
            self.assertEqual(a,b)
            self.assertEqual([x['raw']['index'] for x in b],list(range(612,685)))


if __name__=='__main__':unittest.main(verbosity=2)
