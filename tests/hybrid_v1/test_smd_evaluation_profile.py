"""Prepared fixed-profile tests with synthetic arrays/metadata only."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from hybrid_tsad.hybrid_v1 import smd_evaluate_cli as e,normal_readout as q,smd_baselines as b
from hybrid_tsad.hybrid_v1.evaluation import evaluate_arrays

def config(root):
    banks=[]
    for id in [e.ZS,e.ZT]+list(e.F.values())+list(e.N.values())+[v for d in e.BASE.values() for v in d.values()]:
        base=next((m for m,d in e.BASE.items() if id in d.values()),None);choice=next((c for c in e.CHOICES if id in (e.F[c],e.N[c])),None)
        origin='source' if id==e.ZS or base and id==e.BASE[base]['source'] else 'target'
        kind='deterministic_baseline' if base else 'fewshot_event_model' if id in e.F.values() else 'normal_negative_model' if id in e.N.values() else 'source_model'
        checkpoint=b.algorithm(base) if base else e.F_CHECKPOINTS[choice] if id in e.F.values() else 'b'*64 if id in e.N.values() else q.normal.SOURCE
        banks.append(dict(id=id,directory=str(root/id),ranges=e.modelview.RANGES,calibration_origin=origin,kind=kind,choice=choice,method=base,
            training_seed=None if base else 2025,inference_seed=None if base else 2025,view='model_target' if id==e.ZT else 'baseline_target' if base and origin=='target' else 'direct',
            identity=dict(input_sha256=q.helper.RAW_SHA,checkpoint_sha256=checkpoint),calibration=dict(path='/synthetic/cal.json',sha256='a'*64)))
    return dict(operation=e.OPERATION,profile=e.PROFILE,statistics=copy.deepcopy(e.named.STATISTICS_NA),original_start=17129,original_length=6565,history=512,
        stream=['SMD','machine-2-1','machine-2-1_test'],banks=banks,comparisons=copy.deepcopy(e.PAIRS),source_sha256={'synthetic':'fixture'},output=str(root/'result'),
        labels=dict(path=str(root/'forbidden_labels.txt'),sha256=e.LABEL_SHA,total_rows=23694,numeric_bounds=[17129,23694],shape=[6565],format='txt_suffix'))

class EvaluationProfile(unittest.TestCase):
    def test_fourteen_methods_nine_pairs_and_no_ensemble(self):
        c=config(Path('/synthetic'));self.assertEqual(len(e.contract(c)),14);self.assertEqual(len(c['comparisons']),9)
        # Exercise the same acceptance selector used by real bank_gates, without
        # mocking its schema: F integer group, N choice list, baseline method list.
        f=copy.deepcopy(next(x for x in c['banks'] if x['id']==e.F['K1_choice0']));f.update(manifest_sha256='a'*64,acceptance={'sha256':e.F_RAW_ACCEPT_SHA})
        head=dict(approved_by='root',status='ACCEPTED_SYNTHETIC_SCHEMA')
        self.assertTrue(e.acceptance_matches(f,dict(head,cases=4,records=24212)))
        self.assertFalse(e.acceptance_matches(f,dict(head,cases=4,records=6053)))
        n=copy.deepcopy(next(x for x in c['banks'] if x['id']==e.N['K1_choice0']));n.update(manifest_sha256='b'*64,acceptance={'sha256':'c'*64})
        self.assertTrue(e.acceptance_matches(n,dict(head,cases=[dict(choice='K1_choice0',bank_manifest_sha256='b'*64)])))
        self.assertFalse(e.acceptance_matches(n,dict(head,cases=[dict(choice='K1_choice1',bank_manifest_sha256='b'*64)])))
        bentry=copy.deepcopy(next(x for x in c['banks'] if x['id']==e.BASE['persistence']['source']));bentry.update(manifest_sha256='d'*64,acceptance={'sha256':'c'*64})
        self.assertTrue(e.acceptance_matches(bentry,dict(head,methods=[dict(method='persistence',bank_manifest_sha256='d'*64)])))
        self.assertTrue(e.acceptance_matches(bentry,dict(head,methods={'persistence':dict(manifest_sha256='d'*64)})))
        z=copy.deepcopy(c['banks'][0]);z.update(manifest_sha256='e'*64,acceptance={'sha256':'c'*64})
        self.assertTrue(e.acceptance_matches(z,dict(head,manifest_sha256='e'*64)))
        zt=copy.deepcopy(c['banks'][1]);zt.update(manifest_sha256='f'*64,acceptance={'sha256':e.ZT_ACCEPT_SHA})
        self.assertTrue(e.acceptance_matches(zt,dict(head,records=6053,model_or_fit=False,raw_producer_unchanged=True)))
        c['ensemble']=True
        with self.assertRaises(ValueError):e.contract(c)
    def test_missing_control_extra_pair_or_fake_seed_rejected(self):
        for mutation in ('bank','pair','baseline_seed'):
            c=config(Path('/synthetic'))
            if mutation=='bank':c['banks'].pop()
            elif mutation=='pair':c['comparisons'].append(dict(c['comparisons'][0]))
            else:next(x for x in c['banks'] if x['kind']=='deterministic_baseline')['training_seed']=2025
            with self.assertRaises(ValueError):e.contract(c)
    def test_shifted_axis_source_contact_or_f_choice_rejected(self):
        c=config(Path('/synthetic'));c['original_start']=0
        with self.assertRaises(ValueError):e.contract(c)
        c=config(Path('/synthetic'));c['banks'][1]['calibration_origin']='source'
        with self.assertRaises(ValueError):e.contract(c)
        c=config(Path('/synthetic'));next(x for x in c['banks'] if x['id']==e.F['K1_choice0'])['choice']='K1_choice1'
        with self.assertRaises(ValueError):e.contract(c)
    def test_pending_seal_rejects_before_label_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=config(root);p=root/'pending.json';p.write_text(json.dumps(dict(status='PENDING')));c['raw_seal']=dict(path=str(p),sha256=q.sha(p));loader=Mock(side_effect=AssertionError('No labels'))
            with self.assertRaises(PermissionError):e.run(c,label_loader=loader)
            loader.assert_not_called()
    def test_alias_label_file_as_CAL_rejected_before_open(self):
        c=config(Path('/synthetic'));c['banks'][0]['calibration']['path']=c['labels']['path']
        with self.assertRaises(PermissionError):e.protect_label_paths(c)
    def test_suffix_parser_never_numerically_parses_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'synthetic.txt';p.write_text('NOT_NUMERIC_PREFIX\n'*17129+'0\n'*6500+'1\n'*65)
            labels=e.load_suffix_labels(dict(path=str(p),sha256=q.sha(p),numeric_bounds=[17129,23694]))
            self.assertEqual(labels.shape,(6565,));self.assertEqual(int(labels.sum()),65)
    def test_no_time_axis_compression_when_raw_has_gap(self):
        c=config(Path('/synthetic'));row=dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test',index=17642,available_index=17642,valid=True,score=1.,reason=None)
        with patch.object(e,'read_records',return_value=iter([row])):
            with self.assertRaises(ValueError):e.scalar_inputs(c)
    def test_single_class_and_no_prediction_keep_original_NA_rules(self):
        index=np.arange(17641,23694,dtype=np.int64)
        r=e.evaluate_arrays(index,np.zeros(6053),np.zeros(6565,np.int64),1.,bank_complete=True,original_length=6565,original_start=17129,warmup_points=512)
        self.assertIsNone(r['ranking']['AP']);self.assertEqual(r['ranking']['ranking_metric_NA_reason'],'SINGLE_CLASS_SCORED_LABELS')
        self.assertIsNone(r['streams']['strict_point']['point']['Precision']);self.assertIsNone(r['streams']['strict_point']['point']['F1'])
    def test_full_orchestration_same_input_original_core_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=config(root);c['raw_seal']={'synthetic':True}
            labels=np.zeros(6565,np.int64);labels[500:520]=1;labels[600:610]=1
            index=np.arange(17641,23694,dtype=np.int64);score=labels[512:].astype(np.float64);valid=np.ones(6053,bool);valid[188]=False;score[188]=np.nan
            reasons=[None]*6053;reasons[188]='SYNTHETIC_MISSING_NORMAL_POINT'
            a=dict(index=index,score=score,valid=valid,available_index=index.copy(),invalid_reasons=reasons);arrays={id:copy.deepcopy(a) for id in e.IDS}
            gates=[dict(id=x['id'],threshold=.5) for x in c['banks']];order=[]
            def gate(cfg):order.append('gates');return gates
            def scalar(cfg):order.append('scalars');return arrays
            def load(spec):self.assertEqual(order,['gates','scalars']);order.append('labels');return labels
            with patch.object(e,'bank_gates',side_effect=gate),patch.object(e,'scalar_inputs',side_effect=scalar):summary=e.run(c,label_loader=load)
            expected=evaluate_arrays(index,score,labels,.5,bank_complete=True,original_length=6565,original_start=17129,warmup_points=512,valid=valid,available_index=index,invalid_reasons=reasons)
            self.assertEqual(expected['ranking']['AP'],1.);self.assertEqual(expected['streams']['strict_point']['point']['Precision'],1.)
            self.assertEqual(len(expected['event_coverage']),2);self.assertFalse(expected['event_coverage'][0]['complete']);self.assertTrue(expected['event_coverage'][1]['complete'])
            for id in e.IDS:self.assertEqual(json.loads((Path(c['output'])/id/'result.json').read_text()),expected)
            self.assertEqual(len(summary['comparisons']),9);self.assertIsNone(summary['model_seed_mean'])
            for pair in summary['comparisons']:self.assertEqual(pair['common_point_effect']['AP_difference_candidate_minus_reference'],0.);self.assertIsNone(pair['statistics']['CI'])

if __name__=='__main__':unittest.main()
