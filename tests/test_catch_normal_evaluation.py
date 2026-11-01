"""Synthetic CPU fixtures only. No models, real banks or truth records."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from hybrid_tsad.engine import catch_normal_evaluation as e


class CatchEvaluationTests(unittest.TestCase):
    def test_all_synthetic_row_coverage_and_catch_identity(self):
        start,end=24*192,36*192
        labels={v:np.zeros(24576,dtype=np.int8) for v in ('A','N','S0','SB')}
        labels['A'][start:end]=1;labels['SB'][start:start+192]=1;labels['SB'][end-192:end]=1
        raw={k:np.zeros((128,192,2),dtype=np.float32) for k in ('input','reconstruction','time_error','frequency_error')}
        raw['canonical_score']=np.zeros((128,192),dtype=np.float32)
        for k in ('last','mean128','y_last','y_mean128'):raw[k]=np.zeros(128,dtype=np.float32)
        totals=[0,0,0,0];all_layers=[];all_events=[];all_matched=[]
        for scene in ('LIN','WEAK'):
            for rid in range(8):
                banks={(scene,'DEV',rid,v):raw for v in ('A','N','S0','SB')}
                truth=dict(scenario=scene,role='DEV',run_id=rid,event_start=start,event_end=end,
                           normal_lag=np.zeros(24576,dtype=np.int64),labels=labels)
                result=e.evaluate_run(banks,truth,{r:{'threshold':1.} for r in ('last','mean128')})
                for j,k in enumerate(('layers','events','matched')):totals[j]+=len(result[k])
                totals[3]+=1;all_layers+=result['layers'];all_events+=result['events'];all_matched+=result['matched']
        self.assertEqual(totals,[896,96,64,16])
        self.assertTrue(all(r['model']=='CATCH' for r in all_layers))
        self.assertTrue(e.macro_summary(all_layers,all_events,all_matched)['layers'])

    def test_exact16_pairing_and_shared_draws(self):
        ap={}
        for s in ('LIN','WEAK'):
            for m,r in [('CATCH','last'),('CATCH','mean128'),('LEFT','last'),('LEFT','mean128'),
                        ('MULTI191','last'),('PEAK191','last'),('SYNC191','last'),
                        ('MULTI64','mean128'),('SYNC64','mean128')]:
                for i in range(8): ap[(s,m,r,i)]=None if i==7 else (.4 if m=='CATCH' else .2)
        result,draws=e.paired_comparisons(ap)
        self.assertEqual(len(result),16);self.assertEqual(draws.shape,(10000,8))
        self.assertEqual(result['WEAK_CATCH.last_minus_LEFT.last']['valid_pairs'],7)
        self.assertAlmostEqual(result['LIN_CATCH.mean128_minus_CATCH.last']['mean_difference'],0.)

    def test_raw_native_and_readout_structure(self):
        rng=np.random.default_rng(2025)
        x=rng.normal(size=(128,192,2)).astype('float32')
        reconstruction=x*.5;time_error=(x-reconstruction)**2
        frequency_error=np.abs(x)
        canonical=time_error.mean(-1)+np.float32(.5)*frequency_error.mean(-1)
        y=time_error[:,:,1]+np.float32(.5)*frequency_error[:,:,1]
        arrays=dict(input=x,reconstruction=reconstruction,time_error=time_error,
            frequency_error=frequency_error,canonical_score=canonical,native_score=canonical.reshape(-1),
            last=canonical[:,-1],mean128=canonical[:,64:].mean(1),y_last=y[:,-1],y_mean128=y[:,64:].mean(1),
            **e.score.coordinates())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bank.npz';np.savez_compressed(path,**arrays)
            row={'path':str(path),'sha256':e.score.base.sha(path)}
            self.assertEqual(e.load_score_bank(row)['canonical_score'].shape,(128,192))
            arrays['last']=arrays['last']+1
            np.savez_compressed(path,**arrays);row['sha256']=e.score.base.sha(path)
            with self.assertRaises(ValueError):e.load_score_bank(row)

    def test_physical_alias_rejected_before_npz_or_truth_io(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'alias.npz';path.write_bytes(b'NOT_AN_NPZ')
            rows=[]
            for i in range(2):
                ref=dict(scenario='LIN',role='CAL',run_id=i,variant='N',
                    path=f'/fixture/LIN/CAL/run{i:02d}_N.npy',sha256='0'*64)
                rows.append(dict(reference=ref,path=str(path),sha256='1'*64))
            seal=dict(status='APPROVED',approved_by='root',operation=e.OPERATION,allow_truth=True,
                source_manifest_sha256=e.score.MANIFEST_SHA,
                score_module_sha256=e.score.base.sha(e.score.__file__),banks=rows)
            with self.assertRaisesRegex(PermissionError,'physical NPZ'):
                e.validate_score_inputs({},seal)


if __name__=='__main__':unittest.main()
