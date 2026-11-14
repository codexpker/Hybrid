from evaluate_fixed import *
from fixed_metrics import evaluate,rolling_max64
base=O/'evaluation/PSM';assert (base/'RAW_SEAL.json').exists();banks={}
for name,arm,key in [('LEFT_source','source','native_score'),('Hybrid_mask_zero_update','source','score'),('LEFT_continued','native','score'),('Hybrid_blindspot','blind','score'),('Hybrid_nativeoff','blind','native_score')]:banks[name]={r:load(O/('PSM_'+arm)/'score'/(r+'.npz'))[key]for r in ['CAL','QUERY']}
banks['LEFT_past64max']={r:rolling_max64(v)for r,v in banks['LEFT_source'].items()}
for name in ['HistoryMean512','MeanFIT','InstantRidge']:banks[name]={r:load(O/'cpu_controls/PSM'/(name+'_'+r+'.npz'))['score']for r in ['CAL','QUERY']}
cfg=json.loads((R/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json').read_text());f=Path(cfg['truth']['path'].replace('__RESEARCH_ROOT__',str(R)));truth=load(f)['full_labels'];idx=np.arange(512,87744);rows=[];events=[]
for name,e in banks.items():
 rr,ev=evaluate(idx,e['QUERY'][:len(idx)],e['CAL'],truth,steady_after=192)
 rows.extend(dict(scope='external_common_axis512_87743',method=name,**x)for x in rr);events.extend(dict(scope='external_common_axis512_87743',method=name,**x)for x in ev)
table(base/'common_axis_summary.csv',rows);table(base/'common_axis_events.csv',events)
