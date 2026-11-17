"""Fixed cache-only per-point V/E/E+0.9V diagnosis; no threshold fitting."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(w):
 w=Path(w);out=w/'results/timeradar_error_decomposition_20261117_v1';out.mkdir(exist_ok=False)
 binding=json.loads((w/'preparations/timeradar_cross_variable_20261116_v1/data_binding.json').read_text())
 labels=json.loads((w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json').read_text())
 lm={r['id']:r for r in labels['records']};rows=[];inputs={}
 for r in binding['records']:
  rid=r['id'];rp=next(a for a in r['roles'] if a['role']=='QUERY');xp=w/rp['path'];assert sha(xp)==rp['sha256']
  lp=w/lm[rid]['path'];assert sha(lp)==lm[rid]['sha256'];inputs[str(xp.relative_to(w))]=sha(xp);inputs[str(lp.relative_to(w))]=sha(lp)
  with np.load(xp) as z:x,xi=z['x'].copy(),z['index'].copy()
  with np.load(lp) as z:y=z['label'].copy()
  common=None;banks={}
  for arm in ['zero','decoder_head','self','cross','selective','capacity_self']:
   root='timeradar_selective_context_20261116_v2' if arm in ['selective','capacity_self'] else 'timeradar_cross_variable_20261116_v1'
   bp=w/'results'/root/rid/arm/'score/QUERY.npz';receipt=json.loads((bp.parent/'SCORE_RECEIPT.json').read_text());entry=next(a for a in receipt['banks'] if Path(a['path']).name=='QUERY.npz');assert sha(bp)==entry['sha256'];inputs[str(bp.relative_to(w))]=sha(bp)
   with np.load(bp) as z:
    ix=z['index'].copy();mean=z['reconstruction_mean'].copy();v=z['channel_variance'].copy();official=z['score'].copy()
   assert mean.shape==v.shape==(len(ix),x.shape[1]);assert np.isfinite(mean).all() and np.isfinite(v).all()
   banks[arm]=(ix,mean,v,official);common=ix if common is None else np.intersect1d(common,ix)
  assert np.all(np.diff(common)==1)
  for arm,(ix,mean,v,official) in banks.items():
   pos=np.searchsorted(ix,common);xp=np.searchsorted(xi,common);assert np.array_equal(xi[xp],common)
   V=v[pos].mean(-1);E=((mean[pos]-x[xp])**2).mean(-1)
   assert V.ndim==E.ndim==1 and len(V)==len(common);assert np.allclose(V,official[pos],rtol=2e-6,atol=1e-8)
   for name,s in [('V',V),('E',E),('E+0.9V',E+.9*V)]:
    row=dict(record=rid,arm=arm,readout=name,points=len(common),start=int(common[0]),end_exclusive=int(common[-1])+1,AP=average_precision_score(y[common],s),ROC_AUC=roc_auc_score(y[common],s))
    for group,mask in [('normal',y[common]==0),('anomaly',y[common]==1)]:
     a=s[mask];row.update({group+'_n':len(a),group+'_mean':float(a.mean()),group+'_std':float(a.std()),group+'_q05':float(np.quantile(a,.05)),group+'_median':float(np.median(a)),group+'_q95':float(np.quantile(a,.95)),group+'_q99':float(np.quantile(a,.99))})
    rows.append(row)
 with (out/'comparison.csv').open('x',newline='',encoding='utf8') as f:
  d=csv.DictWriter(f,fieldnames=list(rows[0]));d.writeheader();d.writerows(rows)
 (out/'PROVENANCE.json').write_text(json.dumps(dict(rows=len(rows),inputs=inputs,formula='V=channelmean cached sample variance (native ddof1); E=channelmean((cached reconstruction mean - same FIT-standardized input)^2); combined=E+0.9V, float32 original units',GPU_hours=0,new_forward=False,new_training=False,threshold_fitting=False,labels='posthoc QUERY diagnostic only',source_sha256=sha(__file__)),indent=2),encoding='utf8')
 print(str(out))
if __name__=='__main__':
 import sys
 run(sys.argv[1])
