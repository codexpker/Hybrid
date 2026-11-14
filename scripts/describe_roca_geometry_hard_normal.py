"""Read existing alarm CSVs only; fixed StaticFIT CAL-tau hard-normal groups."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loc(c,p):
    if p.startswith('/home/a312/tsad_research/'):return Path(c['workspace'])/p.split('/home/a312/tsad_research/',1)[1]
    prefix='D:\\code\\bst\\seal_detection_project\\research_workspace\\'
    if p.startswith(prefix):return Path(c['workspace'])/p[len(prefix):].replace('\\','/')
    return Path(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Frozen read-only description')
    c=json.loads(Path(a.config).read_text());start=time.monotonic();base=Path(c['geometry']);meta=json.loads((base/'ALL_SCORE_GATES_BEFORE_LABELS.json').read_text())
    if sha(c['static_banks']['path'])!=c['static_banks']['sha256']:raise PermissionError('Original static definition changed')
    static={(r['domain'],r['entity']):r for r in json.loads(Path(c['static_banks']['path']).read_text()) if r['method']=='STATIC_FIT_DISTANCE'}
    entities=sorted({(r['domain'],r['entity']) for r in meta});rows=[]
    for domain,entity in entities:
        sd=static[domain,entity];sp=loc(c,sd['path'])
        if sha(sp)!=sd['sha256']:raise PermissionError('Original static array changed')
        with np.load(sp,allow_pickle=False) as z:index=z['index'].copy();hard_score=z['score'].copy()
        old=Path(c['accepted_evaluation'])/'entities'/domain/entity/'RoCA_FULL_T'/'calculation_inputs.npz'
        with np.load(old,allow_pickle=False) as z:labels=z['full_labels'][index]
        normal=labels==0;hard=normal&(hard_score>sd['calibration']['threshold'])
        selected=[m for m in meta if (m['domain'],m['entity'])==(domain,entity)];common=np.ones(len(index),bool);valid_by={}
        for m in selected:
            f=base/'raw'/domain/entity/m['method']/'scores_CAL_reference.npz'
            with np.load(f,allow_pickle=False) as z:
                if not np.array_equal(z['index'],index):raise ValueError('Original native axes differ')
                valid_by[m['method']]=z['valid'].copy();common&=z['valid']
        for m in selected:
            method=m['method'];valid=valid_by[method];available=m['operational_status']=='AVAILABLE'
            strict=np.zeros(len(index),bool);confirmed=np.zeros(len(index),bool)
            if available:
                with (base/'entities'/domain/entity/method/'point_coverage.csv').open() as f:
                    for r in csv.DictReader(f):
                        if r['valid']!='True':continue
                        pos=np.searchsorted(index,int(r['index']))
                        if pos>=len(index) or index[pos]!=int(r['index']):raise ValueError('Existing alarm axis differs')
                        strict[pos]=r['strict_point']=='True';confirmed[pos]=r['confirmed3_current']=='True'
            for population,mask in [('OWN_VALID',valid),('ALL24_COMMON_VALID',common)]:
                for group,g in [('ALL_OFFICIAL_NORMAL',normal),('HARD_NORMAL_STATIC_CAL',hard),('OTHER_NORMAL',normal&~hard)]:
                    use=g&mask;n=int(use.sum())
                    rows.append(dict(domain=domain,entity=entity,method=method,population=population,group=group,
                        original_group_points=int(g.sum()),valid_exposure=n,operational_status=m['operational_status'],
                        strict_FP=int((strict&use).sum()) if available else None,confirmed_FP=int((confirmed&use).sum()) if available else None,
                        strict_FPR=float((strict&use).sum()/n) if available and n else None,
                        confirmed_FPR=float((confirmed&use).sum()/n) if available and n else None))
    aggregate=[]
    for domain,method,pop,group in sorted({(r['domain'],r['method'],r['population'],r['group']) for r in rows}):
        rr=[r for r in rows if (r['domain'],r['method'],r['population'],r['group'])==(domain,method,pop,group)]
        available=all(r['operational_status']=='AVAILABLE' for r in rr);n=sum(r['valid_exposure'] for r in rr)
        s=sum(r['strict_FP'] for r in rr) if available else None;t=sum(r['confirmed_FP'] for r in rr) if available else None
        aggregate.append(dict(domain=domain,method=method,population=pop,group=group,original_points=sum(r['original_group_points'] for r in rr),
            entities_with_original_group=sum(r['original_group_points']>0 for r in rr),valid_exposure=n,strict_FP=s,confirmed_FP=t,
            strict_FPR=s/n if available and n else None,confirmed_FPR=t/n if available and n else None,
            reason='CAL_SUPPORT_INCOMPLETE' if not available else 'EMPTY_SUBGROUP' if not n else None))
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'RESULT.json').write_text(json.dumps(dict(status='COMPLETED_READONLY_EXISTING_ALARMS',seconds=time.monotonic()-start,
        aggregate=aggregate,entities=rows,source_config=c,new_score=False,new_tau=False,new_forward=False,GPUh=0),indent=2,allow_nan=False))
    print(json.dumps(dict(status='COMPLETED',seconds=time.monotonic()-start,rows=len(rows))))
if __name__=='__main__':main()
