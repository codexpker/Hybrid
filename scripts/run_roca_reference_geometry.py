"""Single fixed FIT64/k5 stage-localization probe, all six stages reported."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.roca_fixed_reference_geometry import fixed_five_nn,normalized
from hybrid_tsad.hybrid_v1.scoring import weighted_quantile
from hybrid_tsad.hybrid_v1 import evaluation
from hybrid_tsad.hybrid_v1.public_native_evaluation import aggregate_dataset,event_diagnostics
from hybrid_tsad.hybrid_v1.controlled_wave_evaluation import require_metric_dependencies,reject_dependency_na
LAYERS=('CNN','h0','h1','h2','q','qr')
NORMALIZATIONS=('unit','raw')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bound(a):
    if sha(a['path'])!=a['sha256']:raise PermissionError('Changed frozen feature/plan asset')
    return json.loads(Path(a['path']).read_text())
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def table(p,r):
    keys=list(dict.fromkeys(k for x in r for k in x))
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(r)
def loc(c,p):
    if p.startswith('/home/a312/tsad_research/'):return Path(c['workspace'])/p.split('/home/a312/tsad_research/',1)[1]
    return Path(p)
def load(c,d):
    p=loc(c,d['path'])
    if sha(p)!=d['sha256']:raise PermissionError('Frozen stage array changed')
    with np.load(p,allow_pickle=False) as z:i=z['index'].copy();x=z['features'].copy();v=z['valid'].copy()
    if not np.array_equal(v,normalized(x)[1]):raise ValueError('Stage valid/zero policy differs')
    return i,x

def run(c):
    begin=time.monotonic();require_metric_dependencies()
    for path,h in c['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('Frozen geometry/metric source changed')
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    products=[]
    for a in c['products']:products+=bound(a)
    by={(d['domain'],d['method'],d['entity'],d['role'],d['layer']):d for d in products}
    entities=sorted({k[:3] for k in by});arrays={};cals={};scoremeta=[]
    # ALL new scores and FIT/CAL gates precede old labels.
    old={}
    for domain,a in c['old_query_banks'].items():
        for d in bound(a):old[domain,d['method'],d['entity']]=d
    staged={}
    for domain,method,entity in entities:
        for layer in LAYERS:
            fi,fx=load(c,by[domain,method,entity,'FIT',layer]);ci,cx=load(c,by[domain,method,entity,'CAL',layer])
            if len(fi)!=64:raise ValueError('Frozen sameentity FIT64 required')
            if layer in ('q','qr'):
                d=old[domain,method,entity];p=loc(c,d['scores']['path'])
                if sha(p)!=d['scores']['sha256']:raise PermissionError('Canonical old q bank changed')
                with np.load(p,allow_pickle=False) as z:qi=z['index'].copy();qx=z['projection' if layer=='q' else 'reconstructed_projection'].copy()
            else:qi,qx=load(c,by[domain,method,entity,'QUERY',layer])
            start=512 if domain=='SMAP' else 8958;end=23707 if domain=='SMD' else int(qi[-1])+1
            if not np.array_equal(qi,np.arange(start,end)):raise ValueError('Full original QUERY axis required')
            for normalization in NORMALIZATIONS:
                cal=fixed_five_nn(fx,cx,reference_role='FIT',normalization=normalization);q=fixed_five_nn(fx,qx,reference_role='FIT',normalization=normalization)
                name=method+'__'+layer+'__'+normalization;key=name,domain,entity,old[domain,method,entity]['run']
                staged[key]=(qi,q,ci,cal,fi)
    for domain in ('SMAP','SMD'):
        for method in ('RoCA_FULL_T','RoCA_INIT_T'):
            for layer,normalization in [(l,n) for l in LAYERS for n in NORMALIZATIONS]:
                name=method+'__'+layer+'__'+normalization;keys=[k for k in staged if k[:2]==(name,domain)]
                validCAL=np.concatenate([staged[k][3]['score'][staged[k][3]['valid']] for k in keys])
                CAL_complete=all(staged[k][3]['valid'].all() for k in keys)
                tau=weighted_quantile(validCAL,np.ones(len(validCAL))/len(validCAL),.99) if len(validCAL) and CAL_complete else None
                for key in keys:
                    qi,q,ci,cal,fi=staged[key];valid=q['valid']
                    arrays[key]=dict(index=qi,score=q['score'],valid=valid,available_index=qi.copy(),
                        invalid_reasons=[None if v else 'ZERO_NONFINITE_OR_INSUFFICIENT_FIXED_FIT_REFERENCE' for v in valid])
                    cals[key]=tau
                    folder=out/'raw'/domain/key[2]/name;folder.mkdir(parents=True)
                    np.savez_compressed(folder/'scores_CAL_reference.npz',**{k:v for k,v in arrays[key].items() if k!='invalid_reasons'},
                        CAL_index=ci,CAL_score=cal['score'],CAL_valid=cal['valid'],FIT_index=fi)
                    scoremeta.append(dict(method=name,domain=domain,entity=key[2],FIT_valid=q['reference_valid'],
                        CAL_valid=int(cal['valid'].sum()),CAL_original=len(ci),QUERY_valid=int(valid.sum()),QUERY_original=len(qi),
                        threshold=tau,zero_or_missing_kept_UNKNOWN=True,CAL_support_complete=bool(cal['valid'].all()),
                        pooled_CAL_support_complete=CAL_complete,operational_status='AVAILABLE' if tau is not None else 'CAL_SUPPORT_INCOMPLETE'))
    dump(out/'ALL_SCORE_GATES_BEFORE_LABELS.json',scoremeta)
    olddir=Path(c['accepted_evaluation']);labels={};catalog={};results={};phases=[]
    for _,domain,entity,run_id in arrays:
        k=domain,entity,run_id
        if k in labels:continue
        p=olddir/'entities'/domain/entity/'RoCA_FULL_T';prov=json.loads((p/'provenance.json').read_text())
        if sha(p/'calculation_inputs.npz')!=prov['calculation_inputs_sha256']:raise PermissionError('Old full truth changed')
        with np.load(p/'calculation_inputs.npz',allow_pickle=False) as z:labels[k]=z['full_labels'].copy()
        with (p/'event_coverage.csv').open() as f:catalog[k]=list(csv.DictReader(f))
    for key,a in arrays.items():
        name,domain,entity,run_id=key;y=labels[key[1:]];tau=cals[key]
        if tau is None:
            # Preserve valid ranking and original event coverage; never invent
            # a placeholder operating threshold or report all-missed alarms.
            from hybrid_tsad.engine.evaluation import evaluate_point_ranking
            mask=np.zeros(len(y),bool);mask[a['index'][a['valid']]]=True
            rank=evaluate_point_ranking(a['index'][a['valid']],a['score'][a['valid']],y,mask,query_start=0)['metrics'] if mask.any() else dict(AP=None,ROC_AUC=None,reason='NO_VALID_SCORES')
            coverage=evaluation._coverage(y,mask,0)
            folder=out/'entities'/domain/entity/name;folder.mkdir(parents=True)
            np.savez_compressed(folder/'ranking_inputs_no_operating_threshold.npz',index=a['index'],score=a['score'],valid=a['valid'],full_labels=y)
            dump(folder/'result.json',dict(ranking=rank,event_coverage=coverage,streams=None,operational_status='CAL_SUPPORT_INCOMPLETE',threshold=None))
            results[key]=dict(ranking=rank,event_coverage=coverage,streams=None)
            continue
        r=evaluation.evaluate_arrays(**a,full_labels=y,threshold=tau,bank_complete=True,
            original_length=len(y),warmup_points=512 if domain=='SMAP' else 8958,affiliation_identity=c['affiliation_identity']['path'])
        reject_dependency_na(r)
        ids={(int(e['start']),int(e['end_exclusive'])):e['source_event_id'] for e in catalog[key[1:]]}
        for e in r['event_coverage']:e['source_event_id']=ids[e['start'],e['end_exclusive']]
        r['operating_threshold_available']=tau is not None
        evaluation.save_evaluation(out/'entities'/domain/entity/name,r,**a,full_labels=y,provenance=dict(config=c,layer=name))
        phases.extend(dict(method=name,domain=domain,entity=entity,**x) for x in event_diagnostics(r))
        del r['point_coverage'];results[key]=r
    dataset=[]
    for d in ('SMAP','SMD'):
        for method in ('RoCA_FULL_T','RoCA_INIT_T'):
            for l,n in [(l,n) for l in LAYERS for n in NORMALIZATIONS]:
                name=method+'__'+l+'__'+n;keys=[k for k in arrays if k[:2]==(name,d)]
                if cals[keys[0]] is not None:dataset.append(aggregate_dataset(results,arrays,labels,d,name));continue
                from hybrid_tsad.engine.evaluation import evaluate_point_ranking
                y=np.concatenate([labels[k[1:]][arrays[k]['index'][arrays[k]['valid']]] for k in keys]);s=np.concatenate([arrays[k]['score'][arrays[k]['valid']] for k in keys])
                rank=evaluate_point_ranking(np.arange(len(y)),s,y,np.ones(len(y),bool),query_start=0)['metrics'] if len(y) else dict(AP=None,ROC_AUC=None,reason='NO_VALID_SCORES')
                rank={k:v for k,v in rank.items() if 'event' not in k}
                dataset.append(dict(domain=d,method=name,ranking=rank,streams=None,threshold=None,operational_status='CAL_SUPPORT_INCOMPLETE'))
    # Strict common-valid point ranking is only a diagnostic across predeclared layers.
    from hybrid_tsad.engine.evaluation import evaluate_point_ranking
    common=[]
    for domain in ('SMAP','SMD'):
        for entity in sorted({k[2] for k in arrays if k[1]==domain}):
            keys=[k for k in arrays if k[1:3]==(domain,entity)]
            shared=np.logical_and.reduce([arrays[k]['valid'] for k in keys])
            for k in keys:
                a=arrays[k];mask=np.zeros(len(labels[k[1:]]),bool);mask[a['index'][shared]]=True
                if shared.any():metric=evaluate_point_ranking(a['index'][shared],a['score'][shared],labels[k[1:]],mask,query_start=0)['metrics']
                else:metric=dict(AP=None,ROC_AUC=None,evaluated_points=0,reason='NO_PAIRED_COMMON_VALID_POINTS')
                common.append(dict(method=k[0],domain=domain,entity=entity,common_valid_points=int(shared.sum()),**metric))
    table(out/'event_phases.csv',phases);dump(out/'PAIRED_COMMON_VALID_RANKING.json',common)
    old_normality=bound(c['accepted_normality_summary'])
    dump(out/'SUMMARY.json',dict(status='COMPLETED_STAGE_LOCALIZATION_ONLY',dataset=dataset,coverage=scoremeta,
        original_loss_n_reference=[r for r in old_normality['dataset'] if r['method'] in ('RoCA_FULL_T','RoCA_INIT_T')],
        interpretation='All six stages/INIT/FULL/raw-unit normalization reported; not best-layer model selection; invalid coverage not normal',
        original_normality_results='reuse original274 accepted SUMMARY, not reevaluated',k=5,metric='mean5NN_squaredEuclidean_direct_float64',normalizations=NORMALIZATIONS))
    dump(out/'CONFIG.json',c);dump(out/'RECEIPT.json',dict(status='COMPLETED',new_neural_forward=False,new_training=False,
        new_deployed_detector=False,GPUh=0,seconds=time.monotonic()-begin))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Frozen geometry config')
    run(json.loads(Path(a.config).read_text()))
if __name__=='__main__':main()
