"""Fixed U/Z grid, equal-entity macro and shared across-fault run bootstrap."""
import numpy as np

MODES=('temporal','tf','hybrid')
NATIVE=tuple(('Z','source_pretrained',m) for m in MODES)+tuple(('U_NORMAL_HEAD',i,m) for i in ('source_pretrained','random_init') for m in MODES)
LEGACY=tuple(('Z','source_pretrained',m) for m in ('late','history_mean','persistence'))
CONFIGS=NATIVE+LEGACY


def tracks(config):return ('source','target') if config in NATIVE else ('source',)


def fixed_grid(profile):
    faults,runs=profile['faults'],profile['runs']
    if not faults or not runs or len(set(faults))!=len(faults) or len(set(runs))!=len(runs):raise ValueError('Unique complete fault/run grid required')
    if profile['name']=='TEP100' and (faults!=list(range(1,21)) or runs!=list(range(31,51))):raise ValueError('Real evaluation preserves all20faults/20runs')
    if profile['name'] not in ('TEP100','SYNTHETIC'):raise ValueError('Unknown evaluation profile')
    return {(c,t,s,f,r) for c in CONFIGS for t in tracks(c) for s in range(3) for f in faults for r in runs}


def run_statistics(rows,profile):
    expected=fixed_grid(profile);found={}
    for row in rows:
        c=(row['stage'],row['initialization'],row['method']);key=(c,row['threshold_track'],row['seed'],row['fault_id'],row['run_id'])
        if key not in expected or key in found:raise ValueError('Unknown/duplicate stage/init/track statistics cell')
        found[key]=row
    if set(found)!=expected:raise ValueError('INCOMPLETE evaluation statistics grid')
    faults,runs=profile['faults'],profile['runs'];arrays={};macro=[];entities=[];per_fault=[];per_run=[];worst=[];failures=[]
    pi=None
    for c in CONFIGS:
        for track in tracks(c):
            base=dict(stage=c[0],initialization=c[1],method=c[2],threshold_track=track)
            for metric in ('AP','ROC'):
                a=np.asarray([[[found[c,track,s,f,r][metric] for r in runs] for f in faults] for s in range(3)],dtype='float64')
                if np.any(np.isfinite(a)&((a<0)|(a>1))):raise ValueError('Rank metric outside[0,1]')
                arrays[c,track,metric]=a;entity=a.mean(0);seed=a.mean((1,2))
                macro.append(dict(**base,metric=metric,mean=float(entity.mean()),per_seed=seed.tolist(),sample_std=float(seed.std(ddof=1)),n_seeds=3,independent_datasets=1))
                for fi,f in enumerate(faults):
                    for ri,r in enumerate(runs):entities.append(dict(**base,metric=metric,fault_id=f,run_id=r,mean=float(entity[fi,ri])))
                    per_fault.append(dict(**base,metric=metric,fault_id=f,mean=float(entity[fi].mean())))
                for ri,r in enumerate(runs):per_run.append(dict(**base,metric=metric,run_id=r,mean=float(entity[:,ri].mean())))
                valid=np.isfinite(entity);minimum=np.unravel_index(np.where(valid,entity,np.inf).argmin(),entity.shape) if valid.any() else None
                worst.append(dict(**base,metric=metric,worst_entity=float(entity[valid].min()) if valid.any() else float('nan'),
                    worst_entity_fault_id=faults[minimum[0]] if minimum else None,worst_entity_run_id=runs[minimum[1]] if minimum else None,
                    worst_seed=float(seed.min()),worst_seed_id=int(seed.argmin()) if np.isfinite(seed).all() else None,
                    worst_fault=float(entity.mean(1).min()),worst_run=float(entity.mean(0).min()),
                    invalid_entities=int((~valid).sum()),tie_rule='first frozen-order minimum; full table retains ties'))
            for s in range(3):
                for f in faults:
                    for r in runs:
                        row=found[c,track,s,f,r];fraction=row['positive_fraction']
                        if not np.isfinite(fraction) or not 0<fraction<1:raise ValueError('Invalid scored label population')
                        if pi is None:pi=fraction
                        if fraction!=pi:raise ValueError('Different label populations across compared cells')
                        if profile['name']=='TEP100' and fraction!=800/860:raise ValueError('Frozen TEP background800/860 changed')
                        if not np.isfinite(row['AP']) or not np.isfinite(row['ROC']):failures.append(dict(**base,seed=s,fault_id=f,run_id=r,reason='NONFINITE_ENTITY_AP_OR_ROC'))
            if c in NATIVE and track=='target':
                for s in range(3):
                    for f in faults:
                        for r in runs:
                            source=found[c,'source',s,f,r];target=found[c,'target',s,f,r]
                            if source['rank_object_id']!=target['rank_object_id'] or any(not np.array_equal(source[k],target[k],equal_nan=True) for k in ('AP','ROC')):
                                raise ValueError('S/T rank inputs/results must be one identical canonical object')
    comparisons=[];differences=[]
    for mode in MODES:
        u=('U_NORMAL_HEAD','source_pretrained',mode)
        for other in (('Z','source_pretrained',mode),('U_NORMAL_HEAD','random_init',mode)):
            comparisons.append(dict(mode=mode,left=u,right=other,threshold_track='target'))
            differences.append((arrays[u,'target','AP']-arrays[other,'target','AP']).mean(0).mean(0))
    differences=np.stack(differences,axis=1)
    valid_run=np.stack([np.isfinite(a).all((0,1)) for a in arrays.values()]).all(0)
    differences[~valid_run]=np.nan
    draws=np.random.default_rng(99174).integers(0,len(runs),size=(2000,len(runs)))
    delta=np.full((2000,6),np.nan)
    for i,draw in enumerate(draws):
        if valid_run[draw].all():delta[i]=differences[draw].mean(0)
    intervals=[]
    for j,c in enumerate(comparisons):
        valid=np.isfinite(delta[:,j]);observed=float(differences[:,j].mean());limits=np.quantile(delta[valid,j],[.025,.975]) if valid.any() and np.isfinite(observed) else [np.nan,np.nan]
        intervals.append(dict(**c,observed_mean_AP_difference=observed,ci95_low=float(limits[0]),ci95_high=float(limits[1]),replicates=2000,
            valid_replicates=int(valid.sum()),invalid_replicates=int((~valid).sum()),clusters=len(runs),rng_seed=99174,
            scope='exploratory paired run-cluster sensitivity; no multiplicity adjustment'))
    return dict(status='COMPLETE' if not failures else 'INCOMPLETE',macro=macro,entity_seed_mean=entities,per_fault=per_fault,per_run=per_run,worst=worst,failures=failures,
        intervals=intervals,overall_Aff_F1=None,bootstrap_arrays=dict(draws=draws,AP_difference=delta,paired_run_difference=differences,run_ids=np.asarray(runs),valid_run=valid_run,rng_seed=np.asarray(99174)))


def delay_summary(events):
    groups={}
    for e in events:
        key=tuple(e[k] for k in ('stage','initialization','method','threshold_track','seed'));groups.setdefault(key,[]).append(e)
    result=[]
    for key,rows in groups.items():
        eligible=[e for e in rows if e['complete']];detected=[e for e in eligible if e['any_hit']];delays=[e['delay_minutes'] for e in detected]
        result.append(dict(zip(('stage','initialization','method','threshold_track','seed'),key))|dict(eligible=len(eligible),detected=len(detected),missed=len(eligible)-len(detected),incomplete=len(rows)-len(eligible),
            median_detected_only_delay_minutes=float(np.median(delays)) if delays else None,median_reason=None if delays else 'NO_DETECTED_COMPLETE_EVENT',
            all_eligible_mean_delay=None,all_eligible_delay_reason='NOT_CONTRACTED_NO_MISS_IMPUTATION',denominator='all eligible original events retained; median is detected-only'))
    return result


def normal_prefix_summary(observations,profile):
    groups={}
    for o in observations:
        key=tuple(o[k] for k in ('stage','initialization','method','threshold_track','seed','run_id'));groups.setdefault(key,[]).append(o)
    result=[]
    for key,rows in groups.items():
        if len(rows)!=len(profile['faults']) or {r['fault_id'] for r in rows}!=set(profile['faults']):raise ValueError('Missing/duplicate normal-prefix fault group')
        feature_equal=len({r['feature_prefix_sha256'] for r in rows})==1;score_equal=len({r['score_prefix_sha256'] for r in rows})==1;binary_equal=len({r['binary_prefix_sha256'] for r in rows})==1
        counts={r['normal_points'] for r in rows}
        if len(counts)!=1:raise ValueError('Normal-prefix observation denominator differs')
        n=counts.pop();hours=n*profile['sample_interval_minutes']/60
        valid=feature_equal and binary_equal
        fp=[r['normal_false_points'] for r in rows];alarms=[r['normal_alarm_fragments'] for r in rows]
        if binary_equal and (len(set(fp))!=1 or len(set(alarms))!=1):raise ValueError('Identical binary prefix has inconsistent counts')
        result.append(dict(zip(('stage','initialization','method','threshold_track','seed','run_id'),key))|dict(
            source_entities=[r['entity'] for r in rows],feature_prefix_equal=feature_equal,score_prefix_equal=score_equal,binary_prefix_equal=binary_equal,
            per_entity_normal_points=n,per_entity_normal_hours=hours,duplicated_points=n*len(rows),duplicated_hours=hours*len(rows),
            unique_normal_points=n if valid else None,unique_normal_hours=hours if valid else None,
            unique_normal_false_points=sum(fp)//len(fp) if valid else None,unique_normal_alarm_fragments=sum(alarms)//len(alarms) if valid else None,
            unique_normal_FPR=(sum(fp)//len(fp))/n if valid and n else None,
            unique_normal_alarm_fragments_per_hour=(sum(alarms)//len(alarms))/hours if valid and hours else None,
            reason=None if valid else 'NORMAL_FEATURE_OR_BINARY_PREFIX_NOT_IDENTICAL_NO_REPRESENTATIVE_SELECTED'))
    return result
