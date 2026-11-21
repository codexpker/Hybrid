"""CPU-only public-normal input geometry and eligibility. No model API exists."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p,r):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(r,f,indent=2,allow_nan=False)


def write_csv(p,rows):
    with Path(p).open('x',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def envelope(n,ramp):
    if ramp<2 or n<2*ramp:raise ValueError('Two non-overlapping smooth ramps required')
    w=np.ones(n,np.float64);a=(1-np.cos(np.linspace(0,np.pi,ramp)))/2
    w[:ramp]=a;w[-ramp:]=a[::-1];return w


def delayed(base,start,duration,delay,ramp,columns):
    out=base.copy();w=envelope(duration,ramp);t=np.arange(start,start+duration,dtype=float);phi=t-delay*w
    if np.min(np.diff(phi))<=0 or phi[0]!=t[0]or phi[-1]!=t[-1]:raise ValueError('Monotone endpoint-preserving delay required')
    for c in columns:out[start:start+duration,c]=np.interp(phi,np.arange(len(base)),base[:,c]).astype(base.dtype)
    return out,phi


def energy(x,base):return float(np.square(x.astype(np.float64)-base.astype(np.float64)).sum())


def amplitude_control(base,intervention,column,rng):
    out=base.copy();delta=intervention[:,column].astype(float)-base[:,column].astype(float)
    signs=rng.choice(np.array([-1.,1.]),len(base))
    out[:,column]=np.asarray(base[:,column].astype(float)+signs*np.abs(delta),dtype=base.dtype)
    return out


def corr(x,y):
    if np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def marginal_check(x,y):
    a,b=np.sort(x),np.sort(y);u=np.unique(np.r_[a,b])
    ks=float(np.max(np.abs(np.searchsorted(a,u,side='right')/len(a)-np.searchsorted(b,u,side='right')/len(b))))
    sx,sy=float(np.std(x)),float(np.std(y));ds=[]
    for lag in (1,8,32):
        a,b=corr(x[:-lag],x[lag:]),corr(y[:-lag],y[lag:])
        ds.append(abs(a-b)if a is not None and b is not None else None)
    return dict(KS=ks,mean_change_source_sigma=float(abs(np.mean(y)-np.mean(x))),background_std_source_sigma=sx,
        std_ratio=sy/sx if sx>0 else None,max_ACF_change=max(ds)if all(v is not None for v in ds)else None)


def qualify(m,limits):
    return bool(m['KS']<=limits['KS_max']and m['mean_change_source_sigma']<=limits['mean_change_source_sigma_max']
        and m['background_std_source_sigma']>=limits['min_background_std_source_sigma']
        and m['std_ratio']is not None and limits['std_ratio_min']<=m['std_ratio']<=limits['std_ratio_max']
        and m['max_ACF_change']is not None and m['max_ACF_change']<=limits['max_ACF_change_lags_1_8_32'])


def select_pair(fit,cfg):
    continuous=[j for j in range(fit.shape[1])if len(np.unique(fit[:,j]))>=cfg['continuous_unique_values_min']]
    quarters=np.array_split(fit,cfg['quarter_splits']);rows=[];choice=None
    for a,b in itertools.combinations(continuous,2):
        levels=[corr(q[:,a],q[:,b])for q in quarters]
        changes=[corr(np.diff(q[:,a]),np.diff(q[:,b]))for q in quarters]
        good=(all(v is not None for v in levels+changes)and min(map(abs,levels))>=cfg['min_abs_level_correlation_all_quarters']
              and len(set(np.sign(levels)))==1 and np.median(np.abs(changes))>=cfg['median_abs_difference_correlation'])
        rows.append(dict(driver=a,receiver=b,level_correlations=levels,difference_correlations=changes,qualifies=bool(good)))
        if good and choice is None:choice=(a,b)
    return choice,continuous,rows


def variants(base,receiver,cfg,block_id):
    s=cfg['intervention_start'];result=[('base','base',base.copy(),None)]
    for amp,d in itertools.product(cfg['H1']['amplitudes_source_sigma'],cfg['H1']['durations']):
        w=np.sin(np.linspace(0,np.pi,d))**2;w[0]=w[-1]=0
        x=base.copy();x[s:s+d,receiver]+=np.asarray(amp*w,dtype=base.dtype)
        result.append((f'H1_a{amp}_d{d}','H1',x,dict(duration=d,envelope=w)))
    for amp,period in itertools.product(cfg['H2']['amplitudes_source_sigma'],cfg['H2']['periods']):
        d=cfg['H2']['duration'];w=envelope(d,cfg['H2']['ramp']);v=amp*w*np.sin(2*np.pi*np.arange(d)/period)
        x=base.copy();x[s:s+d,receiver]+=v.astype(base.dtype)
        result.append((f'H2_a{amp}_p{period}','H2',x,dict(duration=d,envelope=w)))
    for delay in cfg['H3']['delays']:
        d,r=cfg['H3']['duration'],cfg['H3']['ramp'];x,phi=delayed(base,s,d,delay,r,[receiver])
        result.append((f'H3_lag{delay}','H3',x,dict(duration=d,envelope=envelope(d,r),phi=phi,delay=delay)))
        x,_=delayed(base,s,d,delay,r,range(base.shape[1]))
        result.append((f'H3_sync{delay}','H3_sync_sham',x,dict(duration=d,delay=delay)))
    source=list(result)
    for i,(name,kind,x,meta)in enumerate(source):
        if kind not in ('H1','H2','H3'):continue
        rng=np.random.default_rng(np.random.SeedSequence([cfg['random_seed'],block_id,i,20261121]))
        control=amplitude_control(base,x,receiver,rng)
        result.append((name+'_amp_control','amplitude_control',control,dict(duration=meta['duration'],paired_to=name)))
    return result


def prepare(workspace,config,output):
    started=time.perf_counter();w=Path(workspace);output=Path(output);cfg=read(config)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);allcases=[];eligible=[];binding=[];normal_blocks={}
    for domain in cfg['domains']:
        taskpath=w/f'results/native_v_conditioning_20261119_v1/{domain}/TASK.json';task=read(taskpath)
        rec=task['data']['source_train'];p=w/rec['relative'];assert sha(p)==rec['sha256']
        raw=np.load(p,mmap_mode='r');assert list(raw.shape)==rec['shape']
        scalerpath=taskpath.parent/'SCALER.npz';assert sha(scalerpath)==task['scaler_sha256']
        with np.load(scalerpath,allow_pickle=False)as z:mean,std=z['mean'].copy(),z['std'].copy()
        lo,hi=task['roles']['SOURCE_FIT'];fit=np.asarray((raw[lo:hi].astype(np.float64)-mean)/std,dtype=np.float32)
        assert np.isfinite(fit).all()
        pair,continuous,pairs=select_pair(fit,cfg['pair_qualification_SOURCE_FIT_only'])
        if not continuous:raise ValueError('No source continuous receiver')
        receiver=pair[1]if pair else continuous[0]
        eligible.append(dict(domain=domain,driver=pair[0]if pair else None,receiver=receiver,
            driver_name=task['channel_order'][pair[0]]if pair else None,receiver_name=task['channel_order'][receiver],
            qualifying_pairs=sum(r['qualifies']for r in pairs),continuous_columns=continuous,
            pair_selected_from_SOURCE_FIT_only=True,H3_source_pair_qualified=pair is not None))
        dump(output/(domain+'_SOURCE_PAIRS.json'),pairs)
        normal_lo,normal_hi=task['roles']['SOURCE_SELECT']
        for block_id,offset in enumerate(cfg['block_offsets']):
            a=normal_lo+offset;b=a+cfg['block_length'];assert a>=normal_lo and b<=normal_hi
            base=np.asarray((raw[a:b].astype(np.float64)-mean)/std,dtype=np.float32);assert np.isfinite(base).all()
            block_corr=corr(base[:,pair[0]],base[:,pair[1]])if pair else None
            block_pair_ok=bool(pair is not None and block_corr is not None and abs(block_corr)>=cfg['H3_qualification_full_and_plateau']['block_pair_min_abs_level_correlation']
                and min(np.std(base[:,pair[0]]),np.std(base[:,pair[1]]))>=cfg['H3_qualification_full_and_plateau']['min_background_std_source_sigma'])
            normal_blocks[f'{domain}_block{block_id}']=base
            prepared_variants=variants(base,receiver,cfg,block_id)
            for name,kind,x,meta in prepared_variants:
                row=dict(domain=domain,block=block_id,record_id=rec['record_id'],source_start=a,source_stop=b,
                    case=name,kind=kind,receiver=receiver,receiver_name=task['channel_order'][receiver],
                    L2_energy_source_sigma=energy(x,base),all_finite=bool(np.isfinite(x).all()),
                    array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),model_forwards=0)
                if meta:
                    s,d=cfg['intervention_start'],meta['duration'];r=x.astype(float)-base.astype(float)
                    row.update(intervention_start=a+s,intervention_stop=a+s+d,
                        outside_support_exact=bool(np.all(r[:s]==0)and np.all(r[s+d:]==0)),
                        endpoints_exact=bool(np.all(r[s]==0)and np.all(r[s+d-1]==0)))
                    assert row['outside_support_exact']and row['endpoints_exact']
                    if kind not in ('H3_sync_sham',):
                        other=[c for c in range(base.shape[1])if c!=receiver];assert np.array_equal(x[:,other],base[:,other])
                    if kind=='amplitude_control':
                        target=next(z['L2_energy_source_sigma']for z in allcases if z['domain']==domain and z['block']==block_id and z['case']==meta['paired_to'])
                        row['paired_to']=meta['paired_to'];row['relative_energy_error']=abs(row['L2_energy_source_sigma']-target)/target if target else None
                        assert target==0 or row['relative_energy_error']<=1e-5
                        original=next(v[2]for v in prepared_variants if v[0]==meta['paired_to'])
                        ca,cb=cfg['H3']['primary_core']if meta['paired_to'].startswith('H3')else cfg['H2']['primary_core']
                        if not meta['paired_to'].startswith('H1'):
                            ec,ei=energy(x[ca:cb],base[ca:cb]),energy(original[ca:cb],base[ca:cb])
                            row['core_relative_energy_error']=abs(ec-ei)/ei if ei else None
                            assert ei==0 or row['core_relative_energy_error']<=1e-5
                    if kind=='H3':
                        ramp=cfg['H3']['ramp'];checks=[]
                        for label,aa,bb in (('full',s,s+d),('plateau',s+ramp,s+d-ramp)):
                            m=marginal_check(base[aa:bb,receiver],x[aa:bb,receiver]);checks.append(qualify(m,cfg['H3_qualification_full_and_plateau']))
                            row.update({label+'_'+k:v for k,v in m.items()})
                        edge=float(max(abs(r[s+1,receiver]),abs(r[s+d-2,receiver])))
                        core_a,core_b=cfg['H3']['primary_core'];delay=meta['delay']
                        assert core_a-514>=s+ramp and core_b<=s+d-ramp
                        assert np.array_equal(x[core_a-514:core_b,receiver],base[core_a-delay-514:core_b-delay,receiver])
                        row.update(primary_core_start=a+core_a,primary_core_stop=a+core_b,
                            core_pure_integer_delay_exact=True,extra_edge_displacement=edge,
                            phi_step_min=float(np.min(np.diff(meta['phi']))),block_pair_correlation=block_corr,block_pair_qualified=block_pair_ok,
                            core_intervention_energy=energy(x[core_a:core_b],base[core_a:core_b]),
                            H3_valid=bool(block_pair_ok and all(checks)and energy(x[core_a:core_b],base[core_a:core_b])>0 and edge<=cfg['H3_qualification_full_and_plateau']['edge_extra_displacement_source_sigma_max']))
                    if kind=='H3_sync_sham':
                        core_a,core_b=cfg['H3']['primary_core'];delay=meta['delay']
                        assert np.array_equal(x[core_a-514:core_b],base[core_a-delay-514:core_b-delay])
                        row['core_entire_joint_input_and_target_equal_earlier_base']=True
                allcases.append(row)
        binding.append(dict(domain=domain,source_relative=rec['relative'],source_sha256=rec['sha256'],scaler_sha256=task['scaler_sha256'],
            SOURCE_FIT=[lo,hi],SOURCE_SELECT=[normal_lo,normal_hi],value_access_stop=normal_lo+cfg['block_offsets'][-1]+cfg['block_length'],
            source_normality='Existing public normal-train background; no new point-health certification',QUERY_or_CAL_values_labels_read=False))
    np.savez_compressed(output/'NORMAL_BLOCKS.npz',**normal_blocks)
    write_csv(output/'CASES.csv',allcases);dump(output/'SOURCE_QUALIFICATION.json',eligible)
    receipt=dict(status='CPU_INPUT_GEOMETRY_CHECKED_NOT_MODEL_EFFECT',config=cfg,config_sha256=sha(config),source_sha256=sha(__file__),
        bindings=binding,cases=len(allcases),blocks=len(normal_blocks),model_forwards=0,training=0,GPU_hours=0,torch_imported='torch'in sys.modules,
        CPU_wall_seconds=time.perf_counter()-started,
        H3_valid_counts={d:{str(delay):sum(r.get('H3_valid',False)for r in allcases if r['domain']==d and r['case']==f'H3_lag{delay}')for delay in cfg['H3']['delays']}for d in cfg['domains']},
        inference_release=False)
    assert not receipt['torch_imported'];dump(output/'PREPARATION.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=prepare(a.workspace,a.config,a.out);print(json.dumps({k:r[k]for k in ('status','blocks','cases','H3_valid_counts','CPU_wall_seconds','GPU_hours','inference_release')},indent=2))
