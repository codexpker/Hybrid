"""H5 offline donor admission/plan construction. NumPy only, no learner.

Numeric labels live exclusively in offline audit state; emitted learning plans
carry only original record/index/footprint metadata. No real execution by default.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

OPERATION='HYBRID_V1_CONTAMINATION_DATA'
EVENTS=[(15849,16395),(16963,17517),(18071,18528),(19367,20088),(20786,21195),(24679,24682),(26114,26116),(27554,27556)]
IDENTITY=dict(domain='SMD',entity='machine-1-1',run='machine-1-1_test',role='FIT')
RAW={'values':('bdb801d8d52ce6b0c4a20c310edf31b5cee9bc82eb84fb38e7d4a19309b16ddb',9739818),
     'labels':('cddf33441ae480c6074e173c897d4715eb893f24a7078d3d3e1b79b200c7cd19',56958)}
BASE={'manifest':'2878011215391279d69ad9b1bb5c4732aed348e33b026ff67ab8f34baf990b76',
      'plan':'243c5ccf7026222d2c406811bc143f01322a83523f8b583d3e4959db5b53ec93'}
CLOSURE={'hybrid_tsad/__init__.py','hybrid_tsad/hybrid_v1/__init__.py','hybrid_tsad/hybrid_v1/config.py',
         'hybrid_tsad/hybrid_v1/provenance.py','hybrid_tsad/hybrid_v1/contamination_data.py'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def no_supervision(x):
    if isinstance(x,dict):
        if any(any(w in k.lower() for w in ('label','event','weight','positive','negative')) for k in x):
            raise ValueError('Supervision metadata forbidden in learning input')
        for v in x.values():no_supervision(v)
    elif isinstance(x,list):
        for v in x:no_supervision(v)

def validate_base_manifest(manifest):
    # The byte-pinned original2878 manifest already has this anti-label
    # declaration. It is not a label path; preserve base0 original bytes while
    # omitting ALL top-level declarations from newly derived learner manifests.
    metadata=dict(manifest)
    if 'label_files_not_supported' in metadata:
        if metadata.pop('label_files_not_supported') is not True:raise ValueError('Legacy anti-label declaration must remain true')
    no_supervision(metadata)

def exclusive(identity,forbidden):
    if identity!=IDENTITY:raise ValueError('Only reserved SMD1-1 test donor FIT record allowed')
    key=lambda x:tuple(x[k] for k in ('domain','entity','run'))
    if any(key(x)==key(identity) for x in forbidden):raise ValueError('Donor overlaps source SELECT/CAL, query or F')

def catalogue(labels):
    if labels.ndim!=1 or not len(labels) or not np.isin(labels,[0,1]).all():raise ValueError('Exact 1D binary labels required')
    d=np.diff(np.r_[0,labels,0].astype(np.int8))
    return list(zip(np.flatnonzero(d==1).tolist(),np.flatnonzero(d==-1).tolist()))

def admit(values,labels,expected,history=512,horizon=32):
    """Array API for authorized offline data or synthetic tests; never impute."""
    v=np.asarray(values,dtype=np.float32);y=np.asarray(labels)
    if v.ndim!=2 or len(v)!=len(y) or np.isinf(v).any() or history<3 or horizon<1:raise ValueError('Native finite-or-NaN array/axis/H/P required')
    actual=catalogue(y)
    if actual!=[tuple(x) for x in expected]:raise ValueError('Complete event directory mismatch; no repair: expected=%r actual=%r'%(expected,actual))
    obs=np.isfinite(v);N=len(v);ts=np.arange(history,N-horizon+1,dtype=np.int64)
    cs=np.concatenate([np.zeros((1,v.shape[1]),np.int64),obs.cumsum(0,dtype=np.int64)])
    good=obs[ts].all(1)&((cs[ts]-cs[ts-history+1])>=2).all(1)
    lc=np.r_[0,np.cumsum(y,dtype=np.int64)];normal=ts[good&((lc[ts+horizon]-lc[ts-history])==0)]
    positives=[];events=[]
    for i,(s,e) in enumerate(expected):
        a,b=s-history,e+horizon-1
        if not 0<=a<b<=N or any(a<oe and os<b for j,(os,oe) in enumerate(expected) if j!=i):
            raise ValueError('Complete expanded event footprint crosses record/other event')
        origins=ts[good&(ts>=s)&(ts<e)];positives.extend(origins.tolist())
        events.append(dict(event_id='SMD:machine-1-1:E%d'%i,start=s,end=e,expanded=[a,b],
                           left_clearance=a-(expected[i-1][1] if i else 0),
                           right_clearance=(expected[i+1][0] if i+1<len(expected) else N)-b,
                           eligible_origins=len(origins),original_points=e-s))
    return dict(values=v,observed=obs,labels=y.astype(np.int8),positive=np.array(positives,np.int64),normal=normal,
                events=events,history=history,horizon=horizon,missing_cells=int((~obs).sum()),
                observed_definition='isfinite(canonical benchmark values), not physical missingness truth')

def count_for_rate(rate,smd_slots=3982):
    if type(rate) is not int or rate not in (0,1,5,10,20) or smd_slots!=3982:raise ValueError('Fixed percent rates and3982 denominator required')
    return (smd_slots*rate+50)//100

def nearest_unique(positive,normal):
    pool=set(int(t) for t in normal);result=[]
    for p in positive:
        if not pool:break
        n=min(pool,key=lambda t:(abs(t-int(p)),t));pool.remove(n);result.append(n)
    return result

def exposure(a,origins):
    y,o=a['labels'],a['observed'];H,P=a['history'],a['horizon'];visits=np.zeros(len(y),np.int64);rows=[]
    for origin in origins:
        t=int(origin);row=dict(index=t)
        for name,lo,hi in [('current',t,t+1),('future_GT',t,t+P),('input_context',t-H,t),('reconstruction_context',t-H+1,t+1),('footprint',t-H,t+P)]:
            row[name]=dict(points=hi-lo,anomaly_points=int(y[lo:hi].sum()),observed_cells=int(o[lo:hi].sum()),
                           anomaly_observed_cells=int((o[lo:hi]&y[lo:hi,None].astype(bool)).sum()))
        row['exposed_event_ids']=[e['event_id'] for e in a['events'] if t-H<e['end'] and e['start']<t+P]
        visits[t-H:t+P]+=1;rows.append(row)
    edges=np.diff(np.r_[False,visits>0,False].astype(np.int8))
    intervals=[[int(a),int(b)] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    return dict(origins=rows,unique_origins=len(set(int(t) for t in origins)),unique_footprint_points=int((visits>0).sum()),
                unique_footprint_intervals=intervals,all_exposed_event_ids=sorted({e for row in rows for e in row['exposed_event_ids']}),
                repeated_point_exposures=int(np.maximum(visits-1,0).sum()),unique_anomaly_points=int(y[visits>0].sum()),
                unique_observed_cells=int(o[visits>0].sum()),actual_updated_skipped='NOT_RUN',base_true_contamination='UNKNOWN')

def construct_plan(manifest,plan,a,donor,forbidden=()):
    """No replacement, fixed permutations, nested rates and normal control."""
    exclusive({k:donor[k] for k in IDENTITY},forbidden);validate_base_manifest(manifest);no_supervision(plan);no_supervision(donor)
    exclusive({k:donor[k] for k in IDENTITY},list(forbidden)+[r for r in manifest['runs'] if r.get('role')!='FIT']+plan['SELECT']+plan['CAL'])
    if (plan.get('history'),plan.get('horizon'),plan.get('seed'))!=(a['history'],a['horizon'],2025):raise ValueError('Base H/P/seed mismatch')
    fit=plan['FIT'];slots=np.array([i for i,x in enumerate(fit) if x['domain']=='SMD'],np.int64)
    if len(fit)!=8000 or len(slots)!=3982 or sum(x['domain']=='Sunspot' for x in fit)!=4018:raise ValueError('8000/3982/4018 denominator/order contract required')
    if len(plan['SELECT'])!=64 or len(plan['CAL'])!=1024:raise ValueError('Fixed64SELECT/1024CAL required')
    if any(x['role']!='FIT' for x in fit):raise ValueError('Only FIT slots replaceable')
    ordered_slots=np.random.default_rng(99179).permutation(slots)
    pos=np.random.default_rng(99180).permutation(np.sort(a['positive']))[:796]
    normals=nearest_unique(pos,a['normal']);panels={};ledger={}
    donor_identity={k:donor[k] for k in IDENTITY}
    new_manifest=dict(runs=copy.deepcopy(manifest['runs']))
    if any(tuple(r.get(k) for k in IDENTITY)==tuple(donor_identity[k] for k in IDENTITY) for r in manifest['runs']):raise ValueError('Donor already present in base learning manifest')
    new_manifest['runs'].append(copy.deepcopy(donor))
    for rate in (0,1,5,10,20):
        n=count_for_rate(rate)
        for mode in (['anomaly','normal_record'] if rate==10 else ['anomaly']):
            name=('base0' if rate==0 else mode+str(rate));origins=pos if mode=='anomaly' else normals
            if len(pos)<n or len(origins)<n:
                ledger[name]=dict(status='NA',required=n,positive_available=len(pos),normal_available=len(normals),reason='Insufficient unique eligible origins; no fallback');continue
            p=copy.deepcopy(plan);changes=[]
            for attempt,t in zip(ordered_slots[:n],origins[:n]):
                attempt=int(attempt);t=int(t);entry=dict(donor_identity,index=t,footprint=[t-a['history'],t+a['horizon']])
                changes.append(dict(attempt=attempt,base=copy.deepcopy(fit[attempt]),donor=entry,
                                    positive_origin=int(pos[len(changes)]),normal_match=int(normals[len(changes)]) if len(normals)>len(changes) else None))
                p['FIT'][attempt]=entry
            no_supervision(p)
            panels[name]=dict(manifest=copy.deepcopy(manifest) if n==0 else copy.deepcopy(new_manifest),plan=p)
            ledger[name]=dict(status='AVAILABLE',requested_SMD_percent=rate,n=n,SMD_denominator=3982,all_source_denominator=8000,
                              actual_SMD_fraction=n/3982,actual_all_source_fraction=n/8000,changes=changes,
                              matching=[dict(positive=int(pp),normal=int(nn),distance=abs(int(pp)-int(nn))) for pp,nn in zip(pos[:n],normals[:n])],
                              exposure=exposure(a,origins[:n]),base_contamination='UNKNOWN',actual_updated_skipped='NOT_RUN')
    return dict(panels=panels,auditledger=ledger,
                matrix=[dict(data=d,loss=loss) for d in ('base0','anomaly10','normal_record10') for loss in ('huber','half_mse')],
                reuse_Huber0='Only exact original manifest/plan bytes, initialization and optimization exposure; no clean-base claim')

def verify_raw(inputs):
    # Check BOTH complete identities before any numeric parser is called.
    for key,(digest,size) in RAW.items():
        s=inputs[key]
        if s['sha256']!=digest or s['bytes']!=size:raise PermissionError('Frozen two-file identity changed')
        if Path(s['path']).stat().st_size!=size or sha(s['path'])!=digest:raise ValueError('Donor original byte identity mismatch')

def execute_admit(c):
    verify_raw(c['inputs']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    try:
        v=np.genfromtxt(c['inputs']['values']['path'],delimiter=',',ndmin=2)
        y=np.genfromtxt(c['inputs']['labels']['path'],ndmin=1)
        if v.shape!=(28479,38) or y.shape!=(28479,):raise ValueError('Native donor N/C/axis mismatch')
        verify_raw(c['inputs'])  # Complete identity before AND after single parse.
        a=admit(v,y,EVENTS);np.savez_compressed(out/'donor_values_observed.npz',values=a['values'],observed=a['observed'],original_index=np.arange(len(v),dtype=np.int64))
        np.savez_compressed(out/'OFFLINE_AUDIT_ONLY.npz',labels=a['labels'],observed=a['observed'],positive=a['positive'],normal=a['normal'])
        write(out/'OFFLINE_annotation_ledger.json',dict(points=28479,events=a['events'],normal_full_footprint_screening=True,labels_in_SSL=False,base_true_contamination='UNKNOWN'))
        receipt=dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',shape=[28479,38],values_dtype='float32',observed_dtype='bool',observed_per_channel=a['observed'].sum(0).tolist(),observed_definition=a['observed_definition'],events=a['events'],history=512,horizon=32,positive_count=len(a['positive']),normal_count=len(a['normal']),missing_cells=a['missing_cells'],input_identity_matched_before_and_after_parse=True,inputs=c['inputs'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()},physical_record_loaded=True,training=False)
        write(out/'receipt.json',receipt);return receipt
    except BaseException as e:
        write(out/'FAILURE.json',dict(status='FAILED',error=repr(e),no_fallback=True));raise

def execute_construct(c):
    # Construct is a NEW authorization. It cannot silently follow admission.
    acceptance=read(c['accepted_admission'])
    if (sha(c['accepted_admission'])!=c['accepted_admission_sha256'] or acceptance.get('approved_by')!='root'
        or acceptance.get('status')!='ACCEPTED_LIMITED_DONOR_ADMISSION' or acceptance.get('receipt_sha256')!=sha(c['admission_receipt'])):
        raise PermissionError('Accepted actual donor admission required')
    receipt=read(c['admission_receipt']);folder=Path(c['admission_receipt']).parent
    for key in ('manifest','plan'):
        if c[key+'_sha256']!=BASE[key] or sha(c[key])!=BASE[key]:raise PermissionError('Original base input identity changed')
    for name,digest in receipt['files'].items():
        if sha(folder/name)!=digest:raise ValueError('Accepted admission artifact changed')
    with np.load(folder/'OFFLINE_AUDIT_ONLY.npz',allow_pickle=False) as private:
        a={k:private[k].copy() for k in ('labels','observed','positive','normal')}
    a.update(events=receipt['events'],history=512,horizon=32)
    values=folder/'donor_values_observed.npz'
    donor=dict(IDENTITY,path=str(values.resolve()),format='npz',key='values',observed_key='observed',sha256=receipt['files'][values.name],bounds=[0,28479],original_start=0,expected_shape=[28479,38])
    result=construct_plan(read(c['manifest']),read(c['plan']),a,donor,c['forbidden_records'])
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    for name,panel in result['panels'].items():
        d=out/name;d.mkdir()
        if name=='base0':
            # Preserve ORIGINAL BYTES, not a JSON reserialization approximation.
            (d/'manifest.json').write_bytes(Path(c['manifest']).read_bytes());(d/'plan.json').write_bytes(Path(c['plan']).read_bytes())
        else:write(d/'manifest.json',panel['manifest']);write(d/'plan.json',panel['plan'])
    write(out/'OFFLINE_auditledger.json',result['auditledger']);write(out/'MATRIX.json',result['matrix'])
    return dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',panels=list(result['panels']),source_training=False)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--phase',required=True,choices=['donor_admit','construct_plan']);p.add_argument('--execute',action='store_true');p.add_argument('--release');p.add_argument('--trusted-release-sha256');args=p.parse_args(argv)
    c=read(args.config)
    if c.get('operation')!=OPERATION:raise PermissionError('H5 independent operation required')
    exclusive(c['identity'],c['forbidden_records'])
    if not args.execute:print(json.dumps(dict(preview=True,phase=args.phase,numeric_read=False)));return
    if Path(c['output']).exists():raise FileExistsError('Fresh output/no retry required')
    if not args.release or not args.trusted_release_sha256 or sha(args.release)!=args.trusted_release_sha256:raise PermissionError('Root phase-specific release required')
    r=read(args.release);closure=c['source_sha256'];base=Path(__file__).resolve().parents[2]
    if r.get('status')!='APPROVED' or r.get('approved_by')!='root' or r.get('operation')!=OPERATION or r.get('allowed_actions')!=[args.phase] or r.get('config_sha256')!=sha(args.config) or r.get('source_sha256')!=closure or set(closure)!=CLOSURE:raise PermissionError('Changed phase/config/cold source closure')
    for name,digest in closure.items():
        f=(base/name).resolve()
        if base not in f.parents or sha(f)!=digest:raise PermissionError('Source closure changed')
    print(json.dumps((execute_admit if args.phase=='donor_admit' else execute_construct)(c)))

if __name__=='__main__':main()
