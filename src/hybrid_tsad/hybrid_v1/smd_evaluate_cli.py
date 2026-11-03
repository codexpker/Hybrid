"""Fixed SMD fourteen-method CPU profile; original evaluator/metrics unchanged."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
from . import named_evaluate_cli as named,normal_readout as query,smd_recalibration as modelview,smd_baselines as baseline
from . import bank_recalibration as bank
from .chunked_bank import read_records
from .evaluation import evaluate_arrays,save_evaluation

OPERATION='HYBRID_V1_SMD14_FIXED_EVALUATION'
PROFILE='SMD21_F_MATCHED_FOURTEEN'
LABEL_SHA='416dd20a447f38b8d9394382293ed44f509c1c3f99c4aa2fbff179498626dbdc'
F_RAW_ACCEPT_SHA='d3fcdc3de6042eb2efa7e4b2e0d33f7ba0391c3b216512386edc59c9fb6dc9b6'
ZT_ACCEPT_SHA='104ab7053bc799bc4a6e16d4d55d59835c5af6e41ab385580e73e595a694f9d2'
CHOICES=('K1_choice0','K1_choice1','K1_choice2','K3_choice3')
ZS='Z_SOURCE_R3';ZT='Z_TCAL_R3'
F={c:'F300_'+c+'_EVENT_LOGIT' for c in CHOICES};N={c:'N_NEG300_'+c+'_EVENT_LOGIT' for c in CHOICES}
BASE={m:{'source':m+'_SOURCE','target':m+'_TCAL'} for m in baseline.METHODS}
F_CHECKPOINTS=dict(zip(CHOICES,('b3d6e37e1a0ce26a70780d1fb7716e4bcde868780d5bc55280b2fdfdd80b4ee0','87021d6f5305942176c82d6b78e4d3707ca20a1ffec5db4d1c7f1523572c518c','e5fc88e343af42d2631b39617fdc1d8e1ab6dc9bb7cc9d615f36c45ade89cc69','b63f3fc8246f238956628b536bbeac5a4d2b0dd943d74c7632fa9dd362069c3d')))
IDS={ZS,ZT}|set(F.values())|set(N.values())|{v for d in BASE.values() for v in d.values()}
PAIRS=[dict(reference=ZS,candidate=ZT,pairing='calibration_only_same_raw')]+[
    dict(reference=N[c],candidate=F[c],pairing='same_support_choice') for c in CHOICES]+[
    dict(reference=ZT,candidate=F[c],pairing='end_to_end_vs_calibration') for c in CHOICES]
RUNTIME_FILES=baseline.RUNTIME_FILES|named.REQUIRED_SOURCE|{'hybrid_tsad/hybrid_v1/'+x+'.py' for x in ('named_evaluate_cli','smd_evaluate_cli')}
sha=query.sha
read=query.read
write=query.write

def protect_label_paths(cfg):
    label=Path(cfg['labels']['path']).resolve()
    def walk(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key=='labels':continue
                if key=='path' and isinstance(item,str) and Path(item).resolve()==label:raise PermissionError('Label alias before label stage')
                if key=='directory' and isinstance(item,str) and (Path(item)/'manifest.json').resolve()==label:raise PermissionError('Label alias as manifest')
                walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    walk(cfg)
    if cfg.get('affiliation_identity') and Path(cfg['affiliation_identity']).resolve()==label:raise PermissionError('Label alias as official identity')

def guard_manifest_files(entry,cfg):
    label=Path(cfg['labels']['path']).resolve();directory=Path(entry['directory']);mp=directory/'manifest.json'
    if mp.resolve()==label:raise PermissionError('Label alias manifest')
    m=modelview.metadata(dict(path=str(mp),sha256=entry['manifest_sha256']))
    for chunk in m.get('chunks',[]):
        if (directory/chunk['file']).resolve()==label:raise PermissionError('Label alias raw chunk')

def acceptance_matches(entry,document):
    if document.get('approved_by')!='root' or not document.get('status','').startswith('ACCEPTED'):raise PermissionError('Root actual acceptance required')
    case=document
    if isinstance(document.get('methods'),dict):case=document['methods'].get(entry.get('method') or entry['id'],{})
    elif isinstance(document.get('methods'),list):case=next((x for x in document['methods'] if x.get('method')==entry.get('method') or x.get('id')==entry['id']),{})
    elif isinstance(document.get('cases'),list):case=next((x for x in document['cases'] if x.get('id')==entry['id']),next((x for x in document['cases'] if entry.get('choice') and x.get('choice')==entry['choice']),{}))
    direct=case.get('bank_manifest_sha256',case.get('manifest_sha256'))==entry['manifest_sha256']
    # The already accepted F group lists an integer count, not case records.
    # The whole root14 seal binds each actual manifest; F weights are choice-pinned.
    group=(entry['kind']=='fewshot_event_model' and entry['acceptance']['sha256']==F_RAW_ACCEPT_SHA and document.get('cases')==4 and document.get('records')==24212)
    accepted_ZT=(entry['id']==ZT and entry['acceptance']['sha256']==ZT_ACCEPT_SHA and document.get('records')==6053
                 and document.get('model_or_fit') is False and document.get('raw_producer_unchanged') is True)
    return direct or group or accepted_ZT

def contract(cfg):
    if (cfg.get('operation')!=OPERATION or cfg.get('profile')!=PROFILE or cfg.get('statistics')!=named.STATISTICS_NA
        or cfg.get('original_start')!=17129 or cfg.get('original_length')!=6565 or cfg.get('history')!=512
        or cfg.get('stream')!=['SMD','machine-2-1','machine-2-1_test'] or cfg.get('comparisons')!=PAIRS
        or any(k in cfg for k in ('model_seeds','ensemble','aggregate','seed_groups','model_mean_SD'))):raise ValueError('Exact fourteen methods/nine pairs, not seeds/ensemble')
    entries=cfg['banks']
    if len(entries)!=14 or {e['id'] for e in entries}!=IDS or len({e['directory'] for e in entries})!=14:raise ValueError('All distinct fourteen bank views required')
    for e in entries:
        if e.get('ranges')!=modelview.RANGES or e['identity'].get('input_sha256')!=query.helper.RAW_SHA:raise ValueError('Complete6053 original axis only')
        id=e['id'];origin='source' if id==ZS or id in [x['source'] for x in BASE.values()] else 'target'
        if e.get('calibration_origin')!=origin:raise ValueError('SOURCE/TCAL permissions cannot masquerade')
        b=next((m for m,d in BASE.items() if id in d.values()),None)
        if b:
            if e.get('kind')!='deterministic_baseline' or e.get('method')!=b or e.get('training_seed') is not None or e.get('inference_seed') is not None or e['identity'].get('checkpoint_sha256')!=baseline.algorithm(b):raise ValueError('Baseline is not a trained model/seed')
        elif e.get('training_seed')!=2025 or e.get('inference_seed')!=2025 or e.get('kind') not in {'source_model','normal_negative_model','fewshot_event_model'}:raise ValueError('Actual seed2025 learned identities required')
        if id in {ZS,ZT} and (e.get('kind')!='source_model' or e['identity'].get('checkpoint_sha256')!=query.normal.SOURCE):raise ValueError('Frozen source Z cannot be U/N')
        for choice in CHOICES:
            if id==F[choice] and (e.get('kind')!='fewshot_event_model' or e.get('choice')!=choice or e['identity'].get('checkpoint_sha256')!=F_CHECKPOINTS[choice]):raise ValueError('F support choice differs')
            if id==N[choice] and (e.get('kind')!='normal_negative_model' or e.get('choice')!=choice or e['identity'].get('checkpoint_sha256') in {query.normal.SOURCE,*F_CHECKPOINTS.values()}):raise ValueError('N support choice differs')
        want_view='model_target' if id==ZT else 'baseline_target' if b and origin=='target' else 'direct'
        if e.get('view')!=want_view:raise ValueError('TCAL requires real canonical overlay, not borrowed derived source scores')
    labels=cfg['labels']
    if labels.get('sha256')!=LABEL_SHA or labels.get('total_rows')!=23694 or labels.get('numeric_bounds')!=[17129,23694] or labels.get('shape')!=[6565] or labels.get('format')!='txt_suffix':raise ValueError('Native complete QUERY label axis required')
    return {e['id']:e for e in entries}

def bank_gates(cfg):
    lookup=contract(cfg);protect_label_paths(cfg);seal=modelview.metadata(cfg['raw_seal'])
    if seal!=dict(status='ACCEPTED_SMD14_FIXED_BANKS',approved_by='root',profile=PROFILE,banks=cfg['banks']):raise PermissionError('All fourteen root-sealed before labels')
    labelpath=Path(cfg['labels']['path']).resolve();reports=[]
    for e in cfg['banks']:
        if Path(e['directory']).resolve()==labelpath or Path(e['calibration']['path']).resolve()==labelpath:raise PermissionError('No label alias as bank/CAL')
        guard_manifest_files(e,cfg)
        if e.get('overlay'):
            if e['view']=='model_target':view=e['overlay']
            else:
                receipt=modelview.metadata(e['overlay']['receipt']);view=receipt['view_config']
                if receipt.get('method')!=e['method']:raise ValueError('Baseline overlay method differs')
            expected_parent=lookup[ZS] if e['view']=='model_target' else lookup[BASE[e['method']]['source']]
            if (view['destination']['directory']!=e['directory'] or view['destination']['identity']!=e['identity']
                or any(view['producer'][k]!=expected_parent[k] for k in ('directory','manifest_sha256','identity','ranges'))):raise ValueError('Declared view is not its own sealed source producer')
            guard_manifest_files(view['producer'],cfg)
        m=bank.bank_metadata(e);cal=modelview.metadata(e['calibration'])
        if e['calibration']['sha256']!=e['identity']['calibration_sha256'] or m['contract']['threshold']!=cal.get('threshold') or not np.isfinite(cal.get('threshold',np.nan)):raise ValueError('Bank derived CAL/tau mismatch')
        if e['kind'] in {'fewshot_event_model','normal_negative_model'}:
            r=modelview.metadata(e['calibration_receipt']);phase='target_event_calibrate' if e['kind']=='fewshot_event_model' else 'target_normal_calibrate'
            cc=modelview.metadata(e['calibration_config'])
            expected_operation='HYBRID_V1_STAGE2_EVENT_F' if e['kind']=='fewshot_event_model' else query.normal.OPERATION
            if (r.get('phase')!=phase or r.get('choice')!=e['choice'] or r.get('updates')!=0 or r.get('loaded_roles')!=['NORMAL_CAL'] or r.get('query_or_labels') is not False
                or cc.get('operation')!=expected_operation
                or cc.get('adapted_checkpoint_sha256')!=e['identity']['checkpoint_sha256'] or cc.get('choice')!=e['choice']
                or any(cc.get(k+'_sha256')!=v or r.get(k+'_sha256')!=v for k,v in query.normal.event.ACCEPTED.items())
                or cal.get('score_kind')!='event_logit' or cal.get('q')!=.99 or cal.get('method')!='higher' or cal.get('strict_greater') is not True or r.get('threshold')!=cal['threshold']):raise ValueError('Own F/N normal raw-logit CAL required')
        elif e['id']==ZS:query.fixed_R3(cal,True)
        elif e['id']==ZT:query.fixed_R3(cal,False)
        else:baseline.calibration(cal,e['calibration_origin']=='source')
        if e.get('acceptance') is None:raise PermissionError('Each actual raw/view acceptance required')
        a=modelview.metadata(e['acceptance'])
        if not acceptance_matches(e,a):raise PermissionError('Individual/group acceptance names different bank')
        reports.append(dict(id=e['id'],records=m['committed_records'],threshold=cal['threshold'],manifest_sha256=e['manifest_sha256']))
    if cfg.get('affiliation_identity'):
        if sha(cfg['affiliation_identity'])!=cfg.get('affiliation_identity_sha256'):raise PermissionError('Official Aff identity changed')
        from ..engine.official_affiliation import load_official_affiliation
        load_official_affiliation(cfg['affiliation_identity'])
    return reports

def scalar_inputs(cfg):
    result={}
    for e in cfg['banks']:
        iterator=(modelview.iter_materialized(e['overlay']) if e['view']=='model_target' else baseline.materialized(e['overlay']) if e['view']=='baseline_target' else read_records(e['directory']))
        rows=[]
        for number,row in enumerate(iterator):
            if (number>=6053 or tuple(row.get(k) for k in ('domain','entity','run'))!=tuple(cfg['stream']) or type(row.get('index')) is not int or row['index']!=17641+number
                or row.get('available_index')!=row['index'] or type(row.get('valid')) is not bool):raise ValueError('Original full axis/bool/clock differ')
            if row['valid'] and not np.isfinite(row.get('score',np.nan)):raise ValueError('Valid score must be finite')
            if not row['valid'] and (not isinstance(row.get('reason'),str) or not row['reason']):raise ValueError('Invalid is never normal')
            if e['kind']=='deterministic_baseline' and row.get('method')!=e['method']:raise ValueError('Baseline method raw identity differs')
            rows.append((row['index'],row.get('score'),row['valid'],row['available_index'],row.get('reason')))
        if len(rows)!=6053:raise ValueError('Incomplete scalar coverage before labels')
        result[e['id']]=dict(index=np.asarray([r[0] for r in rows],np.int64),score=np.asarray([np.nan if r[1] is None else r[1] for r in rows],np.float64),valid=np.asarray([r[2] for r in rows],bool),available_index=np.asarray([r[3] for r in rows],np.int64),invalid_reasons=[r[4] for r in rows])
    return result

def load_suffix_labels(spec):
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Original label file identity changed')
    values=[];count=0
    with Path(spec['path']).open('r',encoding='utf-8') as stream:
        for i,line in enumerate(stream):
            count=i+1
            if spec['numeric_bounds'][0]<=i<spec['numeric_bounds'][1]:
                value=float(line.strip())
                if not np.isfinite(value) or value not in (0.,1.):raise ValueError('Binary QUERY labels required')
                values.append(int(value))
    if count!=23694 or len(values)!=6565:raise ValueError('Original/QUERY label count differs')
    return np.asarray(values,np.int64)

def run(cfg,label_loader=load_suffix_labels):
    if Path(cfg['output']).exists():raise FileExistsError('Fresh fixed evaluation namespace')
    gates=bank_gates(cfg);arrays=scalar_inputs(cfg);labels=label_loader(cfg['labels'])  # first label body access
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);results={}
    for e,g in zip(cfg['banks'],gates):
        a=arrays[e['id']];r=evaluate_arrays(a['index'],a['score'],labels,g['threshold'],bank_complete=True,original_length=6565,original_start=17129,warmup_points=512,
            valid=a['valid'],available_index=a['available_index'],invalid_reasons=a['invalid_reasons'],affiliation_identity=cfg.get('affiliation_identity'))
        save_evaluation(out/e['id'],r,full_labels=labels,provenance=dict(descriptor=e,raw_seal=cfg['raw_seal'],source_sha256=cfg['source_sha256'],label_spec=cfg['labels']),**a);results[e['id']]=r
    pairs=[]
    for pair in cfg['comparisons']:
        left,right=pair['reference'],pair['candidate'];folder=out/'pairs'/(left+'__vs__'+right);folder.mkdir(parents=True)
        point,common=named.point_pair(arrays[left],arrays[right],labels,17129);events=named.paired_events(results[left],results[right])
        with (folder/'paired_events.csv').open('x',newline='',encoding='utf8') as f:
            keys=list(dict.fromkeys(k for row in events for k in row));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(events)
        np.savez_compressed(folder/'common_point_inputs.npz',index=common,reference_score=arrays[left]['score'][np.searchsorted(arrays[left]['index'],common)],candidate_score=arrays[right]['score'][np.searchsorted(arrays[right]['index'],common)],full_labels=labels)
        summary=dict(pair,common_point_effect=point,statistics=named.STATISTICS_NA);write(folder/'summary.json',summary);pairs.append(summary)
    summary=dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',profile=PROFILE,methods={id:dict(ranking=r['ranking'],coverage=r['coverage'],streams=r['streams']) for id,r in results.items()},comparisons=pairs,statistics=named.STATISTICS_NA,model_seed_mean=None,no_score_ensemble=True)
    write(out/'SUMMARY.json',summary);write(out/'CONFIG.json',cfg);write(out/'BANK_GATES.json',gates);return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','release','trusted-release-sha256'):p.add_argument('--'+key,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,bank_or_labels_opened=False)));return
    cfg=read(a.config);r=modelview.metadata(dict(path=a.release,sha256=a.trusted_release_sha256))
    if (r.get('operation')!=OPERATION or r.get('status')!='APPROVED' or r.get('approved_by')!='root' or r.get('allowed_actions')!=['evaluate_smd14'] or r.get('allow_labels') is not True
        or r.get('config_sha256')!=sha(a.config) or r.get('raw_seal_sha256')!=cfg['raw_seal']['sha256'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
        or cfg.get('permissions')!={'labels':True,'evaluation':True,'training':False,'target_fit':False,'GPU':False}):raise PermissionError('New exact SMD14 CPU-only label evaluation release')
    query.code_gate(cfg,r,RUNTIME_FILES)
    pins={'hybrid_tsad/hybrid_v1/evaluation.py':named.EVALUATOR_SHA,'hybrid_tsad/hybrid_v1/statistics.py':named.STATISTICS_SHA,
          'hybrid_tsad/hybrid_v1/chunked_bank.py':named.READER_SHA,'hybrid_tsad/hybrid_v1/named_evaluate_cli.py':'d0fe221e2ce6ba9f6f18f3a143ee6aaca05179b92d018cc2e0d4964afef85957'}
    if any(cfg['source_sha256'].get(k)!=v for k,v in pins.items()):raise PermissionError('Accepted numerical core/named helpers unchanged')
    if a.preflight_only:bank_gates(cfg);scalar_inputs(cfg);return
    run(cfg)

if __name__=='__main__':main()
