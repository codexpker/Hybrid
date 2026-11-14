from pathlib import Path
import json,hashlib,shutil
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
domains={}
for domain in ['PSM','SWAT']:
    if domain=='PSM':
        q=R/'preparations/hybrid_native_predictive_cycle_20261113_v1';gate=read(q/'DATA_GATE.json')
        roles={e['role']:dict(path=str((q/'data'/('PSM_'+e['role']+'.npz')).relative_to(R)).replace('\\','/'),sha256=e['sha256'],start=int(e['original_start'][0]),length=int(e['shape'][1]))for e in gate['roles']if e['domain']=='PSM'and e['role']in['FIT','SELECT','CAL','QUERY']}
        config=q/'configs/PSM.json';scaler=q/'PSM_FIT_ONLY_SCALER.json';pt=R/'results/hybrid_native_predictive_cycle_20261113_v1/PSM_native/train/selected.pth';step=read(pt.parent/'TRAIN_RECEIPT.json')['selected']['python_step']
    else:
        q=R/'preparations/left_multievent_mixed_u_20261111_v1';gate=read(q/'INPUT_QUALIFICATION.json')
        roles={e['role']:dict(path=str((q/'data'/(e['role']+'.npz')).relative_to(R)).replace('\\','/'),sha256=e['sha256'],start=int(e['original_start']),length=int(e['shape'][0]))for e in gate['roles']if e['role']in['FIT','SELECT','CAL','QUERY']}
        config=q/'configs/SWAT/model_configs_Left.json';scaler=q/'FIT_ONLY_SCALER.json';pt=R/'results/left_multievent_mixed_u_20261111_v1/left_train/model/SWAT/Left/checkpoints_0/checkpoint.pth';step=read(R/'results/left_multievent_mixed_u_20261111_v1/left_train/RUNTIME_STATE.json')['final_python_step']
    c=read(config);C=c['enc_in'];assert c['seq_len']==192
    starts=np.random.default_rng(2025).integers(0,roles['FIT']['length']-191,size=(2000,128),dtype=np.int64)
    rng=np.random.default_rng(84551);channels=rng.integers(0,C,size=(2000,128),dtype=np.int64)
    select_n=len(np.arange(0,roles['SELECT']['length']-191,192));select_channels=np.random.default_rng(123045).integers(0,C,size=select_n,dtype=np.int64)
    plan=P/(domain+'_PLAN.npz');np.savez_compressed(plan,starts=starts,channels=channels,select_channels=select_channels)
    domains[domain]=dict(roles=roles,config=str(config.relative_to(R)).replace('\\','/'),config_sha256=sha(config),scaler=str(scaler.relative_to(R)).replace('\\','/'),scaler_sha256=sha(scaler),source_PT=str(pt.relative_to(R)).replace('\\','/'),source_sha256=sha(pt),source_python_step=step,plan_sha256=sha(plan),channels=C)
fixed=dict(id=P.name,seed=2025,updates=2000,batch128=True,optimizer='fresh Adam1e-4 fixed',loss='.5 complete native +.5 masked-channel allH nativecoeffMS/cycle/cons',SELECT='every200 min(.5native+.5masked) fixed channelplan, no query',infer='allC blind masks, original native formula j endpoint then Cmean; only current/history <=t',train_role='FIT only; no U_FIT/source retrain',domains=domains,source_untouched=True)
(P/'FIXED_EXPERIMENT.json').write_text(json.dumps(fixed,indent=2))
runtime=P/'runtime/hybrid_tsad';runtime.mkdir(parents=True,exist_ok=True);(runtime/'__init__.py').write_text('')
for name in ['left_predictive_cycle.py','left_channel_blindspot.py']:shutil.copyfile(R/'hybrid_ad/src/hybrid_tsad'/name,runtime/name)
print(json.dumps(dict(status='FROZEN',domains={k:dict(channels=v['channels'],source_step=v['source_python_step'],FIT=v['roles']['FIT']['length'],SELECT=v['roles']['SELECT']['length'])for k,v in domains.items()})))
