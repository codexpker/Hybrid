"""Actual role identity, FIT-only scaler and official SMD source preparation."""
import hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
P=Path(__file__).resolve().parent;R=P.parents[1];D=R/'data/prospective_smd_machine15_roles_20261106_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    with Path(p).open('x')as f:json.dump(d,f,indent=2,allow_nan=False)
def header(p):
    out={}
    with zipfile.ZipFile(p)as z:
        for name in z.namelist():
            with z.open(name)as f:
                version=np.lib.format.read_magic(f)
                assert version in [(1,0),(2,0)]
                shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f)if version==(1,0)else np.lib.format.read_array_header_2_0(f))
                out[name]=dict(shape=list(shape),dtype=str(dtype),fortran=fortran)
    return out
fixed=json.loads((P/'FIXED_COMPARISON.json').read_text());assert fixed['entity']=='SMD machine-1-5'
manifest=json.loads((D/'LEARNER_MANIFEST.json').read_text());roles=[]
for e in manifest['runs']:
    f=Path(e['path']);assert sha(f)==e['sha256'];assert set(header(f))=={'values.npy','observed.npy'}
    roles.append(dict(role=e['role'],local_path=str(f),sha256=e['sha256'],shape=e['expected_shape'],original_start=e['original_start'],headers=header(f),remote_path='__RESEARCH_ROOT__/preparations/'+P.name+'/data/'+e['role']+'.npz'))
with np.load(D/'FIT.npz')as z:fit=z['values'].astype(np.float64);observed=z['observed'];assert observed.all()and np.isfinite(fit).all()
scaler=StandardScaler().fit(np.asfortranarray(fit));dump(P/'FIT_ONLY_SCALER.json',dict(mean=scaler.mean_.tolist(),var=scaler.var_.tolist(),scale=scaler.scale_.tolist(),n_samples_seen=int(scaler.n_samples_seen_),fit_only_role='FIT',fit_original_interval=[0,14223],FIT_sha256=next(e['sha256']for e in roles if e['role']=='FIT'),SELECT_CAL_QUERY_not_used=True,input_values_float32_promoted_float64=True,constant_channels=np.flatnonzero(scaler.var_==0).tolist()))
cfgroot=R/'literature/novelty_review/left_source_20261031/configs/SMD';target=P/'configs/SMD';target.mkdir(parents=True,exist_ok=True);configs=[]
for name in ['model_configs_Left.json','train_configs.json','model_configs_0.json']:
    src=cfgroot/name;dst=target/name;assert not dst.exists();dst.write_bytes(src.read_bytes());configs.append(dict(path=str(src),sha256=sha(src),staged_path=str(dst)))
ad=json.loads((D/'audit_only/ADMISSION_AND_EVENTS.json').read_text())
dump(P/'INPUT_QUALIFICATION.json',dict(entity='machine-1-5',roles=roles,original_train_shape=ad['train_shape'],original_test_shape=ad['test_shape'],geometry=ad['geometry'],original_events=ad['original_event_count'],complete_QUERY_events=ad['main_uncensored_query_events'],events_all_short=True,train_health='UNKNOWN, provider nominal training background not certified clean',label_source='official OmniAnomaly samecommit test_label original, audit_only numerical labels not opened',historical_exposure=ad['historical_exposure'],upstream_pretraining=ad['upstream_pretraining'],existing_matched_LEFT_or_ARX='none located for this entity plus exact role/scaler protocol; original aggregate models cannot be relabeled matched',official_LEFT_configs=configs,source_commit='3fadb4811797075e233076b9faceb0d8ec59b697',new_QUERY_values_or_labels_read=False,admission_sha256=sha(D/'audit_only/ADMISSION_AND_EVENTS.json'),fresh_resource_file='REMOTE_CONFIG_RESOURCE_CHECK.json',old_adapter_differences='replace automatic80/20+allprefix scaler with explicit role loader/FIT-only frozen scaler; use SMD38 official configs/type1; preserve native loss/basic/train/SELECT/score'))
print(json.dumps(dict(status='ROLES_SCALER_OFFICIAL_CONFIGS_READY',roles=len(roles),FIT_shape=list(fit.shape),FIT_only_scaler=True,constant_channels=np.flatnonzero(scaler.var_==0).tolist(),QUERY_values_labels_not_opened=True)))
