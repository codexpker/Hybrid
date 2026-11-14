import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];s=importlib.util.spec_from_file_location('u',R/'hybrid_ad/src/hybrid_tsad/left_native_u.py');u=importlib.util.module_from_spec(s);s.loader.exec_module(u)
fixed=json.loads((P/'FIXED_U_SENSITIVITY.json').read_text());assert fixed['updates']==200
old=R/'preparations/cross_domain_left_arx_20261111_v1';sc=json.loads((old/'FIT_ONLY_SCALER.json').read_text())
with np.load(old/'data/FIT.npz')as z:raw=z['values'].astype(np.float64);assert z['observed'].all()
v=(raw-np.asarray(sc['mean']))/np.asarray(sc['scale']);plan=u.paired_plan(len(v),np.flatnonzero(np.asarray(sc['var'])>0));np.savez_compressed(P/'PAIRED_PLAN.npz',**plan)
assert plan['replacement_mask'].sum()==2560 and np.all(np.abs(plan['donor_starts'][plan['replacement_mask']]-plan['base_starts'][plan['replacement_mask']])>=192)
touched=np.zeros(len(v),bool);changed=0;lo=float('inf');hi=float('-inf')
for update in range(200):
    A=u.native_batch(v,plan,update,'A');B=u.native_batch(v,plan,update,'B');C=u.native_batch(v,plan,update,'C');slots=plan['replacement_mask'][update]
    assert np.array_equal(A[~slots],B[~slots])and np.array_equal(B[~slots],C[~slots])
    diff=B-C;active=diff!=0;assert int(active.sum())==int(slots.sum())*96*4
    assert not active[:,:48].any()and not active[:,144:].any()
    other=[i for i in range(38)if i not in plan['corruption_channels']];assert not active[:,:,other].any()
    changed+=int(active.sum())
    if active.any():lo=min(lo,float(diff[active].min()));hi=max(hi,float(diff[active].max()))
    for donor in plan['donor_starts'][update,slots]:touched[int(donor)+48:int(donor)+144]=True
assert changed==983040
out=dict(status='PASS',model_input_shape=[128,192,38],all_footprints_FIT_only=[0,14223],QUERY_donors=False,plan_sha256=hashlib.sha256((P/'PAIRED_PLAN.npz').read_bytes()).hexdigest(),replaced_presentations=2560,total_presentations=25600,replacement_fraction=.1,altered_time_exposures=2560*96,total_time_exposures=25600*192,altered_time_fraction=.05,altered_scalar_cells=changed,total_scalar_cells=25600*192*38,altered_scalar_fraction=changed/(25600*192*38),numeric_B_minus_C_min=lo,numeric_B_minus_C_max=hi,corruption_channels=plan['corruption_channels'].tolist(),unique_base_starts=int(len(np.unique(plan['base_starts']))),unique_donor_starts=int(len(np.unique(plan['donor_starts'][plan['replacement_mask']]))),unique_original_time_positions_touched=int(touched.sum()),unique_original_time_channel_positions_touched=int(touched.sum())*4,not_raw_stream_contamination=True,not_independent_anomaly_events=True,clean_target_not_given=True,numeric_implementation='normalize float64, construct native float32 model batch, then +float32(3) in same B donor input; C same donor without augmentation',fixed_rule_sha256=hashlib.sha256((P/'FIXED_U_SENSITIVITY.json').read_bytes()).hexdigest(),new_QUERY_labels_read=False)
np.savez_compressed(P/'SYNTHETIC_FOOTPRINTS.npz',unique_original_time_positions=np.flatnonzero(touched),channels=plan['corruption_channels'])
(P/'CPU_U_PLAN_TESTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
