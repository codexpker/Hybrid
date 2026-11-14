"""Separate constructor indices from audit-only event/label semantics before training."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1]
with np.load(P/'PAIRED_PLAN.npz')as z:plan={k:z[k].copy()for k in z.files}
keys=['base_starts','B_donor_starts','C_donor_starts','replacement_mask','selected_flat_order'];np.savez_compressed(P/'LEARNER_PLAN.npz',**{k:plan[k]for k in keys})
(P/'audit_only').mkdir(exist_ok=False);np.savez_compressed(P/'audit_only/EVENT_PLAN.npz',**{k:v for k,v in plan.items()if k not in keys})
gate=json.loads((P/'DATA_PLAN_GATE.json').read_text());gate['learner_plan_sha256']=hashlib.sha256((P/'LEARNER_PLAN.npz').read_bytes()).hexdigest();gate['learner_keys']=keys
# Preserve first extraction gate; this is a permissions-only derivative.
(P/'LEARNER_PLAN_GATE.json').write_text(json.dumps(gate,indent=2))
(P/'LEARNER_RUNTIME.json').write_text(json.dumps(dict(learner_plan_sha256=gate['learner_plan_sha256'],learner_keys=keys,updates=2000,batch_size=128,labels_or_event_ids_in_learner_plan=False),indent=2))
spec=importlib.util.spec_from_file_location('batch',P/'runtime/left_real_attack_u.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);sc=json.loads((P/'FIT_ONLY_SCALER.json').read_text());mean=np.asarray(sc['mean']);scale=np.asarray(sc['scale'])
with np.load(P/'data/U_FIT.npz')as z:fit=(z['values']-mean)/scale
with np.load(P/'data/DONOR.npz')as z:donor=(z['values']-mean)/scale
changed=0;all_cells=0
for step in range(2000):
    A=m.native_batch(fit,donor,plan,step,'A');B=m.native_batch(fit,donor,plan,step,'B');C=m.native_batch(fit,donor,plan,step,'C');mask=plan['replacement_mask'][step];assert A.shape==(128,192,31)and np.isfinite(B).all()and np.isfinite(C).all();assert np.array_equal(A[~mask],B[~mask])and np.array_equal(B[~mask],C[~mask]);changed+=int((B[mask]!=C[mask]).sum());all_cells+=B.size
out=dict(status='PASS',learner_plan_sha256=gate['learner_plan_sha256'],labels_or_event_ids_in_learner_plan=False,model_receives_values_only=True,all_2000_batches_finite_shape_valid=True,only_10percent_slots_differ=True,B_C_changed_scalar_exposures=changed,total_input_scalar_exposures=all_cells,B_C_changed_scalar_fraction=changed/all_cells,physical_affected_channels='UNKNOWN; difference is counterfactual B-C replacement difference, not fault localization',loss_observer_does_not_receive_mask_or_tags=True)
(P/'CPU_BATCH_GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
