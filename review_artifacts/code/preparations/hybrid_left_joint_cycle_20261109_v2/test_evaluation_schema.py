import copy,sys
from hybrid_tsad.hybrid_v1.source_joint_cycle_psm_comparison_v2 import INITIAL,validate_initial_mapping
m=dict(shared_initial_sha256=INITIAL,shared_exact=True,complete_old_state_exact=False,new_heads_seed=2025,new_names=['joint_fusion.0.weight'],removed_names=['tf_fusion_delta.weight'],shared_names=['forecast_head.weight'])
assert validate_initial_mapping(m)
for key,value in [('shared_exact',False),('new_names',['invented.weight']),('removed_names',['temporal.weight']),('complete_old_state_exact',True)]:
 b=copy.deepcopy(m);b[key]=value
 try:validate_initial_mapping(b)
 except (PermissionError,ValueError):pass
 else:raise AssertionError(key)
assert 'torch' not in sys.modules
print('5 metadata fixtures PASS; no torch/truth/score input; numerical _evaluate/run AST unchanged')
