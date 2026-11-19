import numpy as np
import torch
from hybrid_tsad.conditional_energy_readout import shard_axis
from hybrid_tsad.conditional_response_energy import energy_score, fixed_noise


def test_fixed_shards_cover_exact_axes_without_overlap():
    for task in [dict(score_axes=dict(CAL=[586056,607320],QUERY=[608535,944856])),
                 dict(score_axes=dict(CAL=[18624,21888],QUERY=[23016,87744]))]:
        for role,(lo,hi)in task['score_axes'].items():
            parts=[shard_axis(task,role,k)for k in range(4)]
            assert np.array_equal(np.concatenate(parts),np.arange(lo,hi))
            assert len(set(np.concatenate(parts)))==hi-lo


def test_absolute_time_noise_independent_of_shard_and_execution_order():
    task=dict(score_axes=dict(QUERY=[512,533]))
    def score(t):
        s,v=fixed_noise(1,3,torch.Generator().manual_seed(192025+int(t)),device='cpu')
        samples=v[:,:,:,0]+s[:,:,0,None]
        return energy_score(samples,torch.zeros(1,3))[0]
    full=torch.stack([score(t)for t in range(512,533)])
    parts={k:torch.stack([score(t)for t in shard_axis(task,'QUERY',k)])for k in [3,0,2,1]}
    assert torch.equal(full,torch.cat([parts[k]for k in range(4)]))
