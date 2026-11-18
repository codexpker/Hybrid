import numpy as np

from hybrid_tsad.candi_native_a import score_starts,chunked_mahalanobis


def test_native_window_score_is_assigned_to_real_endpoint_without_padding():
    for axis,role in [((608535,944856),(608023,944919)),((585624,607320),(585544,607479)),
                      ((22656,87744),(22504,87841)),((18240,21888),(18112,21960))]:
        starts=score_starts(axis)
        assert starts[0]>=role[0]and starts[-1]+10<=role[1]
        assert np.array_equal(starts+9,np.arange(*axis))


def test_full_reference_chunk_keeps_distance_and_acceptance_mask():
    import torch
    generator=torch.Generator().manual_seed(2025)
    q=torch.randn(4,5,generator=generator);r=torch.randn(37,5,generator=generator)
    cov=torch.randn(5,5,generator=generator);inv=cov@cov.T
    difference=q[:,None,:]-r[None,:,:]
    expected=((difference@inv)*difference).sum(-1).min(1).values[:,None]
    actual=chunked_mahalanobis(q,r,inv,chunk=7)
    assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-6)
    assert torch.equal(actual<3.5,expected<3.5)
