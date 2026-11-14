"""Native LEFT concatenate/reshape/copy/channel mean, explicit channel count.

The historical PSM adapter validates C=25; controlled C=4 needs a corresponding
shape guard. Reduction order and float32 arithmetic remain identical.
"""
import numpy as np

def native_numpy_scores(chunks,channels):
    if not chunks or any(a.dtype!=np.float32 or a.ndim!=3 or a.shape[1:]!=(192,channels)
                         or not np.isfinite(a).all() for a in chunks):
        raise ValueError('Invalid native LEFT per-channel score chunks')
    energy=np.concatenate(chunks,axis=0)
    flat=energy.reshape(-1,energy.shape[-1])
    flat=np.array(flat)
    return energy,np.mean(flat,axis=-1)
