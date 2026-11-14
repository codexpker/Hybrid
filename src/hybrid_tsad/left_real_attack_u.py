"""Numerical A/B/C constructor. Event labels stay in the offline plan builder."""
import numpy as np

def native_batch(background, donor_values, plan, update, condition):
    if condition not in ['A','B','C']:raise ValueError('fixed A/B/C only')
    starts=plan['base_starts'][update]
    x=np.stack([background[int(s):int(s)+192]for s in starts]).astype(np.float32)
    if condition!='A':
        name=condition+'_donor_starts'
        for slot in np.flatnonzero(plan['replacement_mask'][update]):
            a=int(plan[name][update,slot]);x[slot]=np.float32(donor_values[a:a+192])
    return x
