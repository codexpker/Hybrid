"""Extracted unchanged from project-owned frozen research helpers."""
import numpy as np


def ap_cache(y,s,block):
    order=np.argsort(-s,kind='stable');sorted_s=s[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_s)!=0),len(y)-1]
    return y[order].astype('float64'),block[order],ends


def weighted_ap(cache,counts):
    y,b,ends=cache;w=counts[b].astype('float64')
    tp=np.cumsum(w*y)[ends];den=np.cumsum(w)[ends]
    precision=np.divide(tp,den,out=np.zeros_like(tp),where=den>0)
    if tp[-1]==0:return float('nan')
    return float((np.diff(np.r_[0.,tp])*precision).sum()/tp[-1])
