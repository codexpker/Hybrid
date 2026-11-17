import numpy as np
import torch
from hybrid_tsad.timercd_reference_source import legal_pair,reference_statistics,make_batch

def test_reference_rejects_labels_and_actual_deviation():
    x=np.arange(2048,dtype=float).reshape(1024,2)
    sample=dict(time_series=x.copy(),normal_time_series=x.copy(),labels=np.zeros(1024))
    sample['time_series'][600]+=100;sample['labels'][600]=1
    assert legal_pair(sample,512,512)==(True,'LEGAL')
    sample['labels'][10]=1
    assert legal_pair(sample,512,512)[1]=='reference_label'
    sample['labels'][10]=0;sample['time_series'][10]+=1
    assert legal_pair(sample,512,512)[1]=='reference_actual_deviation'

def test_r_statistics_do_not_use_x_or_healthy_suffix():
    r=np.arange(1024,dtype=np.float32).reshape(512,2);x=r.copy()
    samples=[dict(reference=r.copy(),time_series=x.copy(),labels=np.zeros(512))]
    mu,std=reference_statistics(samples[0]['reference'])
    assert torch.equal(mu,torch.as_tensor(r).mean(0,keepdim=True))
    assert torch.equal(std,torch.as_tensor(r).std(0,keepdim=True)+1e-4)
    samples[0]['time_series']*=1000
    after=reference_statistics(samples[0]['reference'])
    assert torch.equal(mu,after[0])and torch.equal(std,after[1])
    b=make_batch(samples,[0],'cpu',2025,16,'reference')
    assert b['time_series'].shape==(1,512,2)
    assert torch.equal(b['time_series'][0],(torch.as_tensor(samples[0]['time_series'])-mu)/std)

def test_two_statistics_modes_share_masks_labels_and_noise():
    rs=np.random.RandomState(5)
    samples=[dict(reference=rs.randn(512,3),time_series=rs.randn(512,3)*5+10,labels=rs.randint(0,2,512))for _ in range(3)]
    a=make_batch(samples,[0,1,2],'cpu',84551,16,'batch')
    b=make_batch(samples,[0,1,2],'cpu',84551,16,'reference')
    for k in ['labels','mask','attention_mask']:assert torch.equal(a[k],b[k])
    assert torch.equal(a['masked_time_series'][a['mask']],b['masked_time_series'][b['mask']])
    assert not torch.equal(a['time_series'],b['time_series'])
