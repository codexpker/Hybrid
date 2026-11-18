import json
from pathlib import Path
import numpy as np
import pytest
from hybrid_tsad.timercd_swat_fixed import SwatReader
from hybrid_tsad.timercd_fixed import sha,window_layout

def test_swat_reader_preserves_order_index_and_preaccess_role_guard(tmp_path):
    values=np.arange(93,dtype=np.float64).reshape(3,31)
    np.savez_compressed(tmp_path/'CAL.npz',values=values,observed=np.ones_like(values,dtype=bool))
    np.savez_compressed(tmp_path/'stats.npz',mean=np.arange(31,dtype=float),official_safe_std=np.ones(31))
    binding=dict(channels_count=31,channels=[' AIT201']+[str(i)for i in range(30)],roles=[dict(role='CAL',path='CAL.npz',sha256=sha(tmp_path/'CAL.npz'),shape=[3,31],original_start=445661,end=445664,score_start=446173)],fit_statistics=dict(path='stats.npz',sha256=sha(tmp_path/'stats.npz')))
    (tmp_path/'binding.json').write_text(json.dumps(binding))
    reader=SwatReader(tmp_path,dict(data_binding='binding.json',data_binding_sha256=sha(tmp_path/'binding.json')))
    x,index,_=reader.load('CAL')
    assert np.array_equal(x,values)and np.array_equal(index,np.arange(445661,445664))
    assert reader.binding['channels'][0]==' AIT201'
    with pytest.raises(PermissionError):reader.load('FIT')
    mean,std=reader.statistics()
    assert np.array_equal(mean,np.arange(31))and np.array_equal(std,np.ones(31))

def test_swat_native_grid_is_before_established_axis_crop():
    index=np.arange(608023,614023,dtype=np.int64)
    layout=window_layout(index,5000)
    assert layout['window_start_index'][512]==608023
    assert layout['window_start_index'][5000]==613023
    assert index[512]==608535
    chosen=np.linspace(446173,494999,512,dtype=np.int64)
    assert len(np.unique(chosen))==512 and chosen[0]==446173 and chosen[-1]==494999
