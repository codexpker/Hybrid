import numpy as np
import pytest
from types import SimpleNamespace
from hybrid_tsad.timesfm3_frozen import history_context,prediction_arrays,residual_score,run_predictions,PREDICT_OPTIONS


def test_strictly_past_joint_channels_and_target_mutation():
    values=np.arange(700*3,dtype=np.float32).reshape(700,3)
    x=history_context(values,1000,1512)
    assert x.shape==(3,512);np.testing.assert_array_equal(x.T,values[:512])
    values[512:]=9999;np.testing.assert_array_equal(history_context(values,1000,1512),x)
    with pytest.raises(PermissionError):history_context(values,1000,1511)
    with pytest.raises(PermissionError):history_context(values,1000,1700)


def test_point_residual_retains_each_point_and_channel():
    pred=np.array([[1.,2.],[3.,8.]],np.float32);actual=np.array([[0.,2.],[1.,4.]],np.float32)
    err,scores=residual_score(pred,actual)
    np.testing.assert_array_equal(err,[[1.,0.],[4.,16.]]);np.testing.assert_array_equal(scores,[.5,10.])
    assert scores.dtype==np.float64
    with pytest.raises(ValueError):residual_score(pred.ravel(),actual.ravel())


def test_official_median_quantile_axis_and_no_variate_split():
    class FakeOfficial:
        def predict_batch(self,contexts,**options):
            assert options==PREDICT_OPTIONS
            assert all(v.shape==(3,512)for v in contexts)
            for x in contexts:
                q=np.repeat(x[:,-1:,None],9,axis=-1)+np.arange(9,dtype=np.float32)[None,None,:]
                yield SimpleNamespace(forecast=q[...,4],quantiles=q)
    x=np.arange(3*512,dtype=np.float32).reshape(3,512)
    med,q=run_predictions(FakeOfficial(),[x,x+1])
    assert med.shape==(2,3)and q.shape==(2,3,9);np.testing.assert_array_equal(med,q[:,:,4])
    with pytest.raises(AssertionError):prediction_arrays([SimpleNamespace(forecast=np.zeros((3,1)),quantiles=np.ones((3,1,9)))],3)
