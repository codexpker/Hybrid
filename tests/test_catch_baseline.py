"""Small synthetic fixtures; no CATCH model construction or forward."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hybrid_tsad import catch_baseline as cb


class CatchAdmissionTests(unittest.TestCase):
    def test_exact_native_layout(self):
        x=cb.native_layout()
        self.assertEqual((x['fit_points'],x['select_points']), (1638,410))
        self.assertEqual((x['fit_windows'],x['fit_batches'],x['fit_last_batch']), (1447,12,39))
        self.assertEqual((x['select_windows'],x['select_batches'],x['select_last_batch']), (219,2,91))
        self.assertEqual((x['mask_interval'],x['score_windows'],x['scored_points'],x['unscored_tail']), (1,2,384,17))
        with self.assertRaises(ValueError):cb.native_layout(1024)

    def test_separate_deterministic_synthetic_streams(self):
        a,b=cb.synthetic_sequences();c,d=cb.synthetic_sequences()
        self.assertEqual((a.shape,b.shape),((2048,2),(401,2)))
        self.assertEqual(a.dtype,np.float64)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
        self.assertFalse(np.array_equal(a[:401],b))

    def test_pending_release_refused_without_model_import(self):
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/'config.json';approval=Path(p)/'approval.json'
            path.write_text('{}');approval.write_text('{"status":"PENDING"}')
            with self.assertRaises(PermissionError):
                cb.require_release(path,{'approval_path':str(approval)})

    def test_native_loader_and_defaults_cpu_only(self):
        # Explicit fixed external source, never load model or external data.
        import pandas as pd
        root=os.environ.get('CATCH_SOURCE_ROOT')
        if root is None:raise RuntimeError('Set fixed CATCH_SOURCE_ROOT; do not skip this integration fixture')
        api=cb.external_api(Path(root))
        from ts_benchmark.baselines.utils import anomaly_detection_data_provider,train_val_split
        from sklearn.preprocessing import StandardScaler
        raw,score=cb.synthetic_sequences()
        fit,select=train_val_split(pd.DataFrame(raw),.8,None)
        scaler=StandardScaler().fit(fit.values)
        for role,values,count in [('train',fit,12),('val',select,2),('thre',pd.DataFrame(score),1)]:
            loader=anomaly_detection_data_provider(scaler.transform(values),128,192,1,role)
            self.assertEqual(len(loader),count)
            self.assertFalse(loader.drop_last)
            if role=='thre':
                self.assertEqual(len(loader.dataset),2)
                first,_=loader.dataset[0];last,_=loader.dataset[1]
                np.testing.assert_array_equal(first,scaler.transform(score)[:192].astype(np.float32))
                np.testing.assert_array_equal(last,scaler.transform(score)[192:384].astype(np.float32))
        cfg=api.TransformerConfig(**cb.PARAMS)
        for key,value in dict(lradj='type1',patience=3,pct_start=.3,auxi_loss='MAE',
                              auxi_type='complex',auxi_mode='fft',module_first=True,
                              mask=False,inference_patch_stride=1).items():
            self.assertEqual(getattr(cfg,key),value)


if __name__=='__main__':unittest.main()
