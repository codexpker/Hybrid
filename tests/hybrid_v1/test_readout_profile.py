"""Synthetic/metadata glue fixtures; no official model/data/weights/GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import readout


class ProfileGlue(unittest.TestCase):
    def test_fixed_synthetic_independent_rng_and_clock(self):
        spec=dict(rng_seed=84551,length=640,channels=25,distribution="standard_normal_float32",observed="all_true")
        before=np.random.get_state();r=readout.synthetic_run(spec);after=np.random.get_state()
        self.assertEqual(r.values.shape,(640,25));self.assertEqual(r.values.dtype,np.float32)
        np.testing.assert_array_equal(r.values,np.random.default_rng(84551).standard_normal((640,25)).astype(np.float32))
        np.testing.assert_array_equal(before[1],after[1]);self.assertTrue(r.observed.all())
        self.assertEqual(list(range(512,len(r.values)))[-31:],list(range(609,640)))
        with self.assertRaises(ValueError):readout.synthetic_run(dict(spec,length=769))

    def test_pending_and_query_release_rejected_before_assets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cfg=p/"config.json";cfg.write_text('{}')
            with self.assertRaises(PermissionError):readout.admit(cfg,None,None,p/"out")
            release=p/"release.json";release.write_text(json.dumps(dict(status="APPROVED",approved_by="root",
                operation="HYBRID_V1_PSM_QUERY",allowed_actions=["synthetic_profile"],
                config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest())))
            with self.assertRaises(PermissionError):
                readout.admit(cfg,release,hashlib.sha256(release.read_bytes()).hexdigest(),p/"out")
            self.assertFalse((p/"out").exists())

    def test_reusable_writer_two_chunks_tail_and_existing_score(self):
        import torch
        from hybrid_tsad.hybrid_v1.chunked_bank import read_records,read_entries,IDENTITY_KEYS
        # Tiny object supplies device only; no model forward or weight file.
        model=torch.nn.Linear(1,1).cpu()
        run=readout.synthetic_run(dict(rng_seed=84551,length=640,channels=25,distribution="standard_normal_float32",observed="all_true"))
        raw=[dict(domain=run.domain,entity=run.entity,run=run.run,index=i,available_index=i,
                  r_tf=1.,r_ms=1.,r_pred=1.) for i in range(512,640)]
        cal=dict(scales=dict(r_tf=1.,r_ms=1.,r_pred=1.),weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),
                 active_components=["r_tf","r_ms","r_pred"],threshold=.5)
        with tempfile.TemporaryDirectory() as d,patch("hybrid_tsad.hybrid_v1.chunked_bank.iter_pointwise_infer",return_value=iter(raw)), \
                patch("torch.cuda.synchronize",side_effect=AssertionError("CPU must not query CUDA")):
            out=readout.write_model_bank(model,[run],cal,d,identity={k:"a"*64 for k in IDENTITY_KEYS},
                history=512,horizon=32,seed=2025,chunk_size=64)
            manifest=json.loads((Path(d)/"manifest.json").read_text())
            self.assertEqual([c["records"] for c in manifest["chunks"]],[64,64])
            self.assertTrue(manifest["complete"]);self.assertEqual(len(out["timings"]),128)
            rows=list(read_records(d));self.assertEqual([r["index"] for r in rows],list(range(512,640)))
            self.assertTrue(all(r["score"]==np.log(2) for r in rows))
            entries=list(read_entries(d));self.assertFalse(entries[1]["derived"]["confirmed3_current"])
            self.assertTrue(entries[2]["derived"]["confirmed3_current"])
            self.assertTrue(entries[64]["derived"]["confirmed3_current"])


if __name__=="__main__":unittest.main()
