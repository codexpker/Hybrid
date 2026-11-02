"""CPU metadata guard regression: no model construction or CUDA queries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class L0Guard(unittest.TestCase):
    def test_01_missing_release_rejects_before_torch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"config.json"; p.write_text("{}", encoding="utf-8")
            code = ("import sys;from hybrid_tsad.hybrid_v1.cli import l0_approval\n"
                    "try:l0_approval(sys.argv[1],None,None,sys.argv[2])\n"
                    "except PermissionError:print('REJECTED_BEFORE_TORCH', 'torch' not in sys.modules)\n"
                    "else:raise RuntimeError('missing release accepted')\n")
            result = subprocess.run([sys.executable, "-c", code, str(p), str(Path(d)/"out")],
                capture_output=True, text=True, timeout=20, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("REJECTED_BEFORE_TORCH True", result.stdout)
            self.assertFalse((Path(d)/"out").exists())

    def test_02_pending_release_rejects_before_torch(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"config.json"; p.write_text("{}", encoding="utf-8")
            release = Path(d)/"PENDING.json"; release.write_text('{"status":"PENDING"}', encoding="utf-8")
            h = hashlib.sha256(release.read_bytes()).hexdigest()
            code = ("import sys;from hybrid_tsad.hybrid_v1.cli import l0_approval\n"
                    "try:l0_approval(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])\n"
                    "except PermissionError:print('PENDING_REJECTED', 'torch' not in sys.modules)\n"
                    "else:raise RuntimeError('pending accepted')\n")
            r = subprocess.run([sys.executable,"-c",code,str(p),str(release),h,str(Path(d)/"out")],
                capture_output=True,text=True,timeout=20)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("PENDING_REJECTED True", r.stdout)

    def test_03_preview_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"config.json";p.write_text("{}",encoding="utf-8")
            r=subprocess.run([sys.executable,"-m","hybrid_tsad.hybrid_v1.cli","--config",str(p),
                "--phase","synthetic_gpu_l0","--output-dir",str(Path(d)/"out")],
                capture_output=True,text=True,timeout=20)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertTrue(json.loads(r.stdout)["preview"])
            self.assertFalse((Path(d)/"out").exists())

    def test_04_preflight_cannot_run_another_phase(self):
        r=subprocess.run([sys.executable,"-m","hybrid_tsad.hybrid_v1.cli","--config","never_open.json",
            "--phase","pretrain","--output-dir","never_create","--execute","--preflight-only"],
            capture_output=True,text=True,timeout=20)
        self.assertNotEqual(r.returncode,0)
        self.assertIn("restricted",r.stderr)


if __name__ == "__main__":
    unittest.main()
