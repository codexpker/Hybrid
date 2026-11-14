"""Create-only CPU preparation; no model, training, score or threshold."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1.stage2_u_psm_data import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True)
    a=p.parse_args();body=Path(a.config).read_bytes()
    if hashlib.sha256(body).hexdigest()!=a.trusted_config_sha256:raise PermissionError('Frozen data recipe required')
    c=json.loads(body)
    for path,expected in c['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise PermissionError('Source changed')
    start=time.monotonic();r=prepare(c)
    print(json.dumps(dict(status=r['status'],seconds=time.monotonic()-start,
        conditions=len(r['artifacts']),query_points=r['query_selection']['scored_points'],GPUh=0)))
