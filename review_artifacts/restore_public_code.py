"""Materialize reviewed text at its original relative layout, never execute it.

Set HYBRID_RESEARCH_ROOT or --root to a separate external research workspace.
Default is check-only; --write refuses to overwrite different existing files.
No data/weights/downloads/SSH/Git/GPU actions are performed by this utility.
"""
from pathlib import Path
import argparse,ast,hashlib,json,os

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=os.environ.get('HYBRID_RESEARCH_ROOT'));ap.add_argument('--write',action='store_true');args=ap.parse_args()
    if not args.root:ap.error('Supply --root or HYBRID_RESEARCH_ROOT; no default machine path')
    root=Path(args.root).expanduser().resolve();here=Path(__file__).resolve().parent
    if root==here or root.is_relative_to(here):ap.error('External workspace must be outside review_artifacts')
    report=json.loads((here/'PUBLIC_CODE_MANIFEST.json').read_text(encoding='utf-8'));jobs=[]
    for e in report['entries']:
        rel=Path(e['source_relative_path']);dest=(root/rel).resolve();source=here/e['export_relative_path']
        if rel.is_absolute()or not dest.is_relative_to(root):raise ValueError('Unsafe manifest target')
        raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['export_sha256']:raise ValueError('Reviewed export hash mismatch: '+str(rel))
        text=raw.decode('utf-8').replace('__RESEARCH_ROOT__',root.as_posix())
        if source.suffix=='.py':ast.parse(text,filename=str(rel))
        elif source.suffix=='.json':json.loads(text)
        content=text.encode('utf-8')
        if dest.exists()and dest.read_bytes()!=content:raise FileExistsError('Existing differing file retained: '+str(rel))
        jobs.append((dest,content))
    if args.write:
        for dest,content in jobs:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with dest.open('xb')as f:f.write(content)
    print(json.dumps(dict(mode='written'if args.write else'check_only',files=len(jobs),external_root=str(root),experiments_executed=False,data_or_weights_included=False)))

if __name__=='__main__':main()
