"""CPU test profiles, one fresh process per module; never weakens test assertions."""
import argparse
import fnmatch
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def child(args):
    path = ROOT / args.module
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT / 'tests'))
    sys.path.insert(0, str(ROOT / 'src'))
    config = json.loads(Path(args.profile_manifest).read_text())
    rules = config['integration']
    loader = unittest.TestLoader()
    suite = loader.discover(str(path.parent), pattern=path.name)
    chosen, excluded = [], []
    for case in flatten(suite):
        if args.case and case.id() != args.case:
            continue
        rule = next((r for r in rules if case.id() == r['id_prefix'] or case.id().startswith(r['id_prefix'] + '.')), None)
        if rule and args.profile == 'unit':
            excluded.append({'test': case.id(), 'status': 'INTEGRATION_NOT_RUN', 'requires': rule['requires']})
        elif not rule and args.profile == 'integration':
            pass
        else:
            chosen.append(case)
    if not args.case and chosen and path.stem in config.get('per_test_process_modules', []):
        reports=[]
        with tempfile.TemporaryDirectory() as temp:
            for i, case in enumerate(chosen):
                out=Path(temp)/f'{i}.json'
                cmd=[sys.executable, str(Path(__file__).resolve()), '--profile', args.profile,
                     '--module', args.module, '--case', case.id(), '--child-output', str(out), '--profile-manifest', args.profile_manifest]
                proc=subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                reports.append(json.loads(out.read_text()) if out.exists() else
                    dict(tests_run=0, failures=[], errors=[dict(test=case.id(),trace_tail=proc.stderr[-2400:])],skipped=[],status='PROCESS_ERROR'))
        report=dict(module=args.module, isolation='per-test', tests_run=sum(r['tests_run'] for r in reports),
            failures=[x for r in reports for x in r['failures']], errors=[x for r in reports for x in r['errors']],
            skipped=[x for r in reports for x in r['skipped']], excluded=excluded,
            status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL')
        Path(args.child_output).write_text(json.dumps(report, indent=2))
        return 0 if report['status']=='PASS' else 1
    result = unittest.TestResult()
    unittest.TestSuite(chosen).run(result)
    # Keep traces compact (some assertions stringify all sys.modules).
    def rows(items):
        return [{'test': t.id(), 'trace_tail': '\n'.join(trace.splitlines()[-6:])[-2400:]} for t, trace in items]
    report = dict(module=args.module, tests_run=result.testsRun, failures=rows(result.failures),
                  errors=rows(result.errors), skipped=[{'test': t.id(), 'reason': r} for t,r in result.skipped],
                  excluded=excluded, status=('INTEGRATION_NOT_RUN' if excluded else 'NO_MATCH') if not chosen else ('PASS' if result.wasSuccessful() else 'FAIL'))
    Path(args.child_output).write_text(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=('unit','integration','all'), default='unit')
    p.add_argument('--pattern', default='test_*.py')
    p.add_argument('--output', type=Path, default=Path('test-profile-results.json'))
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--module', help=argparse.SUPPRESS)
    p.add_argument('--child-output', help=argparse.SUPPRESS)
    p.add_argument('--case', help=argparse.SUPPRESS)
    p.add_argument('--profile-manifest', default=str(ROOT/'tests/profiles.json'), help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.module:
        return child(args)
    if args.output.exists():
        p.error('output exists; choose a new path (old results are not overwritten)')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    env.pop('HYBRID_AUDIT_CUDA_RNG', None)
    env['PYTHONPATH']=str(ROOT/'src')
    start=time.monotonic(); results=[]
    files=sorted(f for f in (ROOT/'tests').rglob('test_*.py') if fnmatch.fnmatch(f.name,args.pattern))
    if not files:
        p.error('pattern matched no test modules; no test success can be reported')
    with tempfile.TemporaryDirectory() as temp:
        frozen_manifest=Path(temp)/'profile.json'
        frozen_manifest.write_bytes(Path(args.profile_manifest).read_bytes())
        for i,file in enumerate(files):
            out=Path(temp)/f'{i}.json'; rel=file.relative_to(ROOT).as_posix()
            cmd=[sys.executable,str(Path(__file__).resolve()),'--profile',args.profile,'--module',rel,'--child-output',str(out),'--profile-manifest',str(frozen_manifest)]
            try:
                proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=args.timeout)
                row=json.loads(out.read_text()) if out.exists() else dict(module=rel,status='PROCESS_ERROR',stderr_tail=proc.stderr[-2400:])
                row['exit_code']=proc.returncode
            except subprocess.TimeoutExpired:
                row=dict(module=rel,status='TIMEOUT',timeout_seconds=args.timeout)
            results.append(row)
            print(f'{rel}: {row["status"]} ({row.get("tests_run",0)} run, {len(row.get("excluded",[]))} integration excluded)',flush=True)
    report=dict(profile=args.profile,process_isolation='per-module; configured mixed modules per-case',recursive=True,CPU_only=True,
                seconds=time.monotonic()-start,modules=results,
                totals={k:sum(r.get(k,0) if k=='tests_run' else len(r.get(k,[])) for r in results) for k in ('tests_run','failures','errors','skipped','excluded')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    return int(any(r['status'] not in ('PASS','INTEGRATION_NOT_RUN','NO_MATCH') for r in results))

if __name__=='__main__':
    raise SystemExit(main())
