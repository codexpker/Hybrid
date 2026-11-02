"""Synthetic CPU only: no project data arrays or Torch imported."""
import copy
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import event_data as e
from hybrid_tsad.hybrid_v1.data import Run, window, resolve_plan


class EventInputs(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(16585*2, dtype=np.float32).reshape(16585, 2)
        self.labels = np.zeros(17129, np.int8)
        for s, t in e.EVENTS:
            self.labels[s:t] = 1
        self.catalogue = [dict(event_id='SMD:machine-2-1:E%d'%i, alias='E%d'%i, event_index=138+i, start=s, end=t)
                          for i, (s, t) in enumerate(e.EVENTS)]
        self.normal = [dict(matched_event_id=self.catalogue[i]['event_id'], start=s, end=t) for i, (s, t) in enumerate(e.NORMAL)]
        self.kw = dict(identity=dict(domain='SMD', entity='machine-2-1', run='machine-2-1_test'), roles=copy.deepcopy(e.ROLES),
                       catalogue=self.catalogue, normal=self.normal, history=512, horizon=32,
                       counts=dict(NORMAL_SELECT=32, NORMAL_CAL=512), seed=2025)

    def build(self, **kw):
        return e.construct(self.values, self.labels, **dict(self.kw, **kw))

    def test_cold_seven_file_package_metadata_preview_without_worktree_or_torch(self):
        package_source=Path(e.__file__).resolve().parents[2]
        candidate=package_source.parent/'configs/hybrid_v1/f_event_data.SMD2_1.v3.PENDING.json'
        cfg=json.loads(candidate.read_text())
        required={'hybrid_tsad/__init__.py','hybrid_tsad/hybrid_v1/__init__.py',
                  'hybrid_tsad/hybrid_v1/event_data.py','hybrid_tsad/hybrid_v1/target_data.py',
                  'hybrid_tsad/hybrid_v1/data.py','hybrid_tsad/hybrid_v1/config.py',
                  'hybrid_tsad/hybrid_v1/provenance.py'}
        self.assertEqual(set(cfg['source_sha256']),required)
        with tempfile.TemporaryDirectory() as d:
            cwd=Path(d); cold=cwd/'freshsource'; cold.mkdir()
            for name,digest in cfg['source_sha256'].items():
                self.assertEqual(e.sha(package_source/name),digest)
                destination=cold/name; destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(package_source/name,destination)
            metadata=cwd/'exact_config.json'; metadata.write_bytes(candidate.read_bytes())
            code="""import sys,json,os
from pathlib import Path
cold=Path(sys.argv[1]).resolve()
# -I removes cwd/PYTHONPATH; explicitly retain only the fresh source plus
# interpreter/library paths, never the working repository.
sys.path.insert(0,str(cold))
assert not any('research_workspace' in p.lower() for p in sys.path),sys.path
from hybrid_tsad.hybrid_v1 import event_data
from hybrid_tsad.hybrid_v1 import data,config,provenance
def denied(*args,**kwargs): raise AssertionError('NO INPUT PARSE IN PREVIEW')
event_data.derive=denied; event_data.prefix_rows=denied; event_data.sha=denied
event_data.main(['--config',sys.argv[2]])
files={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
       if name=='hybrid_tsad' or name.startswith('hybrid_tsad.')}
assert len(files)==7,files
assert all(cold in Path(p).parents for p in files.values()),files
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
print(json.dumps(dict(cold_import='PASS',cwd=os.getcwd(),modules=files,
                     torch_loaded=False,preview_only=True,query_or_real_inputs_opened=False)))
"""
            result=subprocess.run([sys.executable,'-I','-c',code,str(cold),str(metadata)],
                                  cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            lines=result.stdout.strip().splitlines()
            self.assertEqual(json.loads(lines[0]),dict(preview=True,numeric_read=False))
            evidence=json.loads(lines[-1]); self.assertEqual(evidence['cold_import'],'PASS')
            self.assertFalse(evidence['torch_loaded']); self.assertEqual(len(evidence['modules']),7)
            print('COLDTREE_SUBPROCESS_EVIDENCE '+json.dumps(dict(command=result.args,exit_code=result.returncode,
                  stdout=result.stdout,stderr=result.stderr)))

    def test_exact_half_open_support_and_canonical_ids(self):
        r = self.build(); singles = [x for x in r['combinations'] if x['K'] == 1]
        self.assertEqual(len(singles), 3)
        for i, x in enumerate(singles):
            ev, nm = x['events'][0], x['normal'][0]
            self.assertEqual(ev['event_id'], 'SMD:machine-2-1:E%d'%i)
            self.assertEqual(ev['event_index'], 138+i)
            self.assertEqual(nm['end']+31, ev['start'])
            run = Run(**self.kw['identity'], role='SUPPORT', values=r['values'][:9477], observed=r['observed'][:9477])
            for t in (ev['start'], ev['end']-1):
                self.assertEqual(window(run, t, 512, 32)['footprint'], [t-512, t+32])
        changed = copy.deepcopy(self.catalogue); changed[0]['start'] += 1
        with self.assertRaises(ValueError): self.build(catalogue=changed)

    def test_exact_deterministic_32_512_and_all_normal(self):
        before = np.random.get_state(); r = self.build(); after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(r['candidate_counts'], dict(NORMAL_SELECT=1916, NORMAL_CAL=2018, NORMAL_CAL_arrived_not_selected=2049))
        rng = np.random.default_rng(2025)
        for role, candidates, count in [('NORMAL_SELECT', np.r_[np.arange(10533,12142), np.arange(12693,13000)], 32),
                                        ('NORMAL_CAL', np.arange(14087,16105), 512)]:
            points = r['plans'][role]
            self.assertEqual([p['index'] for p in points], rng.choice(candidates, size=count, replace=False).tolist())
            self.assertEqual(len(set(p['index'] for p in points)), count)
            lo, hi = self.kw['roles'][role]
            run = Run(**self.kw['identity'], role=role, values=r['values'][lo:hi], observed=r['observed'][lo:hi], start=lo)
            resolve_plan([run], points, 512, 32)
            for p in points:
                a, b = p['footprint']; self.assertFalse(self.labels[a:b].any())
        self.assertEqual(r['plans'], self.build()['plans'])

    def test_no_query_numeric_read(self):
        class Guarded:
            def __init__(self): self.n = 0
            def __iter__(self): return self
            def __next__(self):
                self.n += 1
                if self.n > 3: raise AssertionError('QUERY BODY READ')
                return '1,2\n'
        stream = Guarded(); a = e.prefix_rows(stream, 3, 2)
        self.assertEqual(a.shape, (3, 2)); self.assertEqual(stream.n, 3)
        with self.assertRaises(ValueError): e.prefix_rows(io.StringIO('1\n'), 2, 1)

    def test_cross_role_and_label_prefix_rejected(self):
        bad = copy.deepcopy(self.kw['roles']); bad['SUPPORT'][1] = 9386
        with self.assertRaises(ValueError): self.build(roles=bad)
        bad = copy.deepcopy(self.kw['roles']); bad['NORMAL_SELECT'][0] = 9478
        with self.assertRaises(ValueError): self.build(roles=bad)
        with self.assertRaises(ValueError): e.construct(self.values, self.labels[:16500], **self.kw)
        self.labels[-1] = 1
        with self.assertRaises(ValueError): self.build()

    def test_negative_contamination_rejects(self):
        self.labels[5941] = 1
        # Even with a faithfully updated catalogue, normal footprint rejects.
        cat = copy.deepcopy(self.catalogue)
        cat.insert(0, dict(event_id='EXTRA', start=5941, end=5942))
        with self.assertRaisesRegex(ValueError, 'contamination'): self.build(catalogue=cat)

    def test_positive_other_event_rejected(self):
        self.labels[5994] = 1
        cat = copy.deepcopy(self.catalogue); cat.insert(0, dict(event_id='EXTRA', start=5994, end=5995))
        with self.assertRaisesRegex(ValueError, 'unbudgeted'): self.build(catalogue=cat)

    def test_observed_all_channels_and_two_past(self):
        self.values[10533, 1] = np.nan
        r = self.build(); self.assertEqual(r['candidate_counts']['NORMAL_SELECT'], 1915)
        self.values[5995:6506, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'observation eligible'): self.build()

    def test_k3_one_k5_na_and_annotation_isolation(self):
        r = self.build()
        self.assertEqual(len([x for x in r['combinations'] if x['K']==3]), 1)
        self.assertEqual([x['status'] for x in r['combinations'] if x['K']==5], ['NA'])
        self.assertEqual(r['ledger']['label_prefix_points'], 17129)
        self.assertEqual(len(r['ledger']['prefix_events']), 6)
        encoded = json.dumps(r['plans'])
        for forbidden in ('ledger', 'labels', 'E4', 'E5', 'QUERY'):
            self.assertNotIn(forbidden, encoded)
        for x in r['combinations']:
            if x['status']=='AVAILABLE':
                self.assertNotIn('E4', json.dumps(x)); self.assertNotIn('E5', json.dumps(x))

    def test_original_axis_nan_zero_no_transformation(self):
        self.values[3,0] = np.nan; self.values[4:7,1] = 0
        r = self.build()
        np.testing.assert_array_equal(r['values'], self.values)
        self.assertTrue(np.isnan(r['values'][3,0])); self.assertFalse(r['observed'][3,0])
        self.assertTrue(r['observed'][4:7,1].all())
        self.assertEqual(r['values'].dtype, np.float32)

    def test_real_execute_needs_exact_root_release_before_read(self):
        cfg = dict(operation=e.OPERATION, roles=e.ROLES, history=512, horizon=32, sampling_seed=2025,
                   selection_counts=self.kw['counts'], identity=self.kw['identity'], catalogue=self.catalogue, normal_matches=self.normal,
                   inputs={name:dict(path='NEVER_OPEN', sha256=digest, bytes=size, numeric_stop=stop) for name,(digest,size,stop) in e.INPUTS.items()})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'; p.write_text(json.dumps(cfg))
            with patch.object(e, 'derive', side_effect=AssertionError('NUMERIC READ')):
                e.main(['--config',str(p)])
                with self.assertRaises(PermissionError): e.main(['--config',str(p),'--execute'])

    def test_portable_manifest_and_npz_copy_retains_hash_axis_and_plans(self):
        from hybrid_tsad.hybrid_v1.data import load_manifest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); lp=root/'synthetic_labels.txt'; vp=root/'synthetic_values.txt'
            lp.write_text(''.join('%d\n'%x for x in self.labels)+'FORBIDDEN_QUERY_NOT_NUMERIC\n')
            vp.write_text((','.join(['0']*38)+'\n')*16585+'FORBIDDEN_SUFFIX_NOT_NUMERIC\n')
            source=root/'original'; destination=root/'different'/'copied'; destination.mkdir(parents=True)
            cfg=dict(identity=self.kw['identity'], output=str(source), catalogue=self.catalogue,
                     normal_matches=self.normal, selection_counts=self.kw['counts'],
                     inputs={name:dict(path=str(p), sha256=e.sha(p), bytes=p.stat().st_size)
                             for name,p in [('labels',lp),('values',vp)]})
            with patch.object(e,'validate_config'):
                e.derive(cfg)
            names=['f_role_manifest.json','values_observed_original_axis.npz','normal_plans.json']
            original_hashes={name:e.sha(source/name) for name in names}
            for name in names:
                shutil.copy2(source/name,destination/name)
            # The old directory no longer exists: absolute references cannot
            # silently pass by continuing to load the original source NPZ.
            source.rename(root/'retired_original')
            self.assertFalse(source.exists())
            self.assertEqual(original_hashes,{name:e.sha(destination/name) for name in names})
            manifest=json.loads((destination/'f_role_manifest.json').read_text())
            for run in manifest['runs']:
                self.assertEqual(run['path'],'values_observed_original_axis.npz')
                self.assertFalse(Path(run['path']).is_absolute())
                self.assertEqual(run['sha256'],original_hashes['values_observed_original_axis.npz'])
            runs=load_manifest(destination/'f_role_manifest.json')
            self.assertEqual([(r.role,r.start,len(r.values)) for r in runs],
                             [('SUPPORT',0,9477),('NORMAL_SELECT',10021,3010),('NORMAL_CAL',13575,3010)])
            plans=json.loads((destination/'normal_plans.json').read_text())
            for role in ('NORMAL_SELECT','NORMAL_CAL'):
                resolved=resolve_plan([r for r in runs if r.role==role],plans[role],512,32)
                self.assertEqual(len(resolved),32 if role=='NORMAL_SELECT' else 512)
                for (run,local), point in zip(resolved,plans[role]):
                    self.assertEqual(run.start+local,point['index'])
                    self.assertEqual(window(run,local,512,32)['footprint'],point['footprint'])
            with np.load(destination/'values_observed_original_axis.npz',allow_pickle=False) as stored:
                self.assertEqual(stored['original_index'].tolist(),list(range(16585)))

    def test_synthetic_derived_files_roundtrip_and_ledger_separation(self):
        from hybrid_tsad.hybrid_v1.data import load_manifest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); lp=root/'synthetic_labels.txt'; vp=root/'synthetic_values.txt'
            lp.write_text(''.join('%d\n'%x for x in self.labels)+'FORBIDDEN_QUERY_NOT_NUMERIC\n')
            vp.write_text((',' .join(['0']*38)+'\n')*16585+'FORBIDDEN_SUFFIX_NOT_NUMERIC\n')
            cfg=dict(identity=self.kw['identity'], output=str(root/'derived'), catalogue=self.catalogue,
                     normal_matches=self.normal, selection_counts=self.kw['counts'],
                     inputs={name:dict(path=str(p), sha256=e.sha(p), bytes=p.stat().st_size)
                             for name,p in [('labels',lp),('values',vp)]})
            # Only production fixed-public-identity validation is bypassed for
            # two synthetic temp files. All parser/construction/writer work runs.
            with patch.object(e,'validate_config'):
                report=e.derive(cfg)
            self.assertFalse(report['query_numeric_read'])
            out=root/'derived'; manifest=json.loads((out/'f_role_manifest.json').read_text())
            plans=json.loads((out/'normal_plans.json').read_text())
            self.assertTrue(manifest['explicit_normal_plans_required'])
            self.assertNotIn('label',json.dumps(manifest)); self.assertNotIn('ledger',json.dumps(plans))
            self.assertEqual(len(json.loads((out/'annotation_ledger.NOT_MODEL_INPUT.json').read_text())['prefix_events']),6)
            runs=load_manifest(out/'f_role_manifest.json')
            self.assertEqual([(r.role,r.start,len(r.values)) for r in runs],
                             [('SUPPORT',0,9477),('NORMAL_SELECT',10021,3010),('NORMAL_CAL',13575,3010)])
            for role in ('NORMAL_SELECT','NORMAL_CAL'):
                resolve_plan([r for r in runs if r.role==role],plans[role],512,32)
            with np.load(out/'values_observed_original_axis.npz',allow_pickle=False) as stored:
                self.assertEqual(stored['original_index'].tolist(),list(range(16585)))
            index=json.loads((out/'support_index.json').read_text())
            self.assertEqual([x['K'] for x in index],[1,1,1,3,5])
            for item in index:
                if item['status']=='AVAILABLE':
                    support=json.loads((out/item['path']).read_text())
                    self.assertEqual(len(support['events']),item['K'])
                    self.assertNotIn('E4',json.dumps(support)); self.assertNotIn('E5',json.dumps(support))
            with patch.object(e,'validate_config'):
                with self.assertRaises(FileExistsError): e.derive(cfg)


if __name__ == '__main__': unittest.main()
