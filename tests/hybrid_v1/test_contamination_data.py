"""Synthetic-only NumPy builder tests. Never read project numeric data."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import contamination_data as c

class Builder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v=np.zeros((5000,3),np.float32);cls.y=np.zeros(5000,np.int8)
        cls.events=[(500*i,500*i+120) for i in range(1,9)]
        for s,e in cls.events:cls.y[s:e]=1
        cls.a=c.admit(cls.v,cls.y,cls.events,4,2)
        base=dict(domain='SMD',entity='machine-1-1',run='machine-1-1_train',role='FIT',index=100,footprint=[96,102])
        cls.plan=dict(history=4,horizon=2,seed=2025,FIT=[dict(base,domain='SMD' if i%2==0 and i<7964 else 'Sunspot') for i in range(8000)],SELECT=[dict(base,role='SELECT') for _ in range(64)],CAL=[dict(base,role='CAL') for _ in range(1024)])
        cls.manifest=dict(runs=[dict(domain='SMD',entity='machine-1-1',run='machine-1-1_train',role='FIT',path='BASE_NOT_OPENED.npz')])
        cls.donor=dict(c.IDENTITY,path='DONOR_NOT_OPENED.npz',format='npz',key='values',observed_key='observed',sha256='synthetic',bounds=[0,5000],original_start=0,expected_shape=[5000,3])
    def build(self,a=None):return c.construct_plan(self.manifest,self.plan,self.a if a is None else a,self.donor)
    def test_two_file_identities_before_any_numeric_parse(self):
        with tempfile.TemporaryDirectory() as d:
            specs={};raw={}
            for key in ('values','labels'):
                p=Path(d)/key;p.write_text('synthetic');raw[key]=(c.sha(p),p.stat().st_size);specs[key]=dict(path=str(p),sha256=raw[key][0],bytes=raw[key][1])
            with patch.dict(c.RAW,raw,clear=True):
                c.verify_raw(specs)
                for key in specs:
                    bad=copy.deepcopy(specs);bad[key]['sha256']='WRONG'
                    with patch.object(np,'genfromtxt',side_effect=AssertionError('NUMERIC')):
                        with self.assertRaises(PermissionError):c.verify_raw(bad)
                    p=Path(specs[key]['path']);p.write_text('different')
                    with self.assertRaises(ValueError):c.verify_raw(specs)
                    p.write_text('synthetic')
    def test_complete_directory_boundary_other_event_refused(self):
        with self.assertRaisesRegex(ValueError,'directory'):c.admit(self.v,self.y,self.events[:-1],4,2)
        for intervals in [[(2,5)],[(20,22),(25,27)]]:
            y=np.zeros(60,np.int8)
            for s,e in intervals:y[s:e]=1
            with self.assertRaisesRegex(ValueError,'expanded'):c.admit(np.zeros((60,3)),y,intervals,10,2)
    def test_natural_missing_current_and_past_no_imputation(self):
        v=self.v.copy();v[500,1]=np.nan;v[598:600,2]=np.nan
        a=c.admit(v,self.y,self.events,4,2)
        self.assertNotIn(500,a['positive']);self.assertNotIn(600,a['positive'])
        self.assertTrue(np.isnan(a['values'][500,1]));self.assertFalse(a['observed'][500,1])
        v[10,0]=np.inf
        with self.assertRaises(ValueError):c.admit(v,self.y,self.events,4,2)
    def test_full_normal_current_future_history_and_tail(self):
        normal=set(self.a['normal'].tolist())
        self.assertNotIn(499,normal) # current0 but future includes500
        self.assertNotIn(620,normal) # current0 but history includes event
        self.assertNotIn(4999,normal);self.assertIn(4998,normal)
        self.assertTrue(all(not self.y[t-4:t+2].any() for t in normal))
    def test_exclusive_source_query_F_and_wrong_record(self):
        for role in ('SELECT','CAL','QUERY','SUPPORT'):
            with self.assertRaises(ValueError):c.exclusive(c.IDENTITY,[dict(c.IDENTITY,role=role)])
        for domain,entity,run in [('PSM','PSM','test'),('SMD','machine-2-1','machine-2-1_test'),('SMD','machine-3-1','machine-3-1_test'),('SMD','machine-1-2','machine-1-2_test')]:
            with self.assertRaises(ValueError):c.exclusive(dict(domain=domain,entity=entity,run=run,role='FIT'),[])
        manifest=copy.deepcopy(self.manifest);manifest['runs'].append(dict(c.IDENTITY,role='SELECT'))
        with self.assertRaises(ValueError):c.construct_plan(manifest,self.plan,self.a,self.donor)
    def test_rounding_denominators_and_no_8000_rate_substitution(self):
        self.assertEqual([c.count_for_rate(r) for r in (0,1,5,10,20)],[0,40,199,398,796])
        self.assertEqual(c.count_for_rate(10)/8000,.04975)
        self.assertEqual(c.count_for_rate(1),40) # floor+half; not truncation39
        for r,n in [(10,8000),(2,3982),(.1,3982)]:
            with self.assertRaises(ValueError):c.count_for_rate(r,n)
    def test_unique_nearest_normal_matching_tie_order(self):
        self.assertEqual(c.nearest_unique([10,10,12],[8,12,14]),[8,12,14])
        self.assertEqual(c.nearest_unique([1,2,3],[0]),[0])
    def test_nested_seed_common_slots_and_old_plan_unchanged(self):
        before=json.dumps(self.plan);mbefore=json.dumps(self.manifest);r=self.build();previous=set()
        ordered=np.random.default_rng(99179).permutation(np.flatnonzero([x['domain']=='SMD' for x in self.plan['FIT']])).tolist()
        for rate in (0,1,5,10,20):
            name='base0' if rate==0 else 'anomaly'+str(rate);row=r['auditledger'][name];ids=[x['attempt'] for x in row['changes']]
            self.assertEqual(ids,ordered[:c.count_for_rate(rate)]);self.assertTrue(previous.issubset(ids));previous=set(ids)
            derived=r['panels'][name]['plan'];self.assertEqual(derived['SELECT'],self.plan['SELECT']);self.assertEqual(derived['CAL'],self.plan['CAL'])
            for i,old in enumerate(self.plan['FIT']):
                self.assertEqual(derived['FIT'][i]['domain'],old['domain'])
                if i not in previous:self.assertEqual(derived['FIT'][i],old)
            self.assertEqual(len({x['donor']['index'] for x in row['changes']}),len(ids))
        self.assertEqual([x['attempt'] for x in r['auditledger']['anomaly10']['changes']],[x['attempt'] for x in r['auditledger']['normal_record10']['changes']])
        self.assertEqual(json.dumps(self.plan),before);self.assertEqual(json.dumps(self.manifest),mbefore)
        self.assertEqual(r['panels']['base0']['plan'],self.plan);self.assertEqual(len(r['matrix']),6)
    def test_insufficient_unique_origins_NA_no_resample(self):
        a=dict(self.a,positive=self.a['positive'][:100]);r=self.build(a)
        self.assertEqual(r['auditledger']['anomaly10']['status'],'NA');self.assertEqual(r['auditledger']['anomaly1']['status'],'AVAILABLE')
        self.assertNotIn('anomaly10',r['panels'])
    def test_learning_has_no_supervision_and_exposure_math(self):
        r=self.build()
        for panel in r['panels'].values():c.no_supervision(panel)
        for key in ('label_path','event_id','sample_weight','positive_flag'):
            with self.assertRaises(ValueError):c.no_supervision({key:'FORBIDDEN'})
        e=c.exposure(self.a,[500,500]);self.assertEqual(e['unique_origins'],1);self.assertEqual(e['unique_footprint_points'],6);self.assertEqual(e['repeated_point_exposures'],6)
        self.assertEqual(e['unique_footprint_intervals'],[[496,502]]);self.assertEqual(e['all_exposed_event_ids'],['SMD:machine-1-1:E0'])
        self.assertEqual(e['origins'][0]['future_GT'],dict(points=2,anomaly_points=2,observed_cells=6,anomaly_observed_cells=6));self.assertEqual(e['base_true_contamination'],'UNKNOWN');self.assertEqual(e['actual_updated_skipped'],'NOT_RUN')
    def test_byte_pinned_legacy_anti_label_declaration_not_new_supervision(self):
        legacy=dict(self.manifest,label_files_not_supported=True)
        r=c.construct_plan(legacy,self.plan,self.a,self.donor)
        self.assertEqual(r['panels']['base0']['manifest'],legacy)
        for key,panel in r['panels'].items():
            if key!='base0':c.no_supervision(panel);self.assertEqual(set(panel['manifest']),{'runs'})
        with self.assertRaises(ValueError):c.validate_base_manifest(dict(legacy,label_path='NO'))
        with self.assertRaises(ValueError):c.validate_base_manifest(dict(legacy,label_files_not_supported=False))
    def test_true_cold_import_closure_and_preview_no_body(self):
        source=Path(c.__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cold=root/'source'
            for name in c.CLOSURE:
                p=cold/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p)
            cfg=root/'cfg.json';cfg.write_text(json.dumps(dict(operation=c.OPERATION,identity=c.IDENTITY,forbidden_records=[])))
            code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);assert not any('research_workspace' in p.lower() for p in sys.path);from hybrid_tsad.hybrid_v1 import contamination_data as c;c.main(['--config',sys.argv[2],'--phase','donor_admit']);files={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n=='hybrid_tsad' or n.startswith('hybrid_tsad.')};assert len(files)==5;assert all(Path(sys.argv[1]) in Path(v).parents for v in files.values());assert 'torch' not in sys.modules;print(json.dumps(files))"
            result=subprocess.run([sys.executable,'-I','-c',code,str(cold),str(cfg)],cwd=root,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);print('H5_COLD '+result.stdout)

if __name__=='__main__':unittest.main()
