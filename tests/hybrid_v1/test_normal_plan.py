"""Hand-fixed metadata examples; no sensor/label arrays or model calls."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from hybrid_tsad.hybrid_v1 import normal_plan as builder
from hybrid_tsad.hybrid_v1 import event_adaptation as event

class NormalPlan(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.remote='/home/a312/tsad_research/';ident=dict(domain='SMD',entity='fixture',run='fixture')
        self.ident=ident;self.c=dict(operation=builder.normal.OPERATION,mode='N_NEG300',choice='K1_choice0',history=512,horizon=32,seed=2025,
            steps=300,select_count=32,cal_count=512,permissions=dict(phase='target_normal_train',query=False,labels=False),
            source_checkpoint_sha256='source-fixture',source_config_sha256='config-fixture')
        support=dict(K=1,events=[dict(ident,event_id='E0',start=6000,end=6010)],
            normal=[dict(ident,matched_event_id='E0',start=5000,end=5010)],forbidden=[dict(ident,start=9477,end=23694)])
        runs=[dict(ident,role=role,bounds=event.ROLES[role]) for role in ('SUPPORT','NORMAL_SELECT','NORMAL_CAL')]
        plans=dict(history=512,horizon=32,seed=2025)
        for role,count in [('NORMAL_SELECT',32),('NORMAL_CAL',512)]:
            start=event.ROLES[role][0]+512
            plans[role]=[dict(ident,role=role,index=t,footprint=[t-512,t+32]) for t in range(start,start+count)]
        trace=dict(route='F',steps=[dict(attempt=a,updated=True,label=float(a%2==0),event_id='E0',
            footprint=[(6000 if a%2==0 else 5001)-512,(6000 if a%2==0 else 5001)+32],normal_ssl_footprint=[5000-512,5000+32],
            actual_mask_seed=2025+a,normal_ssl_mask_seed=12025+a) for a in range(300)])
        self.values=dict(manifest=dict(runs=runs),plans=plans,support=support,reference_trace=trace)
        for key in self.values:self.save(key,self.values[key])
        receipt=dict(status='COMPLETED',phase='target_event_train',adapted_checkpoint_sha256='adapted-fixture',
            **{k:self.c[k] for k in ('choice','manifest_sha256','plans_sha256','support_sha256','source_checkpoint_sha256')})
        self.save('reference_receipt',receipt)
        self.save('reference_acceptance',dict(approved_by='root',status='ACCEPTED_SYNTHETIC_FIXTURE',adapted_checkpoint_sha256='adapted-fixture'))
        self.config=self.root/'template.json';self.bind()
        self.patch1=patch.dict(event.ACCEPTED,{k:self.c[k+'_sha256'] for k in ('manifest','plans')},clear=True);self.patch1.start();self.addCleanup(self.patch1.stop)
        self.patch2=patch.dict(event.SUPPORT,{'K1_choice0':self.c['support_sha256']},clear=True);self.patch2.start();self.addCleanup(self.patch2.stop)
    def save(self,key,value):
        path=self.root/'metadata'/(key+'.json');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(value))
        self.c[key]=self.remote+'metadata/'+key+'.json';self.c[key+'_sha256']=event.sha(path)
    def bind(self):self.config.write_text(json.dumps(self.c))
    def derive(self):return builder.derive(self.config,self.root)
    def test_hand_fixed_A_B_clock_and_counts(self):
        before={p:p.read_bytes() for p in self.root.rglob('*.json')};plan,report=self.derive()
        self.assertIsNone(plan['rows'][0]['A']);self.assertEqual(plan['rows'][0]['B']['index'],5000)
        self.assertEqual(plan['rows'][1]['A']['index'],5001);self.assertEqual(plan['rows'][1]['A']['mask_seed'],2026)
        self.assertEqual(plan['rows'][1]['B']['mask_seed'],12026)
        self.assertEqual((report['normal_SSL'],report['normal_negative_BCE'],report['unique_normal_targets']),(300,150,2))
        self.assertEqual(sum(x['count'] for x in report['exposure_counts']),450)
        self.assertEqual(report['positive_forward_exposures'],0)
        self.assertEqual(before,{p:p.read_bytes() for p in before})
    def test_changed_trace_rejected_before_derivation(self):
        path=self.root/'metadata/reference_trace.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(PermissionError,'reference_trace'):self.derive()
    def test_wrong_reference_acceptance_not_a_new_normal_plan(self):
        self.save('reference_acceptance',dict(approved_by='root',status='ACCEPTED',adapted_checkpoint_sha256='wrong'));self.bind()
        with self.assertRaisesRegex(PermissionError,'lineage'):self.derive()
    def test_resigned_bad_mask_cannot_be_repaired_or_resampled(self):
        tr=copy.deepcopy(self.values['reference_trace']);tr['steps'][7]['normal_ssl_mask_seed']+=1
        self.save('reference_trace',tr);self.bind()
        with self.assertRaisesRegex(ValueError,'masks'):self.derive()
    def test_non_JSON_or_escape_path_rejected(self):
        for path in (self.remote+'metadata/x.npz',self.remote+'../outside.json'):
            self.c['reference_trace']=path;self.bind()
            with self.assertRaises(PermissionError):self.derive()
    def test_changed_CAL_clock_even_when_rehashed_is_rejected(self):
        p=copy.deepcopy(self.values['plans']);p['NORMAL_CAL'][0]['footprint'][1]-=31
        self.save('plans',p);self.bind()
        with patch.dict(event.ACCEPTED,{'plans':self.c['plans_sha256']}):
            # Receipt must still match exact plan identity as well as geometry.
            receipt=json.loads((self.root/'metadata/reference_receipt.json').read_text());receipt['plans_sha256']=self.c['plans_sha256']
            self.save('reference_receipt',receipt);self.bind()
            with self.assertRaisesRegex(ValueError,'footprint'):self.derive()

if __name__=='__main__':unittest.main()
