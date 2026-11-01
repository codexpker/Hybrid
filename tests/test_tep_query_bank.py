import copy,importlib.util,os,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hybrid_tsad.normal_b1_readout import collect_bank,validate_query_manifest,require_runtime_release,QUERY_OPERATION

class QueryBank(unittest.TestCase):
    def meta(self):
        return dict(domain='TEP_REACTOR',role='QUERY',purpose='FIXED_PUBLIC_QUERY',entity_id='entity_0001',
            seed=2025,points=960,window=100,batch_size=1,columns=['xmeas_9','xmeas_21','xmv_10'],
            array_sha256='a'*64,scaler_sha256='b'*64,checkpoint_sha256='c'*64,feature_source_sha256='d'*64,best_checkpoint_step=1880)
    def fp(self):return dict(state='a',gradients='none',modes={'':False},step=0,rng={'all':'fixed'})
    def test960_861windows_original_coords_and_channels(self):
        x=np.arange(2880,dtype=np.float64).reshape(960,3)/2880
        a,r=collect_bank(x,self.meta(),lambda b:b*b,self.fp)
        self.assertEqual(a['raw_scores'].shape,(861,100,3));self.assertEqual(r['steady_calls'],860)
        np.testing.assert_array_equal(a['end_index_0based'],np.arange(99,960))
        np.testing.assert_array_equal(a['available_index_0based'],a['end_index_0based'])
        np.testing.assert_array_equal(a['channel_last'],a['raw_scores'][:,-1,:])
        np.testing.assert_array_equal(a['channel_mean32'],a['raw_scores'][:,-32:,:].mean(1))
    def test500_existing_arrays_bitexact_against_accepted_snapshot(self):
        p=Path(__file__).resolve().parents[2]/'preparations/left_tep_normal_calibration_build_20261101_v1/runtime/src/hybrid_tsad/normal_b1_readout.py'
        spec=importlib.util.spec_from_file_location('accepted379cf',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        m=self.meta();m.pop('entity_id');m.update(role='CAL',purpose='NORMAL_CALIBRATION',run_id=81,fault_number=0,points=500)
        x=np.arange(1500,dtype=np.float64).reshape(500,3)/1500
        a,_=collect_bank(x,m,lambda b:b*b,self.fp);b,_=old.collect_bank(x,m,lambda b:b*b,self.fp)
        for k in b:
            if k!='scorer_and_cpu_return_seconds':np.testing.assert_array_equal(a[k],b[k])
    def test_pending_input_acceptance_blocks_before_release_file(self):
        with patch('hybrid_tsad.normal_b1_readout.read_json',side_effect=AssertionError('No release IO')) as read:
            with self.assertRaises(PermissionError):require_runtime_release(dict(operation=QUERY_OPERATION,seed=2025,binding_state='PENDING_INPUT_ACCEPTANCE'),'x','y')
            read.assert_not_called()
    def test_no_fault_run_onset_fields_allowed(self):
        for key in ('faultNumber','simulationRun','label','onset'):
            m=self.meta();m[key]=1
            with self.assertRaises(PermissionError):collect_bank(np.zeros((960,3)),m,None,None)
    def test400_manifest_fixed_opaque_paths_no_sealed(self):
        records=[dict(entity_id='entity_%04d'%i,path='/home/a312/tsad_research/results/tep_faulty_inputs_run51_70_20261102_v1/inputs/entity_%04d.npz'%i,shape=[960,3],dtype='float64',columns=['xmeas_9','xmeas_21','xmv_10'],sha256='a'*64,raw_sha256='b'*64,bytes=24000) for i in range(1,401)]
        m=dict(status='PUBLIC_SIMULATION_INPUTS_NO_LABEL_FEATURES',records=records,
            raw_source='/home/a312/tsad_research/data/tep_rieth2017_v1/TEP_Faulty_Testing.RData',raw_sha256='b'*64,
            sample_axis='original 1..960; no absolute datetime',imputation=False,standardization=False,sorting=False)
        self.assertEqual(len(validate_query_manifest(m)),400)
        for change in ('missing','label','sealed'):
            bad=copy.deepcopy(m)
            if change=='missing':bad['records'].pop()
            elif change=='label':bad['records'][0]['label']=1
            else:bad['records'][0]['path']=bad['records'][0]['path'].replace('inputs/','sealed_identity/')
            with self.assertRaises(PermissionError):validate_query_manifest(bad)
        for key in ('labels','sealed_identity','faultNumber','onset'):
            bad=copy.deepcopy(m);bad[key]={'unknown':1}
            with self.assertRaises(PermissionError):validate_query_manifest(bad)
        for key in ('labels','sealed_identity','faultNumber','onset'):
            bad=copy.deepcopy(m);bad['records'][0][key]={'unknown':1}
            with self.assertRaises(PermissionError):validate_query_manifest(bad)
        for location in ('sample_axis','raw_sha256','record_sha'):
            bad=copy.deepcopy(m)
            if location=='record_sha':bad['records'][0]['sha256']={'labels':[1]}
            else:bad[location]={'labels':[1]}
            with self.assertRaises(PermissionError):validate_query_manifest(bad)
        nested=copy.deepcopy(m)
        nested['records'][0]['columns']={'xmeas_9':{'labels':[1]},'xmeas_21':{'onset':161},'xmv_10':{'GT':True}}
        with self.assertRaises(PermissionError):validate_query_manifest(nested)
        for wrong in [('xmeas_9','xmeas_21','xmv_10'),['xmeas_9',{'GT':1},'xmv_10'],['xmeas_21','xmeas_9','xmv_10']]:
            bad=copy.deepcopy(m);bad['records'][0]['columns']=wrong
            with self.assertRaises(PermissionError):validate_query_manifest(bad)

if __name__=='__main__':unittest.main()
