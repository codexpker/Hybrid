"""Only synthetic arrays/metadata/callbacks; no model or actual input file."""
import copy
import unittest
import numpy as np
from hybrid_tsad import tep_normal_control as control
from hybrid_tsad.normal_b1_readout import collect_bank, score_identity, validate_query_manifest

def manifest():
    raw='a'*64
    rows=[dict(entity_id=e,role='NORMAL_CONTROL',faultNumber=0,simulationRun=r,
        path='/home/a312/tsad_research/results/tep_normal_control_inputs_run51_70_20261102_v1/inputs/'+e+'.npz',
        sha256='b'*64,bytes=24000,shape=[960,3],dtype='float64',columns=list(control.COLUMNS),raw_sha256=raw)
        for e,r in zip(control.ENTITIES,control.RUNS)]
    return dict(status='PUBLIC_SIMULATION_NORMAL_CONTROL_INPUTS',records=rows,
        raw_source='TEP_FaultFree_Testing.RData',raw_sha256=raw,sample_axis=[1,960,1],
        imputation='none',standardization='none',sorting='none')

def arrays(run):
    return dict(values=np.full((960,3),float(run),np.float64),sample_1based=np.arange(1,961,dtype=np.int64),
                source_row_0based=np.arange((run-1)*960,run*960,dtype=np.int64))

def fingerprint():
    return dict(state='s',gradients='g',modes={'':False},step=0,rng='unchanged')

class NormalControl(unittest.TestCase):
    def test_exact20_normal_manifest_and_arrays(self):
        rows=control.validate_manifest(manifest())
        self.assertEqual(len(rows),20)
        for r in rows:self.assertEqual(control.validate_arrays(arrays(r['simulationRun'])).shape,(960,3))

    def test_400_faults_missing_duplicate_or_reordered_rejected(self):
        for mode in ['400','missing','duplicate','reorder']:
            m=manifest()
            if mode=='400':m['records']*=20
            elif mode=='missing':m['records'].pop()
            elif mode=='duplicate':m['records'][1]=m['records'][0]
            else:m['records'].reverse()
            with self.assertRaises((ValueError,PermissionError)):control.validate_manifest(m)

    def test_role_identity_type_and_fault_rejected(self):
        for key,value in [('role','QUERY'),('entity_id','entity_0001'),('faultNumber',1),('faultNumber',False),('simulationRun',51.0)]:
            m=manifest();m['records'][0][key]=value
            with self.assertRaises((ValueError,PermissionError)):control.validate_manifest(m)

    def test_closed_schema_nested_labels_and_bad_paths(self):
        changes=[('top','labels',[]),('record','onset',161),('record','columns',{'xmeas_9':{'label':1}}),
                 ('record','path','C:\\normal.npz'),('record','raw_sha256','c'*64)]
        for where,key,value in changes:
            m=manifest();(m if where=='top' else m['records'][0])[key]=value
            with self.assertRaises((ValueError,PermissionError)):control.validate_manifest(m)

    def test_axis_missing_duplicate_nonfinite_and_extra_payload(self):
        for mode in ['missing','duplicate','row','nan','extra']:
            a=arrays(51)
            if mode=='missing':a['values']=a['values'][:-1]
            elif mode=='duplicate':a['sample_1based'][1]=1
            elif mode=='row':a['source_row_0based'][1]=a['source_row_0based'][0]
            elif mode=='nan':a['values'][0,0]=np.nan
            else:a['label']=np.zeros(960)
            with self.assertRaises((ValueError,PermissionError)):control.validate_arrays(a)

    def test20_banks_reset_and_first_last_mean_geometry(self):
        for run in control.RUNS:
            count=[0]
            def fake(batch):
                self.assertEqual(batch.shape,(1,100,3));count[0]+=1
                return batch.copy()
            b=control.collect_normal_bank(arrays(run)['values'],run=run,seed=2025,scorer=fake,fingerprint=fingerprint)
            self.assertEqual(count[0],861)
            np.testing.assert_array_equal(b['end_index_0based'],np.arange(99,960))
            self.assertEqual(b['raw_scores'].shape,(861,100,3))
            np.testing.assert_array_equal(b['last'],np.full(861,run,np.float32))
            np.testing.assert_array_equal(b['mean32'],b['last'])

    def test_frozen_original_math_equivalence_synthetic_oracle_only(self):
        x=np.arange(2880,dtype=np.float64).reshape(960,3)/100
        fake=lambda batch:batch.copy()
        a=control.collect_normal_bank(x,run=51,seed=2025,scorer=fake,fingerprint=fingerprint)
        # Separate synthetic oracle identity; not a runtime reclassification of a normal run.
        old=dict(domain='TEP_REACTOR',role='QUERY',purpose='FIXED_PUBLIC_QUERY',entity_id='entity_0001',
            seed=2025,points=960,window=100,batch_size=1,columns=list(control.COLUMNS),
            array_sha256='a'*64,scaler_sha256='b'*64,checkpoint_sha256='c'*64,
            feature_source_sha256='d'*64,best_checkpoint_step=1880)
        b,_=collect_bank(x,old,fake,fingerprint)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(a['mean32'][0],a['raw_scores'][0].mean(axis=1,dtype=np.float32)[-32:].mean(dtype=np.float32))

    def test_old_guards_still_reject_actual_normal_role(self):
        with self.assertRaises((PermissionError,ValueError)):validate_query_manifest(manifest())
        with self.assertRaises((PermissionError,ValueError)):
            score_identity(dict(domain='TEP_REACTOR',role='NORMAL_CONTROL',purpose='NORMAL_CONTROL',run_id=51,points=960))

    def test_mutation_fingerprint_and_shape_failures(self):
        for mode in ['mutate','fingerprint','batch']:
            calls=[0]
            def fp():
                f=fingerprint();calls[0]+=1
                if mode=='fingerprint':f['step']=calls[0]
                return f
            def scorer(batch):
                if mode=='mutate':batch[0,0,0]+=1
                return batch.copy() if mode!='batch' else np.repeat(batch,2,axis=0)
            with self.assertRaises((RuntimeError,ValueError)):
                control.collect_normal_bank(arrays(51)['values'],run=51,seed=2025,scorer=scorer,fingerprint=fp)

    def test_pure_group_exact_selected_frame_and_no_full_parse(self):
        import pandas as pd
        a=arrays(51)
        frame=pd.DataFrame(dict(faultNumber=np.zeros(960,dtype=np.int64),simulationRun=np.full(960,51),
            sample=a['sample_1based'],_source_row_0based=a['source_row_0based'],
            **{name:a['values'][:,i] for i,name in enumerate(control.COLUMNS)}))
        got=control.extract_group_arrays(frame,51)
        for k in a:np.testing.assert_array_equal(got[k],a[k])
        with self.assertRaises(ValueError):control.group_normal_control_arrays(frame)
        frame.loc[1,'sample']=1
        with self.assertRaises(ValueError):control.extract_group_arrays(frame,51)

if __name__=='__main__':unittest.main(verbosity=2)
