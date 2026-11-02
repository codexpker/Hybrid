"""NumPy metadata/cap-order mocks only, never actual CUDA/Torch/model."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from hybrid_tsad.hybrid_v1 import event_readout as r,event_query as q


def child_module():
    # Local source tests and frozen test copies both use an explicit package path.
    import os
    package=Path(os.environ['HYBRID_EVENT_QUERY_PREP'])
    spec=importlib.util.spec_from_file_location('event_query_resource_test',package/'resources/event_query_child.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m,package


class EventReadoutMetadata(unittest.TestCase):
    def test_query_manifest_exact_suffix_no_prefix_or_label(self):
        manifest=dict(runs=[dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test',role='QUERY',format='npz',key='values',observed_key='observed',expected_shape=[6565,38],bounds=[0,6565],original_start=17129)],source_raw=dict(sha256=q.RAW_SHA,numeric_bounds=[17129,23694]),geometry=q.geometry())
        self.assertEqual(r.query_metadata(manifest)['original_start'],17129)
        for change in ('prefix','role','start','label'):
            v=copy.deepcopy(manifest)
            if change=='prefix':v['runs'][0]['expected_shape']=[16585,38]
            elif change=='role':v['runs'][0]['role']='NORMAL_CAL'
            elif change=='start':v['runs'][0]['original_start']=0
            else:v['labels']='forbidden'
            with self.subTest(change=change),self.assertRaises(PermissionError):r.query_metadata(v)

    def test_preview_and_pending_before_any_Torch_or_asset(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('No IO')),contextlib.redirect_stdout(io.StringIO()):r.main(['--config','missing'])
        with self.assertRaises(PermissionError):r.preflight('missing',None,None)

    def test_resource_exact_operation_caps_and_fresh_lifecycle(self):
        child,p=child_module();tmp=p/'synthetic_tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(tmp)) as d:
            root=Path(d);resources=root/'resources';resources.mkdir()
            for n in ('event_query_child.py','launch_event_query_once.py','owned_session.py'):shutil.copyfile(str(p/'resources'/n),str(resources/n))
            with patch.object(child,'__file__',str(resources/'event_query_child.py')):
                cfg=dict(operation=r.OPERATION,physical_gpu=2,expected_gpu_uuid='GPU-synthetic',device='cuda:0',output=str(root/'out'),resources=dict(wall_seconds=60,kill_grace_seconds=5,evaluation_GiB=30,automatic_retry=False))
                cp=root/'config.json';cp.write_text(json.dumps(cfg));release=root/'release.json'
                signed=dict(status='APPROVED',approved_by='root',operation=r.OPERATION,allowed_actions=['event_model_infer'],config_sha256=child.sha(cp),resource_sha256=child.resource_identity(),physical_gpu=2,expected_gpu_uuid='GPU-synthetic',resources=cfg['resources'],timeout_seconds=60,lifecycle=str(root/'lifecycles'/'one'))
                def sign():release.write_text(json.dumps(signed));return child.sha(release)
                h=sign();self.assertEqual(child.metadata(cp,release,h)[0]['physical_gpu'],2)
                signed['operation']='HYBRID_V1_STAGE2_EVENT_F';h=sign()
                with self.assertRaises(PermissionError):child.metadata(cp,release,h)

    def test_cap_precedes_model_and_fresh_failure_never_imports_Torch(self):
        child,p=child_module();tmp=p/'synthetic_tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(tmp)) as d:
            root=Path(d);life=root/'life';life.mkdir();events=[]
            cfg=dict(physical_gpu=1,expected_gpu_uuid='GPU-fake',resources=dict(wall_seconds=60));signed=dict(config_sha256='a'*64)
            cuda=SimpleNamespace(get_device_properties=lambda i:SimpleNamespace(uuid='GPU-fake',name='A40',total_memory=48*1024**3),set_per_process_memory_fraction=lambda x,i:events.append('CAP'),reset_peak_memory_stats=lambda i:None,max_memory_allocated=lambda i:0,max_memory_reserved=lambda i:0)
            def science(*args):
                self.assertTrue((life/'GPU_CAP_BEFORE_MODEL.json').is_file());self.assertIn('CAP',events);events.append('MODEL');return {'mock':True}
            def imported(name):
                events.append(name)
                return SimpleNamespace(cuda=cuda) if name=='torch' else SimpleNamespace(execute=science)
            with patch.object(child,'preflight',return_value=(cfg,signed,life)),patch.object(child,'fresh_gpu',side_effect=lambda c:events.append('FRESH') or 'fake'),patch.object(child.importlib,'import_module',side_effect=imported),patch.dict(child.os.environ,HYBRID_F_QUERY_CHILD='b'*64,CUDA_VISIBLE_DEVICES='1'):
                child.execute('c','r','b'*64)
            self.assertLess(events.index('FRESH'),events.index('torch'));self.assertLess(events.index('CAP'),events.index('MODEL'))
            with patch.object(child,'preflight',return_value=(cfg,signed,life)),patch.object(child,'fresh_gpu',side_effect=RuntimeError('busy')),patch.object(child.importlib,'import_module',side_effect=AssertionError('Must not import Torch')),patch.dict(child.os.environ,HYBRID_F_QUERY_CHILD='b'*64,CUDA_VISIBLE_DEVICES='1'):
                with self.assertRaises(RuntimeError):child.execute('c','r','b'*64)

    def test_cold_import_retains_accepted_helper(self):
        self.assertEqual(q.sha(Path(q.__file__)),'327214150a7300d26d7691184064a175b874e6b7d6cf8b915444b4b5e8ad805d')
        src=str(Path(r.__file__).resolve().parents[2]);code='import sys;sys.path.insert(0,'+repr(src)+');import hybrid_tsad.hybrid_v1.event_readout;assert "torch" not in sys.modules;print("F_EXEC_COLD")'
        a=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20);self.assertEqual(a.returncode,0,a.stderr)

    def test_whole_metadata_seal_rejects_changed_assets_and_U_phase(self):
        from test_event_query import bindings
        from hybrid_tsad.hybrid_v1 import provenance
        c,d,_=bindings();parent=c['assets']['parent_source_checkpoint']['sha256'];real_sha=q.sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def document(name,value):
                p=root/(name+'.json');p.write_text(json.dumps(value));return dict(path=str(p),sha256=real_sha(p))
            c.update(permissions=dict(query=True,labels=False,training=False,threshold_fit=False,target_fit=False),chunk_size=64,active_components=['r_tf','r_ms','r_pred'],sources={},output=str(root/'fresh'),source_sha256={})
            base=Path(r.__file__).resolve().parents[2]
            names=['__init__','cli','config','data','event_adaptation','event_data','target_data','fractional','lag','model','multiview','objectives','provenance','scoring','temporal','training','readout','event_query','chunked_bank','event_readout']
            c['source_sha256']={'hybrid_tsad/hybrid_v1/'+n+'.py':real_sha(base/'hybrid_tsad/hybrid_v1'/(n+'.py')) for n in names}
            source=root/'parent.pt';source.write_bytes(b'SYNTHETIC PARENT ONLY');adapted=root/'F.pt';adapted.write_bytes(b'SYNTHETIC ADAPTED ONLY')
            c['assets']={'parent_source_checkpoint':dict(path=str(source),sha256=parent),'checkpoint':dict(path=str(adapted),sha256=real_sha(adapted))}
            for k in ('history','horizon','sources','active_components'):d['source_config'][k]=c[k];d['F_cal_config'][k]=c[k]
            d['F_train_receipt']['adapted_checkpoint_sha256']=real_sha(adapted);d['F_cal_config']['adapted_checkpoint_sha256']=real_sha(adapted)
            c['assets']['source_config']=document('source_config',d['source_config'])
            c['assets']['source_receipt']=document('source_receipt',dict(phase='source_train',files={'source.pt':parent,'config.json':c['assets']['source_config']['sha256']}))
            c['assets']['training_receipt']=document('F_train_receipt',d['F_train_receipt']);d['F_cal_config']['adapted_receipt_sha256']=c['assets']['training_receipt']['sha256']
            traincfg=dict(d['F_cal_config']);c['assets']['training_config']=document('F_train_config',traincfg)
            for name,key in [('calibration_config','F_cal_config'),('calibration_receipt','F_cal_receipt'),('calibration','calibration')]:c['assets'][name]=document(name,d[key])
            c['assets']['training_acceptance']=document('training_acceptance',dict(status='ACCEPTED_SYNTHETIC_FIXTURE',approved_by='root',adapted_checkpoint_sha256=c['assets']['checkpoint']['sha256']))
            c['assets']['calibration_acceptance']=document('calibration_acceptance',dict(status='ACCEPTED_SYNTHETIC_FIXTURE',approved_by='root',calibration_sha256=c['assets']['calibration']['sha256'],receipt_sha256=c['assets']['calibration_receipt']['sha256']))
            npz=root/'not_loaded.npz';npz.write_bytes(b'NOT A NUMPY ARRAY: METADATA ONLY')
            run=dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test',role='QUERY',format='npz',key='values',observed_key='observed',expected_shape=[6565,38],bounds=[0,6565],original_start=17129,path=str(npz),sha256=real_sha(npz))
            c['assets']['query_manifest']=document('query_manifest',dict(runs=[run],geometry=q.geometry(),source_raw=dict(sha256=q.RAW_SHA,numeric_bounds=[17129,23694])))
            c['assets']['query_input_acceptance']=document('query_input_acceptance',dict(status='ACCEPTED_SMD_EVENT_QUERY_INPUTS',approved_by='root',manifest_sha256=c['assets']['query_manifest']['sha256'],NPZ_sha256=real_sha(npz),numeric_bounds=[17129,23694],labels_read=False))
            seal=dict(status='ACCEPTED_SMD_EVENT_QUERY_ASSETS',approved_by='root',assets=c['assets'],choice=c['choice'],score_kind='raw_event_logit');c['asset_seal']=document('seal',seal)
            config=root/'config.json';release=root/'release.json'
            def sign():
                config.write_text(json.dumps(c));release.write_text(json.dumps(dict(status='APPROVED',approved_by='root',operation=r.OPERATION,allowed_actions=[r.ACTION],config_sha256=real_sha(config),allow_labels=False,source_sha256=c['source_sha256'],asset_seal_sha256=c['asset_seal']['sha256'],external_source_sha256={})));return real_sha(release)
            with patch.object(q,'sha',side_effect=lambda p:parent if Path(p)==source else real_sha(p)),patch.object(provenance,'OfficialSources') as official,patch('numpy.load',side_effect=AssertionError('No array load during metadata')):
                official.return_value.manifest.return_value=[]
                self.assertEqual(r.preflight(config,release,sign())[3]['score_kind'],'raw_event_logit')
                p=Path(c['assets']['calibration']['path']);old=p.read_bytes();p.write_bytes(old+b' ')
                with self.assertRaises(PermissionError):r.preflight(config,release,sign())
                p.write_bytes(old);c['assets']['training_receipt']=document('F_train_receipt',dict(d['F_train_receipt'],phase='target_train'))
                # A freshly root-sealed U receipt still must not pass the F schema.
                seal['assets']=c['assets'];c['asset_seal']=document('seal',seal)
                with self.assertRaises(ValueError):r.preflight(config,release,sign())


if __name__=='__main__':unittest.main()
