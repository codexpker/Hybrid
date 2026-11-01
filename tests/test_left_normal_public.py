"""Synthetic CPU data and pure-controller fixtures; no model/forward/optimizer."""
import ast
import copy
import json
import os
import builtins
import symtable
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.normal_run_data import NormalRun,fit_scaler,IndependentNormalWindows,inference_footprints
from hybrid_tsad.left_normal_public import (COLUMNS,TRAIN_CONFIG,effective_configs,
    eligible_checkpoints,normal_exp_class,validate_normal_metadata,load_normal_arrays,native_infer_score)


class PublicNormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'LEFT_SOURCE_ROOT' not in os.environ:raise RuntimeError('Pinned external source root required; no skip')
        cls.root=Path(os.environ['LEFT_SOURCE_ROOT'])
        cls.model=json.loads((cls.root/'configs/PSM/model_configs_Left.json').read_text())
        cls.data=json.loads((cls.root/'configs/PSM/model_configs_0.json').read_text())

    def fixture(self):
        rng=np.random.default_rng(422)
        ids=list(range(1,21))+list(range(51,61))
        def runs(ids,role):
            return [NormalRun('CPU_SYNTHETIC_TEP_SHAPE',role,i,COLUMNS,
                np.asfortranarray(rng.normal(size=(500,3))*[.02,.13,.55]+[120+i*.01,94,41])) for i in ids]
        return runs(ids,'FIT'),runs(range(61,81),'SELECT')

    def test_full_fixed_window_footprints_and_float64_fit(self):
        fit,select=self.fixture();scaler=fit_scaler(fit)
        train=IndependentNormalWindows(fit,scaler,'train',window=100)
        val=IndependentNormalWindows(select,scaler,'val',window=100)
        self.assertEqual((len(train),len(val)),(12030,100));self.assertEqual(scaler['fit_points'],15000)
        for ds in [train,val]:
            for i in range(len(ds)):
                f=ds.footprint(i)
                self.assertLessEqual(f['end_exclusive'],500);self.assertEqual(f['end_exclusive']-f['start'],100)
            for i in [0,400,401,len(ds)-1] if ds is train else [0,4,5,99]:
                f=ds.footprint(i);run=ds.runs[f['run_position']];x,_=ds[i]
                ref=((run.values-scaler['mean'])/scaler['scale'])[f['start']:f['end_exclusive']].astype(np.float32)
                np.testing.assert_array_equal(x,ref);self.assertEqual(x.shape,(100,3))
        self.assertEqual(train.footprint(400)['input_end_index'],499)
        self.assertEqual(train.footprint(401)['start'],0)
        self.assertEqual(val.footprint(99)['input_end_index'],499)

    def test_select_perturbation_cannot_fit_scaler(self):
        fit,select=self.fixture();a=fit_scaler(fit)
        train=IndependentNormalWindows(fit,a,'train',window=100)
        before=train[0][0].copy()
        for r in select:r.values[:]=1e8
        np.testing.assert_array_equal(a['mean'],fit_scaler(fit)['mean'])
        np.testing.assert_array_equal(before,train[0][0])
        with self.assertRaises(PermissionError):fit_scaler(select)
        with self.assertRaises(PermissionError):IndependentNormalWindows(fit,a,'test',window=100)
        wrong=copy.deepcopy(a);wrong['columns']=tuple(reversed(COLUMNS))
        with self.assertRaises(PermissionError):IndependentNormalWindows(select,wrong,'val',window=100)

    def test_effective_model_and_data_shapes_and_only_explicit_changes(self):
        before=copy.deepcopy((self.model,self.data));parsed=effective_configs(self.model,self.data,TRAIN_CONFIG)
        self.assertEqual((parsed['model']['seq_len'],parsed['model']['enc_in'],parsed['data']['seq_len'],parsed['data']['enc_in']),(100,3,100,3))
        self.assertEqual({k for k in self.model if self.model[k]!=parsed['model'][k]}, {'seq_len','enc_in','warmup_steps','ramp_steps'})
        self.assertEqual({k for k in self.data if self.data[k]!=parsed['data'][k]}, {'seq_len','enc_in'})
        self.assertEqual((self.model,self.data),before)
        self.assertEqual((parsed['model']['warmup_steps'],parsed['model']['ramp_steps'],parsed['model']['lambda_max']),(123,246,.3))

    def test_real_author_curriculum_method_without_importing_model(self):
        tree=ast.parse((self.root/'models/Left/LEFT.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Left')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_curriculum_lambda')
        namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'pinned_author_curriculum','exec'),namespace)
        f=namespace['_curriculum_lambda']
        for step,value in [(122,0),(123,0),(124,.3/246),(368,.3*245/246),(369,.3)]:
            self.assertAlmostEqual(f(SimpleNamespace(training=True,step=step,warmup_steps=123,ramp_steps=246,lambda_max=.3)),value)
            self.assertEqual(f(SimpleNamespace(training=False,step=step,lambda_max=.3)),.3)

    def test_eligibility_does_not_save_count_or_stop_before369(self):
        class FakeNative:
            def __init__(self,patience=3,**kw):
                self.counter=0;self.early_stop=False;self.best_score=None;self.val_loss_min=float('inf');self.saved=[]
            def __call__(self,loss,model,path):
                if self.best_score is None or -loss>=self.best_score:
                    self.best_score=-loss;self.val_loss_min=loss;self.counter=0;self.saved.append((loss,model.step))
                else:self.counter+=1;self.early_stop=self.counter>=3
        official=SimpleNamespace(EarlyStopping=FakeNative);events=[]
        with eligible_checkpoints(official,events):
            controller=official.EarlyStopping()
            for step in [94,188,282,368]:controller(-100,SimpleNamespace(step=step),'unused')
            self.assertEqual((controller.saved,controller.counter,controller.early_stop),([],0,False))
            controller(10,SimpleNamespace(step=369),'unused');controller(9,SimpleNamespace(step=376),'unused')
            for step,loss in [(470,10),(564,11),(658,12)]:controller(loss,SimpleNamespace(step=step),'unused')
            self.assertEqual(controller.saved,[(10,369),(9,376)]);self.assertTrue(controller.early_stop)
        self.assertIs(official.EarlyStopping,FakeNative)
        self.assertTrue(all(not x['native_called'] for x in events if x['model_step']<369))

    def test_controller_restores_on_exception(self):
        official=SimpleNamespace(EarlyStopping=object)
        with self.assertRaises(RuntimeError):
            with eligible_checkpoints(official,[]):raise RuntimeError('fixture abort')
        self.assertIs(official.EarlyStopping,object)

    def test_factory_inherits_loop_and_actual_cpu_loader_shapes(self):
        class FakeOfficialExp:
            def train(self):raise AssertionError('Never execute fixture training')
            def vali(self,*a):raise AssertionError('Never call real validation')
            def _select_optimizer(self):raise AssertionError('No optimizer allowed')
        fit,select=self.fixture();scaler=fit_scaler(fit)
        cls=normal_exp_class(SimpleNamespace(Exp_Anomaly_Detection=FakeOfficialExp),fit,select,scaler,[])
        self.assertIs(cls.train,FakeOfficialExp.train);self.assertIs(cls.vali,FakeOfficialExp.vali)
        self.assertIs(cls._select_optimizer,FakeOfficialExp._select_optimizer)
        fake=SimpleNamespace(train_configs=SimpleNamespace(batch_size=128))
        ds,loader=cls._get_data(fake,'train',1);self.assertEqual(len(loader),94)
        batches=list(loader);self.assertEqual(tuple(batches[0][0].shape),(128,100,3));self.assertEqual(tuple(batches[-1][0].shape),(126,100,3))
        ds,val=cls._get_data(fake,'val');self.assertEqual((len(ds),len(val)),(100,1))
        self.assertEqual(tuple(next(iter(val))[0].shape),(100,100,3))
        with self.assertRaises(PermissionError):cls._get_data(fake,'test')

    def test_real_array_body_unavailable_by_default(self):
        with patch('numpy.load',side_effect=AssertionError('No actual normal body allowed')) as load:
            with self.assertRaises(PermissionError):load_normal_arrays({'roles':{'CAL':[]}})
            load.assert_not_called()

    def test_existing_normal_metadata_rejects_role_dtype_or_column_changes(self):
        path=os.environ.get('NORMAL_METADATA_PATH')
        if path is None:raise RuntimeError('Existing metadata reference required, never an array body')
        source=json.loads(Path(path).read_text());validate_normal_metadata(source)
        cases=[]
        x=copy.deepcopy(source);x['roles']['CAL']=[];cases.append(x)
        x=copy.deepcopy(source);x['columns']=list(reversed(COLUMNS));cases.append(x)
        x=copy.deepcopy(source);x['roles']['FIT'][0]['dtype']='float32';cases.append(x)
        x=copy.deepcopy(source);x['roles']['SELECT'][0]['simulationRun']=51;cases.append(x)
        x=copy.deepcopy(source);x['roles']['FIT'][0]['raw_source']='TEP_Faulty_Testing.RData';cases.append(x)
        for x in cases:
            with self.assertRaises(PermissionError):validate_normal_metadata(x)

    def test_B1_inference_coordinates_do_not_wait_future127_windows(self):
        rows=inference_footprints(500,window=100,inference_batch_size=1)
        self.assertEqual(len(rows),401)
        self.assertEqual((rows[0]['endpoint'],rows[-1]['endpoint']),(99,499))
        self.assertTrue(all(r['available_index']==r['endpoint']==r['mean32_available_index'] for r in rows))
        self.assertEqual((1,rows[0]['end_exclusive']-rows[0]['start'],len(COLUMNS)),(1,100,3))
        with self.assertRaises(PermissionError):inference_footprints(500,window=100,inference_batch_size=128)

    def test_native_infer_tuple_is_unpacked_before_shape_access(self):
        tree=ast.parse((self.root/'models/Left/LEFT.py').read_text())
        author=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Left')
        infer=next(x for x in author.body if isinstance(x,ast.FunctionDef) and x.name=='infer')
        self.assertIsInstance(infer.body[-1],ast.Return)
        self.assertIsInstance(infer.body[-1].value,ast.Tuple)
        self.assertIsNone(infer.body[-1].value.elts[1].value)
        values=np.zeros((1,100,3),np.float32);calls=[]
        def fake_infer(x,*args):calls.append(x.shape);return values,None
        result=native_infer_score(SimpleNamespace(infer=fake_infer),values)
        self.assertIs(result,values);self.assertEqual(result.shape,(1,100,3));self.assertEqual(calls,[(1,100,3)])
        with self.assertRaises(ValueError):native_infer_score(SimpleNamespace(infer=lambda *a:(values,values)),values)
        with self.assertRaises(TypeError):native_infer_score(SimpleNamespace(infer=lambda *a:values),values)

    def test_profile_python_rng_binding_and_roundtrip(self):
        from hybrid_tsad import left_normal_public as module
        state=module.random.getstate()
        try:
            module.random.seed(1717);module.random.random()
        finally:module.random.setstate(state)
        self.assertEqual(module.random.getstate(),state)

    def test_no_unbound_globals_in_unexecuted_profile_or_data_paths(self):
        from hybrid_tsad import left_normal_public as module
        from hybrid_tsad import normal_run_data as data_module
        for m in [module,data_module]:
            table=symtable.symtable(Path(m.__file__).read_text(),m.__file__,'exec')
            bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
            allowed=bound|set(dir(builtins))|{'__file__','__name__','__package__','__builtins__'}
            missing=set()
            def walk(scope):
                for symbol in scope.get_symbols():
                    if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed:
                        missing.add(symbol.get_name())
                for child in scope.get_children():walk(child)
            walk(table)
            self.assertFalse(missing,(m.__name__,missing))


if __name__=='__main__':unittest.main()
