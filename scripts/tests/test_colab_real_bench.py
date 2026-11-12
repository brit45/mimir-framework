#!/usr/bin/env python3
"""Focused integrity and resource-budget regression checks (no accelerator needed)."""
import importlib.util
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('colab_real_suite',ROOT/'scripts/benchmarks/colab_real_suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
try:
    import numpy as np
    import pandas
except ImportError:
    np=None


class Provenance(unittest.TestCase):
    @staticmethod
    def checks(source):
        body=ast.parse(source).body[0].body
        measured=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
                      and any(isinstance(target,ast.Name) and target.id=='REAL_RECORDS'
                              for target in node.targets))
        return compile(ast.Module(body=body[measured+1:],type_ignores=[]),'<real-benchmark-integrity>','exec')

    def test_notebook_and_generator_check_registered_helpers_and_real_mutations(self):
        notebook=json.loads((ROOT/'Colab-bench-multi-framework.ipynb').read_text())
        notebook_source=next(''.join(cell['source']) for cell in notebook['cells']
                             if 'REAL_RECORDS=REAL_SUITE.run_suite(' in ''.join(cell.get('source',[])))
        generator=ast.parse((ROOT/'scripts/colab/add_real_benchmark.py').read_text())
        generated_source=next(ast.literal_eval(node.value) for node in generator.body
                              if isinstance(node,ast.Assign)
                              and any(isinstance(target,ast.Name) and target.id=='MEASURE'
                                      for target in node.targets))
        mutations={
            'unchanged':None,
            'changed_matrix':'colab_config_matrix.py',
            'removed_matrix':'colab_config_matrix.py',
            'changed_worker':'colab_real_worker.py',
            'changed_architecture':'colab_architecture_worker.py',
            'added_real':'colab_real_added.py',
            'added_architecture':'colab_architecture_added.py',
            'changed_data':'Donnees modifiees',
            'unrelated_file':None,
        }
        for origin,source in [('notebook',notebook_source),('generator',generated_source)]:
            checks=self.checks(source)
            for mutation,error in mutations.items():
                with self.subTest(origin=origin,mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                    root=Path(temporary);real_source=root/'sources';real_source.mkdir()
                    for name in ['colab_real_suite.py','colab_real_worker.py','colab_real_model.cpp',
                                 'colab_architecture_suite.py','colab_architecture_worker.py','colab_config_matrix.py']:
                        (real_source/name).write_text('original fixture\n')
                    expected={p.name:suite.sha256(p) for p in real_source.iterdir()}
                    real_data=root/'data';real_data.mkdir()
                    for name in ['train.images.u8','train.labels.u8','weights-2-0.f32']:
                        (real_data/name).write_bytes(b'\x01\x02\x03\x04')
                    expected_data={p.name:suite.sha256(p) for p in real_data.iterdir()}
                    provenance={'sources':dict(expected)}
                    if mutation=='removed_matrix':(real_source/'colab_config_matrix.py').unlink()
                    elif mutation=='changed_data':(real_data/'train.images.u8').write_bytes(b'changed')
                    elif mutation=='unrelated_file':(real_source/'notes.txt').write_text('not a benchmark source')
                    elif error:(real_source/error).write_text('modified or newly added\n')
                    environment=dict(provenance=provenance,real_source=real_source,REAL_SUITE=suite,
                                     REAL_DATA=real_data,REAL_DATA_MANIFEST={'files':dict(expected_data)})
                    if error:
                        with self.assertRaisesRegex(AssertionError,error):exec(checks,environment)
                    else:exec(checks,environment)
                    self.assertEqual(provenance['sources'],expected)
                    self.assertEqual(environment['REAL_DATA_MANIFEST']['files'],expected_data)


class Budgets(unittest.TestCase):
    @unittest.skipIf(np is None,'requires numpy and pandas')
    def test_report_keeps_optimizer_cases_separate(self):
        notebook=json.loads((ROOT/'Colab-bench-multi-framework.ipynb').read_text())
        source=next(''.join(c['source']) for c in notebook['cells']
                    if 'valid.pivot(' in ''.join(c['source']))
        generator=ast.parse((ROOT/'scripts/colab/add_real_benchmark.py').read_text())
        report=next(n.value.value for n in generator.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='REPORT' for t in n.targets))
        self.assertEqual(source.strip(),report.strip())
        pivot=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Attribute) and n.func.attr=='pivot')
        valid=pandas.DataFrame([dict(batch=64,framework=f,case='batch-64-'+opt,
                                    train_examples_per_s=i+1)
                               for f in ('mimir','jax') for i,opt in enumerate(('sgd','adam','adamw'))])
        table=eval(compile(ast.Expression(body=pivot),'<report-pivot>','eval'),
                   dict(valid=valid,metric='train_examples_per_s'))
        self.assertEqual(table.shape,(3,2))
        self.assertEqual(set(table.index),{'batch-64-sgd','batch-64-adam','batch-64-adamw'})

    @unittest.skipUnless(sys.platform=='linux','requires Linux /proc and process groups')
    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            result=suite.constrained_run([sys.executable,'-c','import time; time.sleep(10)'],dict(os.environ),Path(d)/'log',.2,512)
            self.assertEqual(result['budget_status'],'time_limit')
            self.assertLess(result['process_wall_s'],5)

    @unittest.skipUnless(sys.platform=='linux','requires Linux /proc and process groups')
    def test_child_memory_counted(self):
        script="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; b=bytearray(96*1024*1024); time.sleep(10)']); time.sleep(10)"
        with tempfile.TemporaryDirectory() as d:
            result=suite.constrained_run([sys.executable,'-c',script],dict(os.environ),Path(d)/'log',5,48)
            self.assertEqual(result['budget_status'],'memory_limit')
            self.assertGreater(result['peak_tree_rss_mib'],48)


@unittest.skipIf(np is None,'numpy/pandas not installed')
class Ranking(unittest.TestCase):
    def test_incomplete_excluded_and_geometric_score(self):
        def record(framework,trial,train_s,infer_s):
            return dict(framework=framework,case='toy',trial=trial,status='ok',train_step_samples_ms=[1.,2.],
                train_wall_s=train_s,examples_trained=128,cold_step_ms=1.,worker_wall_s=2.,process_wall_s=2.1,
                peak_tree_rss_mib=10.,time_to_goal_s=None,mode='unit-test fixture',version='fixture',device='cpu',
                test=dict(samples_ms=[1.,2.],examples=64,accuracy=.9,mse=.1),inference_wall_s=infer_s)
        records=[record(f,t,train,infer) for f,train,infer in [('mimir',1.28,.64),('pytorch',.64,.16)] for t in (0,1)]
        records.append(dict(framework='jax',case='toy',trial=0,status='memory_limit',error='fixture limit'))
        detail,board=suite.summarize(records,[dict(case='toy',batch=64,hidden=4)],['mimir','pytorch','jax'],2)
        scores=board.set_index('framework')
        self.assertAlmostEqual(scores.loc['mimir','score_100'],100*np.sqrt(.5*.25))
        self.assertAlmostEqual(scores.loc['pytorch','score_100'],100)
        self.assertTrue(np.isnan(scores.loc['jax','score_100']))
        self.assertTrue(np.isnan(scores.loc['jax','rank']))
        self.assertTrue(detail[detail.framework=='mimir'].time_to_goal_s.isna().all())
        self.assertTrue(np.isnan(scores.loc['jax','lifecycle_score_100']))
        for row in records:
            if row['framework'] in ('mimir','pytorch'):
                row['process_wall_s']=4. if row['framework']=='mimir' else 2.
                row['peak_tree_rss_mib']=20. if row['framework']=='mimir' else 10.
        _,costs=suite.summarize(records,[dict(case='toy',batch=64,hidden=4)],['mimir','pytorch','jax'],2)
        costs=costs.set_index('framework')
        self.assertAlmostEqual(costs.loc['mimir','lifecycle_score_100'],50.)
        self.assertAlmostEqual(costs.loc['pytorch','lifecycle_score_100'],100.)

    def test_mimir_requires_the_requested_planner(self):
        config=dict(case='toy',mimir=dict(planner_mode='cost'))
        result=dict(case='toy',framework='mimir',status='measured',training_plan=dict(planned=False))
        with self.assertRaisesRegex(RuntimeError,'planner not active'):
            suite.validate_result(result,config,{},.8)
        result['training_plan']=dict(planned=True,mode='static',fusion=True)
        with self.assertRaisesRegex(RuntimeError,'planner not active'):
            suite.validate_result(result,config,{},.8)

    def test_quality_and_elapsed_times_cannot_bypass_gate(self):
        config=dict(case='toy',batch=2,steps=2,probe_indices=[0,1],test_examples=2)
        gold=dict(prediction=np.zeros(20),probe=np.zeros(2),loss=0.)
        measured=dict(case='toy',status='measured',first_prediction=[0.]*20,first_parameter_probe=[0.,0.],first_loss=0.,
            steps_completed=2,examples_trained=4,train_step_samples_ms=[1.,2.],train_losses=[0.,0.],
            test=dict(accuracy=.9,mse=.1,samples_ms=[1.],examples=2),final_parameter_probe=[0.,0.],
            train_wall_s=1.,inference_wall_s=1.,cold_step_ms=1.,cold_predict_ms=1.,worker_wall_s=2.)
        suite.validate_result(measured,config,gold,.8)
        measured['test']['accuracy']=.1
        with self.assertRaisesRegex(RuntimeError,'Quality gate'):suite.validate_result(measured,config,gold,.8)
        measured['test']['accuracy']=.9;measured['train_wall_s']=float('nan')
        with self.assertRaisesRegex(RuntimeError,'elapsed'):suite.validate_result(measured,config,gold,.8)


if __name__=='__main__':unittest.main()
