#!/usr/bin/env python3
"""Verify the finite configuration sweep used by the Colab benchmark."""
import importlib.util
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'colab_config_matrix', ROOT / 'scripts/benchmarks/colab_config_matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)
real_spec = importlib.util.spec_from_file_location(
    'colab_real_suite', ROOT / 'scripts/benchmarks/colab_real_suite.py')
real_suite = importlib.util.module_from_spec(real_spec)
real_spec.loader.exec_module(real_suite)


class ConfigurationMatrix(unittest.TestCase):
    def test_full_factorial_counts_and_unique_identities(self):
        for runtimes, expected in (([], 1536), (['vulkan'], 3072),
                                   (['vulkan', 'opencl'], 6144),
                                   (['cuda', 'vulkan', 'opencl'], 12288)):
            configs = list(matrix.full_factorial(runtimes))
            self.assertEqual(matrix.configuration_count(runtimes), expected)
            self.assertEqual(len(configs), expected)
            self.assertEqual(len({c['configuration_id'] for c in configs}), expected)

    def test_every_switch_has_both_states(self):
        configs = list(matrix.full_factorial(['vulkan']))
        for name in matrix.PLANNER_FLAGS:
            self.assertEqual({c['environment'][name] for c in configs}, {'0', '1'})
        self.assertEqual({c['environment']['MIMIR_PLANNER_MODE'] for c in configs},
                         set(matrix.PLANNER_MODES))
        self.assertEqual({c['environment']['OMP_WAIT_POLICY'] for c in configs},
                         set(matrix.WAIT_POLICIES))
        self.assertEqual({c['environment']['MIMIR_GPU_VULKAN'] for c in configs},
                         {'0', '1'})
        for runtime in ('CUDA', 'OPENCL'):
            name = 'MIMIR_GPU_' + runtime
            self.assertEqual({c['environment'][name] for c in configs}, {'0'})
        for config in configs:
            env = config['environment']
            for runtime in ('CUDA', 'VULKAN', 'OPENCL'):
                self.assertEqual(env['MIMIR_DISABLE_' + runtime],
                                 str(1 - int(env['MIMIR_GPU_' + runtime])))
            self.assertIn(config['configuration_id'], env['MIMIR_PLANNER_JSON'])
            options = config['mimir_options']
            self.assertEqual(options['planner_enabled'],
                             env['MIMIR_ENABLE_PLANNER'] == '1')
            self.assertEqual(options['planner_mode'], env['MIMIR_PLANNER_MODE'])
            for flag, option in matrix.PLANNER_OPTION_NAMES.items():
                self.assertEqual(options[option], env[flag] == '1')

    def test_invalid_runtime_selection_is_rejected(self):
        for runtimes in (['vulkan', 'vulkan'], ['rocm'], ['CUDA']):
            with self.assertRaises(ValueError):
                matrix.configuration_count(runtimes)
            with self.assertRaises(ValueError):
                matrix.recommended(runtimes)

    def test_recommended_matrix_is_compact_and_covers_each_control(self):
        for runtimes, expected in (([], 12), (['vulkan'], 13),
                                   (['cuda', 'vulkan'], 15),
                                   (['cuda', 'vulkan', 'opencl'], 16)):
            configs = matrix.recommended(runtimes)
            self.assertEqual(len(configs), expected)
            self.assertEqual(matrix.recommended_configuration_count(runtimes), expected)
            self.assertEqual(len({c['configuration_id'] for c in configs}), expected)

            baseline = next(c for c in configs
                            if c['mimir_options']['planner_enabled']
                            and c['mimir_options']['planner_mode'] == 'cost'
                            and c['environment']['OMP_WAIT_POLICY'] == 'active'
                            and all(c['environment'][flag] == '1'
                                    for flag in (matrix.GPU_FLAGS[r] for r in runtimes)))
            baseline_env = baseline['environment']
            self.assertEqual(baseline['mimir_options']['planner_dump'], False)
            self.assertEqual(baseline['mimir_options']['device_residency'], True)
            if runtimes:
                cpu_fallback = next(c for c in configs
                                    if all(c['environment'][flag] == '0'
                                           for flag in matrix.GPU_FLAGS.values()))
                self.assertTrue(cpu_fallback['mimir_options']['allow_cpu_fallback'])
                self.assertEqual(cpu_fallback['environment']['MIMIR_DISABLE_CPU'], '0')

            for config in configs:
                env = config['environment']
                for runtime in ('CUDA', 'VULKAN', 'OPENCL'):
                    self.assertEqual(env['MIMIR_DISABLE_' + runtime],
                                     str(1 - int(env['MIMIR_GPU_' + runtime])))
                self.assertIn(config['configuration_id'], env['MIMIR_PLANNER_JSON'])

            for flag, default in matrix.RECOMMENDED_FLAGS.items():
                variant = next(c for c in configs
                               if c['environment'][flag] == str(1 - default)
                               and c['mimir_options']['planner_mode'] == 'cost'
                               and c['mimir_options']['planner_enabled']
                               and c['environment']['OMP_WAIT_POLICY'] == 'active'
                               and all(c['environment'][gpu_flag] == baseline_env[gpu_flag]
                                       for gpu_flag in matrix.GPU_FLAGS.values()))
                changed = {key for key in baseline_env
                           if baseline_env[key] != variant['environment'][key]}
                self.assertEqual(changed, {flag, 'MIMIR_PLANNER_JSON'})

    def test_profile_selector_preserves_exhaustive_matrix(self):
        runtimes = ['cuda', 'vulkan']
        self.assertEqual(list(matrix.select('full', runtimes)),
                         list(matrix.full_factorial(runtimes)))
        self.assertEqual(matrix.select('recommended', runtimes), matrix.recommended(runtimes))
        with self.assertRaises(ValueError):
            matrix.select('unknown', runtimes)

    def test_notebook_expands_cases_and_records_resume_protocol(self):
        notebook = json.loads((ROOT / 'Colab-bench-multi-framework.ipynb').read_text())
        cell = next(c for c in notebook['cells']
                    if c.get('id') == 'colab-config-matrix-expansion')
        source = ''.join(cell['source'])
        ast.parse(source)
        self.assertIn("select(CONFIGURATION_PROFILE,MIMIR_GPU_RUNTIMES if BACKEND=='cuda' else [])", source)
        setup = next(''.join(c['source']) for c in notebook['cells']
                     if 'CONFIGURATION_PROFILE=' in ''.join(c.get('source', [])))
        self.assertIn("CONFIGURATION_PROFILE='recommended'", setup)
        self.assertIn("case['mimir_environment']=dict(configuration['environment'])", source)
        self.assertIn("resume_completed_trials=True", source)
        self.assertIn("architecture='MNIST MLP (shared existing workload)'", source)

    def test_real_suite_applies_mimir_environment_and_resumes_valid_trials(self):
        calls = []

        def fake_runner(command, environment, log_path, timeout_s, memory_mib):
            calls.append((command, environment.copy()))
            framework = 'mimir' if any(str(arg).endswith('run_mimir.sh') for arg in command) else 'pytorch'
            Path(command[-1]).write_text(json.dumps(dict(
                framework=framework, backend='cpu', case='mlp-config-test', status='measured')))
            Path(log_path).write_text('')
            return dict(budget_status='completed', process_wall_s=.1, peak_tree_rss_mib=1.)

        config = dict(case='mlp-config-test', backend='cpu', hidden=2, batch=2, lr=.1,
                      steps=1, probe_indices=[0], test_examples=2, mimir_environment={
                          'MIMIR_ENABLE_PLANNER': '0',
                          'MIMIR_PLANNER_MODE': 'legacy',
                          'MIMIR_PLANNER_JSON': '/tmp/plan.json',
                      })
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(real_suite, 'first_step_reference', return_value={}), \
                    patch.object(real_suite, 'validate_result'), \
                    patch.object(real_suite, 'constrained_run', side_effect=fake_runner):
                first = real_suite.run_suite(
                    directory, [config], ['mimir', 'pytorch'], {'mimir': 'python', 'pytorch': 'python'},
                    '/tmp/mimir', '/tmp/worker.py', dict(os.environ), [0], 1, 5, 512, .1, 7)
                self.assertEqual(len(first), 2)
                self.assertEqual(len(calls), 2)
                mimir_environment = next(env for command, env in calls
                                         if any(str(arg).endswith('run_mimir.sh') for arg in command))
                self.assertEqual(mimir_environment['MIMIR_ENABLE_PLANNER'], '0')
                self.assertEqual(mimir_environment['MIMIR_PLANNER_MODE'], 'legacy')
                self.assertTrue(mimir_environment['MIMIR_PLANNER_JSON'].endswith(
                    'planner-dumps/mlp-config-test.mimir.0.json'))
                resumed = real_suite.run_suite(
                    directory, [config], ['mimir', 'pytorch'], {'mimir': 'python', 'pytorch': 'python'},
                    '/tmp/mimir', '/tmp/worker.py', dict(os.environ), [0], 1, 5, 512, .1, 7)
                self.assertEqual(len(resumed), 2)
                self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
