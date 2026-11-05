"""Contract tests for the forward-only architecture benchmark matrix."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / 'scripts' / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
spec = importlib.util.spec_from_file_location('colab_architecture_suite', BENCHMARKS / 'colab_architecture_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class Workloads(unittest.TestCase):
    def test_all_required_workloads_have_valid_native_contracts(self):
        configs = suite.workload_configs()
        self.assertEqual({config['case'] for config in configs}, {
            'basic_mlp', 'vgg16', 'resnet', 'vae_conv', 'transformer_encoder',
            'transformer_causal', 'vit', 'unet',
        })
        for config in configs:
            suite._validate_config(config)
            self.assertEqual(config['benchmark_kind'], 'architecture')
            self.assertTrue(config['architecture'])
            self.assertTrue(config['model_config'])
            self.assertGreater(config['expected_output_size'], 0)
            self.assertGreater(config['iterations'], config['warmup'] - 100)
        encoder = next(c for c in configs if c['case'] == 'transformer_encoder')
        causal = next(c for c in configs if c['case'] == 'transformer_causal')
        vae = next(c for c in configs if c['case'] == 'vae_conv')
        self.assertEqual(encoder['architecture'], causal['architecture'])
        self.assertEqual(encoder['family'], 'transformer')
        self.assertEqual(causal['family'], 'transformer')
        self.assertEqual(encoder['input_kind'], 'tokens')
        self.assertFalse(encoder['model_config']['causal'])
        self.assertTrue(causal['model_config']['causal'])
        self.assertEqual(vae['expected_output_size'],
                         vae['model_config']['image_h'] * vae['model_config']['image_w'] *
                         vae['model_config']['image_c'])

    def test_image_and_token_dimensions_match_model_config(self):
        for config in suite.workload_configs():
            model = config['model_config']
            if config['case'] in ('vgg16', 'resnet', 'vae_conv', 'unet'):
                self.assertEqual(config['input_size'], model['image_h'] * model['image_w'] * model['image_c'])
            if config['case'].startswith('transformer_'):
                self.assertEqual(config['token_count'], model['seq_len'])
                self.assertEqual(config['vocab_size'], model['vocab_size'])
            if config['case'] == 'vit':
                self.assertEqual(config['input_size'], model['num_tokens'] * model['d_model'])


class Expansion(unittest.TestCase):
    def test_full_factorial_expansion_and_unique_identities(self):
        workload = suite.workload_configs()[:1]
        matrix = [
            dict(configuration_id='cfg-a', environment={'MIMIR_ENABLE_PLANNER': '0'},
                 mimir_options={'planner_enabled': False}),
            dict(configuration_id='cfg-b', environment={'MIMIR_ENABLE_PLANNER': '1'},
                 mimir_options={'planner_enabled': True}),
        ]
        jobs = suite.expand_matrix(workload, ('mimir', 'pytorch', 'tensorflow', 'jax'),
                                   ('cpu', 'cuda'), 3, matrix)
        self.assertEqual(len(jobs), 4 * 2 * 3 * 2)
        identities = [tuple(job['config'][key] for key in
                            ('case', 'framework', 'backend', 'trial', 'configuration_id'))
                      for job in jobs]
        self.assertEqual(len(identities), len(set(identities)))
        for job in jobs:
            config = job['config']
            self.assertEqual(config['benchmark_kind'], 'architecture')
            self.assertEqual(config['configuration_id'], job['configuration']['configuration_id'])
            if job['framework'] == 'mimir':
                self.assertEqual(config['mimir']['planner_enabled'],
                                 job['configuration']['mimir_options']['planner_enabled'])
            else:
                self.assertEqual(config['mimir']['planner_enabled'], suite.DEFAULT_MIMIR['planner_enabled'])

    def test_process_estimate_matches_expanded_work(self):
        workloads = suite.workload_configs()
        frameworks = ('mimir', 'pytorch', 'tensorflow', 'jax')
        backends = ('cpu', 'tpu')
        matrix = [dict(configuration_id='a'), dict(configuration_id='b')]
        estimate = suite.estimate_process_count(workloads, frameworks, backends, 2, matrix)
        self.assertEqual(estimate, len(workloads) * len(frameworks) * len(backends) * 2 * len(matrix))
        self.assertEqual(estimate, len(suite.expand_matrix(workloads, frameworks, backends, 2, matrix)))

    def test_rejects_duplicate_matrix_ids_and_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            suite.expand_matrix(workloads=suite.workload_configs()[:1],
                                matrix=[dict(configuration_id='same'), dict(configuration_id='same')])
        with self.assertRaisesRegex(ValueError, 'positive'):
            suite.expand_matrix(trials=0)

    def test_summary_ranks_only_architecture_rows_with_complete_shared_coverage(self):
        workload = [suite.workload_configs()[0]]
        records = []
        for framework, times in (('mimir', [4.0, 6.0]), ('pytorch', [2.0, 4.0])):
            for trial, elapsed in enumerate(times):
                records.append(dict(status='measured', case='basic_mlp', backend='cpu',
                    configuration_id='cfg-a', framework=framework, trial=trial,
                    forward_median_ms=elapsed, parameter_count=1000,
                    forward_samples_ms=[elapsed * .9, elapsed, elapsed * 1.1]))
        detail, scores = suite._summary(records, workload, ('mimir', 'pytorch'), ('cpu',), 2)
        self.assertEqual(len(detail), 2)
        self.assertTrue(all(row['status'] == 'ranked' for row in scores))
        self.assertAlmostEqual(next(row['parameter_count_ratio'] for row in detail), 1.0)
        self.assertTrue(all(row['forward_p95_ms'] is not None for row in detail))
        self.assertTrue(all(row['variability'] in {'high', 'insufficient trials'} for row in detail))
        self.assertTrue(any(row['variability'] == 'high' for row in detail))
        self.assertAlmostEqual(next(row['score_100'] for row in scores
                                    if row['framework'] == 'pytorch'), 100.0)
        detail, scores = suite._summary(records[:2], workload, ('mimir', 'pytorch'), ('cpu',), 2)
        self.assertTrue(all(row['score_100'] is None for row in scores))
        self.assertTrue(all(row['comparison_status'] == 'incomplete' for row in detail))

    def test_summary_suppresses_scores_when_model_sizes_are_not_comparable(self):
        workload = [suite.workload_configs()[0]]
        records = [dict(status='measured', case='basic_mlp', backend='cpu',
                        configuration_id='cfg-a', framework=framework, trial=0,
                        forward_median_ms=elapsed, forward_samples_ms=[elapsed],
                        parameter_count=count)
                   for framework, elapsed, count in (('mimir', 1.0, 1000),
                                                     ('pytorch', 1.1, 2000))]
        detail, scores = suite._summary(records, workload, ('mimir', 'pytorch'), ('cpu',), 1)
        self.assertTrue(all(row['score_100'] is None for row in scores))
        self.assertEqual({row['comparison_status'] for row in detail},
                         {'parameter-count mismatch'})

    def test_resume_cache_requires_matching_config_identity_and_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cached.json'
            path.write_text(json.dumps(dict(config_sha256='config', job_identity={'case': 'mlp'},
                                            execution_fingerprint='machine', status='measured')))
            self.assertTrue(suite._valid_cached(path, 'config', {'case': 'mlp'}, 'machine'))
            self.assertFalse(suite._valid_cached(path, 'config', {'case': 'mlp'}, 'other-machine'))
            self.assertFalse(suite._valid_cached(path, 'other-config', {'case': 'mlp'}, 'machine'))

    def test_planner_diagnostics_report_operation_and_runtime_mix(self):
        diagnostic = suite._planning_diagnostics(dict(case='basic_mlp',
            configuration_id='cfg-a', trial=0, status='measured', parameter_count=100,
            inference_plan={'planned': True, 'mode': 'cost', 'fusion_groups': [{}],
                'nodes': [{'type': 'Linear', 'runtime': 'CPU', 'skip': False,
                           'graph_fused': True, 'scratch_bytes': 8, 'bytes': 16},
                          {'type': 'GELU', 'runtime': 'CPU', 'skip': True,
                           'graph_fused': True, 'scratch_bytes': 0, 'bytes': 0}]},
            runtime_audit='runtime trace'))
        self.assertEqual(diagnostic['operations'], {'Linear': 1})
        self.assertEqual(diagnostic['runtimes'], {'CPU': 1})
        self.assertEqual(diagnostic['active_nodes'], 1)
        self.assertEqual(diagnostic['skipped_nodes'], 1)
        self.assertEqual(diagnostic['fusion_groups'], 1)
        self.assertEqual(diagnostic['scratch_bytes'], 8)


if __name__ == '__main__':
    unittest.main()
