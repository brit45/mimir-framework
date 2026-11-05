#!/usr/bin/env python3
"""Forward-only fixed-input architecture comparisons for Mímir and tensor frameworks."""
import argparse
import csv
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name('colab_architecture_worker.py')
_worker_spec = importlib.util.spec_from_file_location('colab_architecture_worker', WORKER)
_worker_module = importlib.util.module_from_spec(_worker_spec)
_worker_spec.loader.exec_module(_worker_module)
_validate_config = _worker_module._validate_config
DEFAULT_FRAMEWORKS = ('mimir', 'pytorch', 'tensorflow', 'jax')
DEFAULT_BACKENDS = ('cpu',)
DEFAULT_MIMIR = {
    'planner_enabled': True,
    'planner_mode': 'cost',
    'cost_model': True,
    'fusion': True,
    'fusion_train': False,
    'buffer_reuse': True,
    'device_residency': True,
    'runtime_autotune': False,
    'planner_dump': False,
    'allow_cpu_fallback': False,
}


def _image_config(width, height, channels=3):
    return dict(image_w=width, image_h=height, image_c=channels)


def workload_configs(warmup=2, iterations=8):
    """Return small deterministic registry-aligned architecture configurations."""
    rows = [
        ('basic_mlp', 'basic_mlp', dict(input_dim=64, hidden_dim=32, output_dim=16,
          hidden_layers=2, dropout=0.0), 'float', 64, 0, 0, 16),
        ('vgg16', 'vgg16', dict(**_image_config(32, 32), base_channels=4,
          num_classes=10, fc_hidden=32, dropout=0.0), 'float', 32*32*3, 0, 0, 10),
        ('resnet', 'resnet', dict(**_image_config(32, 32), base_channels=4,
          num_classes=10, blocks1=1, blocks2=1, blocks3=1, blocks4=1),
          'float', 32*32*3, 0, 0, 10),
        ('vae_conv', 'vae_conv', dict(**_image_config(16, 16), latent_h=4, latent_w=4,
          latent_c=4, base_channels=4, stochastic_latent=False, latent_spatial_context=False,
          latent_detail_path=False, resnet=False, attention=False, text_cond=False,
          decoder_upsample='nearest_conv'),
          'float', 16*16*3, 0, 0, 16*16*3),
        ('transformer_encoder', 'transformer', dict(seq_len=8, d_model=16, vocab_size=64,
          padding_idx=0, num_layers=1, num_heads=4, mlp_hidden=32, output_dim=24,
          causal=False), 'tokens', 0, 8, 64, 24),
        ('transformer_causal', 'transformer', dict(seq_len=8, d_model=16, vocab_size=64,
          padding_idx=0, num_layers=1, num_heads=4, mlp_hidden=32, output_dim=24,
          causal=True), 'tokens', 0, 8, 64, 24),
        ('vit', 'vit', dict(num_tokens=8, d_model=16, num_layers=1, num_heads=4,
          mlp_hidden=32, output_dim=10, causal=False), 'float', 8*16, 0, 0, 10),
        ('unet', 'unet', dict(**_image_config(16, 16), base_channels=4, depth=2),
          'float', 16*16*3, 0, 0, 16*16*3),
    ]
    configs = []
    for case, architecture, model_config, input_kind, input_size, token_count, vocab_size, output_size in rows:
        config = dict(benchmark_kind='architecture', case=case,
                      family='transformer' if architecture == 'transformer' else case,
                      architecture=architecture, model_config=model_config,
                      input_kind=input_kind, input_size=input_size, token_count=token_count,
                      vocab_size=vocab_size, expected_output_size=output_size,
                      warmup=warmup, iterations=iterations, mimir=copy.deepcopy(DEFAULT_MIMIR))
        _validate_config(config)
        if case in ('transformer_encoder', 'transformer_causal'):
            if model_config['seq_len'] != token_count or model_config['vocab_size'] != vocab_size:
                raise ValueError(f'{case}: token dimensions do not match model_config')
        elif case == 'vit':
            if model_config['num_tokens'] * model_config['d_model'] != input_size:
                raise ValueError('ViT input dimensions do not match model_config')
        elif case in ('vgg16', 'resnet', 'vae_conv', 'unet'):
            if model_config['image_h'] * model_config['image_w'] * model_config['image_c'] != input_size:
                raise ValueError(f'{case}: image dimensions do not match input_size')
        configs.append(config)
    return configs


def _matrix_entries(matrix):
    if matrix is None:
        return [dict(configuration_id='default', environment={}, mimir_options=copy.deepcopy(DEFAULT_MIMIR))]
    entries = [copy.deepcopy(entry) for entry in matrix]
    if not entries:
        raise ValueError('The supplied configuration matrix must not be empty')
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each matrix entry must be an object')
        identity = entry.get('configuration_id')
        if not identity or identity in seen:
            raise ValueError('Matrix configuration_id values must be non-empty and unique')
        seen.add(identity)
        if not isinstance(entry.get('environment', {}), dict) or not isinstance(entry.get('mimir_options', {}), dict):
            raise ValueError('Matrix environment and mimir_options must be objects')
    return entries


def expand_matrix(workloads=None, frameworks=DEFAULT_FRAMEWORKS, backends=DEFAULT_BACKENDS,
                  trials=1, matrix=None):
    """Cartesian expansion; framework/backend pairs each run in a separate worker."""
    workloads = workload_configs() if workloads is None else workloads
    frameworks, backends = tuple(frameworks), tuple(backends)
    if not frameworks or set(frameworks) - set(DEFAULT_FRAMEWORKS):
        raise ValueError('Unsupported or empty framework list')
    if not backends or set(backends) - {'cpu', 'cuda', 'tpu'}:
        raise ValueError('Unsupported or empty backend list')
    if len(set(frameworks)) != len(frameworks) or len(set(backends)) != len(backends):
        raise ValueError('Framework and backend values must be unique')
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError('trials must be a positive integer')
    entries = _matrix_entries(matrix)
    jobs = []
    for workload in workloads:
        _validate_config(workload)
        for framework in frameworks:
            for backend in backends:
                for entry in entries:
                    for trial in range(trials):
                        config = copy.deepcopy(workload)
                        config['framework'] = framework
                        config['backend'] = backend
                        config['trial'] = trial
                        config['configuration_id'] = entry['configuration_id']
                        if framework == 'mimir':
                            config['mimir'].update(entry.get('mimir_options', {}))
                        config['job_identity'] = dict(case=config['case'], framework=framework,
                            backend=backend, trial=trial, configuration_id=entry['configuration_id'])
                        jobs.append(dict(config=config, framework=framework, backend=backend,
                            trial=trial, configuration=entry))
    return jobs


def estimate_process_count(workloads=None, frameworks=DEFAULT_FRAMEWORKS,
                           backends=DEFAULT_BACKENDS, trials=1, matrix=None):
    workloads = workload_configs() if workloads is None else workloads
    return len(workloads) * len(tuple(frameworks)) * len(tuple(backends)) * int(trials) * len(_matrix_entries(matrix))


def _digest(value):
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _host_metadata(base_environment, cpu_affinity):
    environment = dict(os.environ)
    environment.update({str(key): str(value) for key, value in base_environment.items()})
    cpu_model = platform.processor()
    if cpu_model.lower() in ('', 'unknown', 'inconnu') and Path('/proc/cpuinfo').is_file():
        for line in Path('/proc/cpuinfo').read_text(errors='replace').splitlines():
            if line.lower().startswith(('model name', 'hardware')):
                cpu_model = line.split(':', 1)[-1].strip()
                break
    try:
        memory_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, OSError, ValueError):
        memory_bytes = None
    environment_keys = (
        'OMP_NUM_THREADS', 'OMP_WAIT_POLICY', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'JAX_PLATFORMS', 'PJRT_DEVICE',
        'CUDA_VISIBLE_DEVICES', 'MIMIR_TPU', 'MIMIR_TPU_TARGET',
    )
    mimir_environment_keys = {key for key in environment
        if (key.startswith(('MIMIR_GPU_', 'MIMIR_DISABLE_', 'MIMIR_PLANNER_', 'MIMIR_ENABLE_'))
            and key != 'MIMIR_PLANNER_JSON')}
    effective_environment = {key: environment[key] for key in
        set(environment_keys) | mimir_environment_keys if key in environment}
    accelerator = dict(nvidia_smi=None,
        tpu_type=environment.get('TPU_ACCELERATOR_TYPE'),
        tpu_environment_present=any(environment.get(key) for key in
            ('COLAB_TPU_ADDR', 'TPU_NAME', 'TPU_WORKER_ID', 'TPU_ACCELERATOR_TYPE')))
    nvidia_smi = shutil.which('nvidia-smi')
    if nvidia_smi:
        probe = subprocess.run([nvidia_smi,
            '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader'],
            text=True, capture_output=True, timeout=15, check=False)
        accelerator['nvidia_smi'] = dict(returncode=probe.returncode,
            devices=probe.stdout.strip().splitlines() if probe.returncode == 0 else [],
            error=probe.stderr.strip() if probe.returncode else None)
    return dict(platform=platform.platform(), machine=platform.machine(),
                processor=cpu_model, logical_cpus=os.cpu_count(),
                affinity=list(cpu_affinity) if cpu_affinity else None,
                memory_bytes=memory_bytes, python=sys.version, accelerator=accelerator,
                environment=effective_environment)


def _python_metadata(executable, framework):
    package = {'pytorch': 'torch', 'tensorflow': 'tensorflow', 'jax': 'jax'}[framework]
    script = (
        'import importlib.metadata as m,json,sys; '
        f'package={package!r}; '
        'version=None; '
        'exec("try:\\n version=m.version(package)\\nexcept m.PackageNotFoundError:\\n pass"); '
        'print(json.dumps({"python":sys.version,"package":version}))'
    )
    completed = subprocess.run([str(executable), '-c', script], text=True,
                               capture_output=True, timeout=30, check=False)
    if completed.returncode:
        raise RuntimeError(f'Cannot inspect {framework} interpreter {executable}: '
                           f'{completed.stderr.strip()}')
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise RuntimeError(f'Invalid interpreter metadata from {executable}') from exc


def _execution_metadata(frameworks, mimir_executable, python_executables,
                        base_environment, cpu_affinity):
    host = _host_metadata(base_environment or {}, cpu_affinity)
    executables = {}
    for framework in frameworks:
        if framework == 'mimir':
            if not mimir_executable:
                raise ValueError('A Mímir native executable is required for Mímir cases')
            path = Path(mimir_executable).resolve()
            if not path.is_file():
                raise ValueError(f'Mímir native executable does not exist: {path}')
            stat = path.stat()
            executables[framework] = dict(path=str(path), size_bytes=stat.st_size,
                                          sha256=_file_sha256(path))
        else:
            path = Path((python_executables or {}).get(framework, sys.executable)).resolve()
            if not path.is_file():
                raise ValueError(f'{framework} Python executable does not exist: {path}')
            stat = path.stat()
            executables[framework] = dict(path=str(path), size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, **_python_metadata(path, framework))
    native_source = ROOT / 'scripts/benchmarks/colab_real_model.cpp'
    model_source = ROOT / 'src/Model.cpp'
    sources = {path.name: _file_sha256(path) for path in
               (Path(__file__), WORKER, native_source, model_source)}
    version_path = ROOT / 'VERSION'
    mimir_version = version_path.read_text().strip() if version_path.is_file() else None
    return dict(host=host, executables=executables, sources=sources,
                mimir_version=mimir_version,
                fingerprint=_digest(dict(host=host, executables=executables, sources=sources,
                                         mimir_version=mimir_version)))


def _constrained_run():
    spec = importlib.util.spec_from_file_location('colab_real_suite', ROOT / 'scripts/benchmarks/colab_real_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.constrained_run


def _safe(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value)).strip('-') or 'case'


def _result_path(output_dir, job):
    config = job['config']
    identity = _digest(config['job_identity'])[:10]
    name = '-'.join((_safe(config['case']), _safe(config['framework']),
                     _safe(config['backend']), f't{config["trial"]}',
                     _safe(config['configuration_id']), identity)) + '.json'
    return output_dir / 'results' / name


def _valid_cached(path, config_hash, identity, execution_fingerprint):
    try:
        result = json.loads(path.read_text())
        return (result.get('config_sha256') == config_hash and
                result.get('job_identity') == identity and result.get('status') == 'measured' and
                result.get('execution_fingerprint') == execution_fingerprint)
    except (OSError, ValueError, TypeError):
        return False


def _validate_mimir(result, config, backend, configuration):
    if result.get('output_size') != config['expected_output_size']:
        raise RuntimeError('Mímir output size differs from the workload contract')
    values = result.get('first_output')
    if not isinstance(values, list) or len(values) != config['expected_output_size']:
        raise RuntimeError('Mímir did not return the full fixed-input output')
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise RuntimeError('Mímir produced non-finite output')
    plan = result.get('inference_plan')
    nodes = plan.get('nodes', []) if isinstance(plan, dict) else []
    options = config['mimir']
    if options.get('planner_enabled', True):
        effective_mode = ('cost' if options.get('planner_mode', 'cost') == 'cost' and
                          options.get('cost_model', True) else
                          'static' if options.get('planner_mode', 'cost') == 'static' else 'legacy')
        if not isinstance(plan, dict) or not plan.get('planned') or plan.get('mode') != effective_mode:
            raise RuntimeError('Mímir planner mode differs from the requested architecture configuration')
        if plan.get('fusion') != options.get('fusion', True):
            raise RuntimeError('Mímir planner fusion differs from the requested architecture configuration')
    elif isinstance(plan, dict) and plan.get('planned'):
        raise RuntimeError('Mímir planner is active although disabled in the requested architecture configuration')
    if backend != 'cpu':
        if not nodes:
            raise RuntimeError('Mímir plan exposes no device assignments; placement cannot be verified')
        runtimes = {str(node.get('runtime', '')).lower() for node in nodes if isinstance(node, dict)}
        enabled = configuration.get('environment', {})
        aliases = {'cuda': ('cuda', 'nvidia', 'vulkan', 'opencl'),
                   'tpu': ('tpu',)}[backend]
        if backend == 'cuda':
            active = {name.lower() for name, value in enabled.items()
                      if name in ('MIMIR_GPU_CUDA', 'MIMIR_GPU_VULKAN', 'MIMIR_GPU_OPENCL')
                      and str(value) == '1'}
            aliases = tuple(runtime for runtime in aliases if runtime in active)
            if not aliases:
                raise RuntimeError('No Mímir accelerator runtime is enabled for this configuration')
        if not any(any(alias in runtime for alias in aliases) for runtime in runtimes):
            raise RuntimeError(f'Mímir plan did not prove placement on requested {backend} device')
        compute_nodes = [node for node in nodes if isinstance(node, dict) and
                         str(node.get('type', '')).lower() != 'identity']
        compute_runtimes = {str(node.get('runtime', '')).lower() for node in compute_nodes}
        if not options.get('allow_cpu_fallback', False) and any(
                runtime in ('cpu', 'portable', 'dispatcher (legacy)') for runtime in compute_runtimes):
            raise RuntimeError('Mímir plan contains a CPU fallback despite fallback being disabled')
    elif result.get('backend') != 'cpu':
        raise RuntimeError('Mímir result backend identity mismatch')
    elif nodes and any(str(node.get('runtime', '')).lower() not in ('cpu', 'native')
                       for node in nodes if isinstance(node, dict)):
        raise RuntimeError('Mímir CPU workload was routed to a non-CPU runtime')


def _validate_worker_result(result, job):
    config = job['config']
    if result.get('status') != 'measured':
        raise RuntimeError(result.get('error', 'Worker failed') + '\n' + result.get('traceback', ''))
    for key in ('framework', 'backend', 'case'):
        expected = config[key] if key in config else config['case'] if key == 'case' else None
        if key == 'case':
            expected = config['case']
        if result.get(key) != expected:
            raise RuntimeError(f'Worker {key} identity mismatch')
    if result.get('seed') != config.get('seed', 12031):
        raise RuntimeError('Worker seed differs from the requested deterministic seed')
    if job['framework'] == 'mimir':
        _validate_mimir(result, config, job['backend'], job['configuration'])
    else:
        if result.get('output_size') != config['expected_output_size']:
            raise RuntimeError('Worker output size differs from workload contract')
        values = result.get('first_output')
        if not isinstance(values, list) or len(values) != config['expected_output_size']:
            raise RuntimeError('Worker did not return the full fixed-input output')
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise RuntimeError('Worker produced non-finite output')
    samples = result.get('forward_samples_ms')
    if not isinstance(samples, list) or len(samples) != config['iterations']:
        raise RuntimeError('Worker did not return the requested number of latency samples')
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
               for value in samples):
        raise RuntimeError('Worker returned invalid latency samples')
    expected_median = statistics.median(samples)
    if not math.isclose(result.get('forward_median_ms', math.nan), expected_median,
                        rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError('Worker latency median does not match its raw samples')
    if not isinstance(result.get('parameter_count'), int) or result['parameter_count'] <= 0:
        raise RuntimeError('Worker did not report a positive parameter count')
    if not math.isfinite(result.get('setup_s', math.nan)) or result['setup_s'] < 0:
        raise RuntimeError('Worker returned an invalid setup duration')
    if not math.isfinite(result.get('cold_forward_ms', math.nan)) or result['cold_forward_ms'] <= 0:
        raise RuntimeError('Worker returned an invalid cold-forward duration')


def _percentile(values, percentile):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _trial_metrics(records):
    rows = []
    for result in records:
        samples = result.get('forward_samples_ms', [])
        row = dict(framework=result.get('framework'), backend=result.get('backend'),
                   case=result.get('case'), configuration_id=result.get('configuration_id'),
                   trial=result.get('trial'), status=result.get('status'),
                   error=result.get('error'), device=result.get('device'),
                   framework_version=result.get('version'),
                   parameter_count=result.get('parameter_count'),
                   setup_s=result.get('setup_s', result.get('initialization_s')),
                   cold_forward_ms=result.get('cold_forward_ms'),
                   forward_median_ms=result.get('forward_median_ms'),
                   forward_p95_ms=result.get('forward_p95_ms'),
                   min_ms=min(samples) if samples else None,
                   max_ms=max(samples) if samples else None,
                   process_wall_s=result.get('process_budget', {}).get('process_wall_s'),
                   peak_process_rss_mib=result.get('process_budget', {}).get('peak_tree_rss_mib'),
                   peak_process_rss_plus_swap_mib=result.get('process_budget', {}).get('peak_tree_rss_swap_mib'),
                   configuration_sha256=result.get('config_sha256'),
                   execution_fingerprint=result.get('execution_fingerprint'))
        rows.append(row)
    return rows


def _planning_diagnostics(result):
    plan = result.get('inference_plan')
    nodes = plan.get('nodes', []) if isinstance(plan, dict) else []
    active = [node for node in nodes if isinstance(node, dict) and not node.get('skip', False)]
    def counts(key):
        output = {}
        for node in active:
            name = str(node.get(key, 'unknown'))
            output[name] = output.get(name, 0) + 1
        return dict(sorted(output.items()))
    return dict(case=result.get('case'), configuration_id=result.get('configuration_id'),
                trial=result.get('trial'), status=result.get('status'),
                planned=plan.get('planned') if isinstance(plan, dict) else None,
                planner_mode=plan.get('mode') if isinstance(plan, dict) else None,
                total_nodes=len(nodes), active_nodes=len(active), skipped_nodes=len(nodes)-len(active),
                operations=counts('type'), runtimes=counts('runtime'),
                fusion_groups=len(plan.get('fusion_groups', [])) if isinstance(plan, dict) else 0,
                graph_fused_nodes=sum(bool(node.get('graph_fused')) for node in active),
                parameter_count=result.get('parameter_count'),
                scratch_bytes=sum(int(node.get('scratch_bytes', 0) or 0) for node in active),
                planned_tensor_bytes=sum(int(node.get('bytes', 0) or 0) for node in active),
                layer_profile=result.get('layer_profile', []),
                layer_profile_scope=result.get('layer_profile_scope'),
                runtime_trace=result.get('runtime_audit', ''))


def _summary(records, workloads, frameworks, backends, trials):
    grouped = {}
    for record in records:
        key = (record.get('case'), record.get('backend', 'cpu'),
               record.get('configuration_id', 'default'))
        grouped.setdefault(key, {framework: [] for framework in frameworks})
        if record.get('framework') in frameworks:
            grouped[key][record['framework']].append(record)
    detail, score_values = [], {framework: [] for framework in frameworks}
    for (case, backend, configuration_id), framework_times in grouped.items():
        medians = {}
        metrics = {}
        for framework, rows in framework_times.items():
            trial_ids = [row.get('trial') for row in rows]
            measured = [row for row in rows if row.get('status') == 'measured']
            unique_trials = set(trial_ids) == set(range(trials)) and len(trial_ids) == trials
            values = [row['forward_median_ms'] for row in measured
                      if isinstance(row.get('forward_median_ms'), (int, float)) and
                      math.isfinite(row['forward_median_ms']) and row['forward_median_ms'] > 0]
            complete = unique_trials and len(values) == trials
            samples = [value for row in measured for value in row.get('forward_samples_ms', [])
                       if isinstance(value, (int, float)) and math.isfinite(value) and value > 0]
            if complete:
                medians[framework] = statistics.median(values)
            mean = statistics.mean(values) if values else None
            cv = (statistics.stdev(values) / mean if len(values) >= 2 and mean else None)
            metrics[framework] = dict(p95_ms=_percentile(samples, .95) if samples else None,
                min_ms=min(samples) if samples else None, max_ms=max(samples) if samples else None,
                trial_medians_ms=values, trial_cv=cv,
                parameter_count=statistics.median([row['parameter_count'] for row in measured
                    if isinstance(row.get('parameter_count'), int)]) if measured else None,
                peak_process_rss_mib=max([row.get('process_budget', {}).get('peak_tree_rss_mib', 0)
                    for row in measured] or [0]),
                peak_process_rss_plus_swap_mib=max([
                    row.get('process_budget', {}).get('peak_tree_rss_swap_mib', 0)
                    for row in measured] or [0]))
        common = len(frameworks) >= 2 and len(medians) == len(frameworks)
        parameter_counts = [metrics[framework]['parameter_count'] for framework in frameworks
                            if metrics[framework]['parameter_count'] is not None]
        parameter_ratio = (max(parameter_counts) / min(parameter_counts)
                           if parameter_counts and min(parameter_counts) > 0 else None)
        parameter_parity = (len(parameter_counts) == len(frameworks) and
                            parameter_ratio is not None and parameter_ratio <= 1.05)
        common = common and parameter_parity
        best = min(medians.values()) if medians else None
        for framework in frameworks:
            elapsed = medians.get(framework)
            speedup = best / elapsed if common and elapsed and best else None
            stats = metrics[framework]
            cv = stats['trial_cv']
            row = dict(case=case, backend=backend, configuration_id=configuration_id,
                       framework=framework, forward_median_ms=elapsed,
                       forward_p95_ms=stats['p95_ms'], min_ms=stats['min_ms'],
                       max_ms=stats['max_ms'], trial_medians_ms=stats['trial_medians_ms'],
                       trial_cv=cv, variability='high' if cv is not None and cv >= .10 else
                       ('insufficient trials' if len(stats['trial_medians_ms']) < 3 else 'stable'),
                       parameter_count=stats['parameter_count'],
                       parameter_count_ratio=parameter_ratio,
                       comparison_status=('comparable' if speedup is not None else
                           'parameter-count mismatch' if len(medians) == len(frameworks) and
                           not parameter_parity else 'incomplete'),
                       peak_process_rss_mib=stats['peak_process_rss_mib'],
                       peak_process_rss_plus_swap_mib=stats['peak_process_rss_plus_swap_mib'],
                       speedup_vs_fastest=speedup, trials_ok=len(stats['trial_medians_ms']),
                       trials_requested=trials, status='measured' if elapsed is not None else 'incomplete')
            detail.append(row)
            if speedup is not None:
                score_values[framework].append(speedup)
    scores = []
    for framework in frameworks:
        values = score_values[framework]
        score = 100 * math.exp(sum(math.log(x) for x in values) / len(values)) if values else None
        scores.append(dict(framework=framework, score_100=score,
                           coverage=f'{len(values)}/{len(grouped)}',
                           status='ranked' if values else 'incomplete (no complete shared comparisons)',
                           metric='geometric mean forward-inference latency vs fastest per architecture'))
    if any(row['score_100'] is None for row in scores):
        for row in scores:
            row['score_100'] = None
            row['status'] = 'incomplete (shared coverage required for every framework)'
    ranked = [row for row in scores if row['score_100'] is not None]
    ranked.sort(key=lambda row: row['score_100'], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row['rank'] = rank
    for row in scores:
        row.setdefault('rank', None)
    return detail, scores


def run_suite(output_dir, mimir_executable, frameworks=DEFAULT_FRAMEWORKS,
              backends=DEFAULT_BACKENDS, trials=1, matrix=None, seed=20261105,
              timeout_s=600, memory_mib=8192, resume=False, workloads=None,
              python_executables=None, base_environment=None, cpu_affinity=None):
    output_dir = Path(output_dir)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    workloads = workload_configs() if workloads is None else workloads
    jobs = expand_matrix(workloads, frameworks, backends, trials, matrix)
    execution = _execution_metadata(frameworks, mimir_executable, python_executables,
                                    base_environment or {}, cpu_affinity)
    configurations = _matrix_entries(matrix)
    rng = random.Random(seed)
    rng.shuffle(jobs)
    runner = _constrained_run()
    records = []
    for job_index, job in enumerate(jobs, 1):
        config = job['config']
        config['seed'] = seed + job['trial']
        result_path = _result_path(output_dir, job)
        config_path = result_path.with_suffix('.config.json')
        execution_fingerprint = _digest(dict(execution=execution['fingerprint'],
            framework=job['framework'], configuration=job['configuration']))
        config_hash = _digest(dict(config=config, execution_fingerprint=execution_fingerprint))
        if job_index == 1 or job_index % 50 == 0:
            print(f'Architecture benchmark [{job_index}/{len(jobs)}] '
                  f'{config["case"]}/{job["framework"]}/{job["backend"]}/'
                  f'{config["configuration_id"]}/trial-{job["trial"] + 1}', flush=True)
        if resume and _valid_cached(result_path, config_hash, config['job_identity'],
                                    execution_fingerprint):
            cached = json.loads(result_path.read_text())
            try:
                _validate_worker_result(cached, job)
                if cached.get('execution_fingerprint') != execution_fingerprint:
                    raise RuntimeError('Cached result execution fingerprint changed')
                records.append(cached)
                continue
            except RuntimeError:
                pass
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False))
        if job['framework'] == 'mimir':
            if not mimir_executable:
                raise ValueError('A Mímir native executable is required for Mímir cases')
            command = ['bash', str(ROOT / 'run_mimir.sh'), job['backend'],
                       str(config_path), str(result_path)]
        else:
            executable=(python_executables or {}).get(job['framework'], sys.executable)
            command = [str(executable), str(WORKER), job['framework'], job['backend'],
                       str(config_path), str(result_path)]
        if cpu_affinity:
            command = ['taskset', '-c', ','.join(map(str, cpu_affinity)), *command]
        environment = dict(os.environ)
        if base_environment:
            environment.update({str(k): str(v) for k, v in base_environment.items()})
        if job['framework'] == 'mimir':
            environment['MIMIR_EXECUTABLE'] = str(Path(mimir_executable).resolve())
            if cpu_affinity:
                environment['GOMP_CPU_AFFINITY'] = ' '.join(map(str, cpu_affinity))
            environment.update({str(k): str(v) for k, v in job['configuration'].get('environment', {}).items()})
            options = config['mimir']
            if job['backend'] == 'cpu':
                environment.update(MIMIR_GPU_CUDA='0', MIMIR_GPU_VULKAN='0',
                                   MIMIR_GPU_OPENCL='0')
            elif job['backend'] == 'cuda' and not job['configuration'].get('environment'):
                environment.update(MIMIR_GPU_CUDA='1', MIMIR_GPU_VULKAN='0',
                                   MIMIR_GPU_OPENCL='0')
            environment.update(
                MIMIR_ENABLE_PLANNER='1' if options.get('planner_enabled', True) else '0',
                MIMIR_PLANNER_MODE=options.get('planner_mode', 'cost'),
                MIMIR_DISABLE_CPU='0' if job['backend'] == 'cpu' or options.get('allow_cpu_fallback', False) else '1',
                MIMIR_PLANNER_JSON=str(result_path.with_suffix('.planner.json')),
            )
        budget = runner(command, environment, result_path.with_suffix('.log'),
                        timeout_s, memory_mib, cwd=ROOT)
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError):
            result = dict(status='error', framework=job['framework'], backend=job['backend'],
                          case=config['case'], error='Worker produced no valid result JSON')
        result.update(job_identity=config['job_identity'], config_sha256=config_hash,
                      configuration_id=config['configuration_id'], trial=job['trial'],
                      seed=config['seed'], execution_fingerprint=execution_fingerprint,
                      process_budget=budget, claim='fixed-input forward inference only')
        if budget['budget_status'] != 'completed':
            result['status'] = budget['budget_status']
            result.setdefault('error', budget['budget_status'])
        if result.get('status') == 'measured':
            try:
                _validate_worker_result(result, job)
            except RuntimeError as exc:
                result.update(status='error', error=str(exc))
        if result.get('status') != 'measured':
            print(f'Architecture benchmark failed: {config["case"]}/{job["framework"]}/'
                  f'{job["backend"]}/{config["configuration_id"]}: '
                  f'{result.get("error", result["status"])}', flush=True)
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        config_path.unlink(missing_ok=True)
        records.append(result)
    detail, scores = _summary(records, workloads, frameworks, backends, trials)
    planning = [_planning_diagnostics(record) for record in records
                if record.get('framework') == 'mimir']
    generated_at = datetime.now(timezone.utc).isoformat()
    protocol = dict(schema_version='1.0', generated_at=generated_at,
        run_id=generated_at.replace(':', '').replace('-', '') + '-' + execution['fingerprint'][:12],
        benchmark_kind='architecture', claim='fixed-input forward inference only; not a training benchmark',
        execution=execution, seed=seed, trials=trials, process_count=len(jobs),
        timeout_s=timeout_s, memory_limit_mib=memory_mib, backends=list(backends),
        frameworks=list(frameworks), configuration_matrix=configurations,
        workloads=[dict(case=w['case'], architecture=w['architecture'],
                        model_config=w['model_config'], input_kind=w['input_kind'],
                        input_size=w['input_size'], token_count=w['token_count'],
                        expected_output_size=w['expected_output_size'],
                        warmup=w['warmup'], iterations=w['iterations'])
                   for w in workloads],
        timing='cold_forward_ms is first synchronized forward; forward_samples_ms are post-warmup synchronized forwards',
        score='geometric mean of fastest-relative median latency; generated only with complete shared coverage',
        variability_warning_threshold_cv=0.10)
    report = dict(protocol=protocol, process_count=len(jobs), seed=seed,
                  backends=list(backends), frameworks=list(frameworks),
                  workloads=[dict(case=w['case'], architecture=w['architecture'], model_config=w['model_config'],
                                  input_kind=w['input_kind'], input_size=w['input_size'],
                                  token_count=w['token_count'], expected_output_size=w['expected_output_size'])
                             for w in workloads], details=detail, score_summary=scores,
                  planning_diagnostics=planning, results=records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'architecture_benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False))
    with (output_dir / 'architecture-trials.jsonl').open('w') as destination:
        for record in records:
            destination.write(json.dumps(record, allow_nan=False) + '\n')
    _write_csv(output_dir / 'architecture-details.csv', detail,
        ('case', 'backend', 'configuration_id', 'framework', 'status', 'trials_ok',
         'trials_requested', 'parameter_count', 'forward_median_ms', 'forward_p95_ms',
         'parameter_count_ratio', 'comparison_status', 'min_ms', 'max_ms', 'trial_cv', 'variability',
         'peak_process_rss_mib', 'peak_process_rss_plus_swap_mib',
         'speedup_vs_fastest'))
    _write_csv(output_dir / 'architecture-trials.csv', _trial_metrics(records),
        ('framework', 'backend', 'case', 'configuration_id', 'trial', 'status', 'error',
         'device', 'framework_version', 'parameter_count', 'setup_s', 'cold_forward_ms',
         'forward_median_ms', 'forward_p95_ms', 'min_ms', 'max_ms',
         'process_wall_s', 'peak_process_rss_mib', 'peak_process_rss_plus_swap_mib',
         'configuration_sha256', 'execution_fingerprint'))
    (output_dir / 'planning-diagnostics.json').write_text(json.dumps(planning, indent=2, allow_nan=False))
    layer_rows = []
    for diagnostic in planning:
        for layer in diagnostic['layer_profile']:
            layer_rows.append(dict(case=diagnostic['case'],
                configuration_id=diagnostic['configuration_id'], trial=diagnostic['trial'],
                **layer))
    _write_csv(output_dir / 'architecture-layer-profile.csv', layer_rows,
        ('case', 'configuration_id', 'trial', 'layer', 'name', 'type', 'runtime',
         'call', 'elapsed_ms', 'output_size'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimir-executable', required=True, help='External colab_real_model native executable')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frameworks', nargs='+', default=list(DEFAULT_FRAMEWORKS))
    parser.add_argument('--backends', nargs='+', default=list(DEFAULT_BACKENDS))
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--matrix-json', help='JSON array of full-factorial configuration entries')
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--memory-mib', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    matrix = json.loads(Path(args.matrix_json).read_text()) if args.matrix_json else None
    report = run_suite(args.output_dir, args.mimir_executable, args.frameworks, args.backends,
                       args.trials, matrix, args.seed, args.timeout_s, args.memory_mib, args.resume)
    print(json.dumps(dict(process_count=report['process_count'], output=str(Path(args.output_dir) / 'architecture_benchmark.json'),
                          scores=report['score_summary']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
