"""Curated and full-factorial sweeps of discrete run_mimir.sh controls."""
from itertools import product
import hashlib
import json


PLANNER_MODES = ('legacy', 'static', 'cost')
PLANNER_FLAGS = (
    'MIMIR_ENABLE_PLANNER',
    'MIMIR_PLANNER_BUFFER_REUSE',
    'MIMIR_PLANNER_FUSION',
    'MIMIR_ENABLE_FUSION_TRAIN',
    'MIMIR_RUNTIME_AUTOTUNE',
    'MIMIR_PLANNER_COST_MODEL',
    'MIMIR_PLANNER_DUMP',
    'MIMIR_PLANNER_DEVICE_RESIDENCY',
)
GPU_FLAGS = {
    'cuda': 'MIMIR_GPU_CUDA',
    'vulkan': 'MIMIR_GPU_VULKAN',
    'opencl': 'MIMIR_GPU_OPENCL',
}
PLANNER_OPTION_NAMES = {
    'MIMIR_PLANNER_BUFFER_REUSE': 'buffer_reuse',
    'MIMIR_PLANNER_FUSION': 'fusion',
    'MIMIR_ENABLE_FUSION_TRAIN': 'fusion_train',
    'MIMIR_RUNTIME_AUTOTUNE': 'runtime_autotune',
    'MIMIR_PLANNER_COST_MODEL': 'cost_model',
    'MIMIR_PLANNER_DUMP': 'planner_dump',
    'MIMIR_PLANNER_DEVICE_RESIDENCY': 'device_residency',
}
WAIT_POLICIES = ('active', 'passive')
RECOMMENDED_FLAGS = {
    'MIMIR_PLANNER_BUFFER_REUSE': 1,
    'MIMIR_PLANNER_FUSION': 1,
    'MIMIR_ENABLE_FUSION_TRAIN': 1,
    'MIMIR_RUNTIME_AUTOTUNE': 1,
    'MIMIR_PLANNER_COST_MODEL': 1,
    'MIMIR_PLANNER_DUMP': 0,
    'MIMIR_PLANNER_DEVICE_RESIDENCY': 1,
}


def full_factorial(validated_runtimes):
    """Yield every meaningful discrete combination for the validated host.

    Runtime flags for unavailable backends are held off: enabling a runtime that
    was not validated would turn a benchmark result into an initialization test.
    Continuous/hardware-specific settings such as thread count and CPU affinity
    are fixed by the notebook's common host protocol, not multiplied here.
    """
    runtimes = tuple(validated_runtimes)
    if len(set(runtimes)) != len(runtimes) or any(runtime not in GPU_FLAGS for runtime in runtimes):
        raise ValueError('Expected unique validated runtimes from cuda, vulkan, opencl')

    dimensions = (
        tuple(product((0, 1), repeat=len(runtimes))),
        (0, 1),  # planner enabled
        PLANNER_MODES,
        *((0, 1) for _ in PLANNER_FLAGS[1:]),
        WAIT_POLICIES,
    )
    for values in product(*dimensions):
        runtime_values, planner_enabled, mode, *remaining = values
        flags = dict(zip(PLANNER_FLAGS[1:], remaining[:-1]))
        wait_policy = remaining[-1]
        enabled_runtimes = [runtime for runtime, enabled in zip(runtimes, runtime_values)
                            if enabled]
        yield _configuration(enabled_runtimes, planner_enabled, mode,
                             flags, wait_policy)


def _validated_runtimes(validated_runtimes):
    runtimes = tuple(validated_runtimes)
    if len(set(runtimes)) != len(runtimes) or any(runtime not in GPU_FLAGS for runtime in runtimes):
        raise ValueError('Expected unique validated runtimes from cuda, vulkan, opencl')
    return runtimes


def _configuration(enabled_runtimes, planner_enabled, mode, flags, wait_policy,
                   allow_cpu_fallback=None):
    environment = {name: '0' for name in GPU_FLAGS.values()}
    environment.update({GPU_FLAGS[runtime]: '1' for runtime in enabled_runtimes})
    for runtime, flag in GPU_FLAGS.items():
        environment['MIMIR_DISABLE_' + runtime.upper()] = str(
            1 - int(environment[flag]))
    environment.update({name: str(value) for name, value in flags.items()})
    environment.update(
        MIMIR_ENABLE_PLANNER=str(int(planner_enabled)),
        MIMIR_PLANNER_MODE=mode,
        OMP_WAIT_POLICY=wait_policy,
    )
    if allow_cpu_fallback is not None:
        environment['MIMIR_DISABLE_CPU'] = str(int(not allow_cpu_fallback))
    options = dict(
        planner_mode=mode,
        planner_enabled=bool(planner_enabled),
        **{option: bool(int(environment[flag]))
           for flag, option in PLANNER_OPTION_NAMES.items()},
    )
    if allow_cpu_fallback is not None:
        options['allow_cpu_fallback'] = allow_cpu_fallback
    identity = json.dumps(environment, sort_keys=True, separators=(',', ':'))
    configuration_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
    environment['MIMIR_PLANNER_JSON'] = '.mimir-spill/execution-plan-' + configuration_id + '.json'
    return dict(configuration_id=configuration_id, environment=environment,
                mimir_options=options)


def recommended(validated_runtimes):
    """Return a compact baseline plus one-factor-at-a-time diagnostic matrix.

    The baseline enables all validated GPU runtimes and uses the launcher's
    production-oriented settings. Each remaining row changes one planner,
    runtime, or wait-policy choice. Interactions remain available in the
    exhaustive matrix.
    """
    runtimes = _validated_runtimes(validated_runtimes)
    cases = []
    seen = set()

    def add(enabled_runtimes, planner_enabled=True, mode='cost',
            flags=None, wait_policy='active'):
        selected_flags = dict(RECOMMENDED_FLAGS)
        if flags:
            selected_flags.update(flags)
        allow_cpu_fallback = True if runtimes and not enabled_runtimes else None
        config = _configuration(enabled_runtimes, planner_enabled, mode,
                                selected_flags, wait_policy, allow_cpu_fallback)
        if config['configuration_id'] not in seen:
            seen.add(config['configuration_id'])
            cases.append(config)

    add(runtimes)
    add(runtimes, planner_enabled=False)
    for mode in ('legacy', 'static'):
        add(runtimes, mode=mode)
    for flag, default in RECOMMENDED_FLAGS.items():
        add(runtimes, flags={flag: 1 - default})
    add(runtimes, wait_policy='passive')

    runtime_selections = [[]]
    runtime_selections.extend([[runtime] for runtime in runtimes])
    for enabled in runtime_selections:
        add(enabled)
    return cases


def select(profile, validated_runtimes):
    """Select the recommended matrix by default or retain exhaustive coverage."""
    if profile == 'recommended':
        return recommended(validated_runtimes)
    if profile == 'full':
        return full_factorial(validated_runtimes)
    raise ValueError("profile must be 'recommended' or 'full'")


def configuration_count(validated_runtimes):
    """Return the number of cases without materializing the factorial matrix."""
    runtimes = _validated_runtimes(validated_runtimes)
    return (2 ** (len(runtimes) + len(PLANNER_FLAGS)) *
            len(PLANNER_MODES) * len(WAIT_POLICIES))


def recommended_configuration_count(validated_runtimes):
    """Return the compact matrix size for this host's validated runtimes."""
    return len(recommended(validated_runtimes))
