"""Full factorial of discrete run_mimir.sh controls supported by a host."""
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
        environment = {name: '0' for name in GPU_FLAGS.values()}
        environment.update({GPU_FLAGS[runtime]: str(enabled)
                            for runtime, enabled in zip(runtimes, runtime_values)})
        for runtime, flag in GPU_FLAGS.items():
            environment['MIMIR_DISABLE_' + runtime.upper()] = str(
                1 - int(environment[flag]))
        environment.update({name: str(value) for name, value in flags.items()})
        environment.update(
            MIMIR_ENABLE_PLANNER=str(planner_enabled),
            MIMIR_PLANNER_MODE=mode,
            OMP_WAIT_POLICY=wait_policy,
        )
        options = dict(
            planner_mode=mode,
            planner_enabled=bool(planner_enabled),
            **{option: bool(int(environment[flag]))
               for flag, option in PLANNER_OPTION_NAMES.items()},
        )
        identity = json.dumps(environment, sort_keys=True, separators=(',', ':'))
        configuration_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
        environment['MIMIR_PLANNER_JSON'] = '.mimir-spill/execution-plan-' + configuration_id + '.json'
        yield dict(
            configuration_id=configuration_id,
            environment=environment,
            mimir_options=options,
        )


def configuration_count(validated_runtimes):
    """Return the number of cases without materializing the factorial matrix."""
    runtimes = tuple(validated_runtimes)
    if len(set(runtimes)) != len(runtimes) or any(runtime not in GPU_FLAGS for runtime in runtimes):
        raise ValueError('Expected unique validated runtimes from cuda, vulkan, opencl')
    return (2 ** (len(runtimes) + len(PLANNER_FLAGS)) *
            len(PLANNER_MODES) * len(WAIT_POLICIES))
