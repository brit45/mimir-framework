# Native OpenCL / Vulkan reference operations

The OpenCL and Vulkan adapters retain their existing fast paths and use
`PortableRuntime` for the remaining operations and backward passes. Numerical
kernels execute through OpenCL compute or Vulkan compute shaders. Host memory
stages tensors and accumulates returned parameter gradients; it does not call
CPU forward/backward arithmetic as a hidden fallback.

Coverage matches the executable CPU dispatcher: 72 forward types and 73 backward
types. `Lambda` forward is rejected by the CPU dispatcher itself and therefore
also rejected here. Its identity backward is supported. Hard NMS returns discrete indices; its backward returns zero gradients for
boxes, scores and optional classes. Selection boundaries are nondifferentiable;
this is an explicit stop-gradient contract, not a differentiable NMS surrogate.
The CPU dispatcher uses the same NMS stop-gradient contract. Configuration flags and shape checks still apply.

## Parallel execution and budgets

Kernels with independent output ownership use a parallel grid; supported
reductions use cooperative workgroups. Intrinsically serial algorithms retain
a bounded reference path. Capability queries and staging validate the operation
geometry before device submission. Numerical coverage does not imply a speedup
over the CPU.

`Planning/ExecutionPolicy.hpp` defines the protection limits used by
`PortableWorkBudget.hpp`. The serial reference path is limited to 65,536 staged
floats and 4,194,304 units of logical work. Parallel kernels have separate
invocation, arena and work limits. Pointwise fusion additionally bounds its
whole packed program to 8,388,608 floats, including saved inputs and outputs.
A refused dispatch returns to the router's ordinary fallback policy.

## Native pointwise fusion

`PointwiseFusionKernels.cuh` is the common GPU source for forward and backward
chains of Identity, ReLU, LeakyReLU, GELU, SiLU, Tanh, Sigmoid, Softplus, Mish,
HardSigmoid and HardSwish. The generator emits two additional portable kernels
and `PointwiseFusionIds.hpp`; CUDA and HIP compile the common source directly.

`GpuPointwiseFusion.hpp` packs validated programs. `PortablePointwiseFusion.hpp`
submits one kernel per direction through `Device::runBatch()`, downloading the
requested output range. Each invocation evaluates all stages for its element.
Forward can retain every intermediate output for training, while backward uses
saved stage inputs. This is distinct from submitting several resident kernels
without downloading their intermediate tensors.

The runtime advertises a native candidate through the pointwise fusion hooks.
The planner selects it only after exact comparison with the ordinary routed
chain and a measured latency win; a faster CPU or unfused route can still win.
See the [planner documentation](../../../docs/04-Architecture-Internals/22-Planning.md)
for cache keys, invalidation and periodic recalibration.

## Generation

From the repository root:

```sh
python3 scripts/dev/generate_portable_reference.py
```

Generation requires Python 3, `glslangValidator` and `spirv-val`. Generated host
wrappers and OpenCL/GLSL kernels reuse the existing GPU algorithms aligned with
the CPU contract. Embedded sources/SPIR-V allow ordinary builds without shader
compiler or CUDA toolkit dependencies. CPU sources are read-only inputs to
capability checks and numerical tests.

## Validation

With `MIMIR_ENABLE_TESTS=ON` and the relevant backend enabled, CTest registers
`RuntimeTest.OPENCLNativeOps` and `RuntimeTest.VULKANNativeOps`. Tests compare
forward values, input/parameter gradients and repeated accumulation against
the CPU dispatcher, validate stochastic saved context, and assert the router
selects the requested GPU backend. A missing device returns skip code 77.


`RuntimeTest.OPENCLPointwiseFusion` and `RuntimeTest.VULKANPointwiseFusion`
exercise the fused kernels directly on the requested backend: 121 activation
pairs, forward/backward, retained/final-only outputs, empty inputs and irregular
sizes. CPU comparisons use a numerical tolerance; planner selection separately
requires bit-exact agreement with its routed reference. `RuntimeTest.GpuFusionPacking`
checks packed layouts and work budgets without a device.

On 28 October 2026 these fusion tests passed on AMD Radeon Graphics with
OpenCL/Rusticl and Vulkan/RADV RENOIR. The shared HIP runtime source also compiled
for `gfx90c`, including through CMake, without ROCm execution. CUDA fusion was
implemented but not compiled or executed locally because `nvcc` was unavailable.
These results do not establish a GPU performance gain. See the
[backend validation details](../../../docs/04-Architecture-Internals/21-GPU-Runtimes.md#validation-de-la-fusion-au-28-octobre-2026).
