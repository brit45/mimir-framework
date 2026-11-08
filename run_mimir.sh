#!/bin/bash

MPI_REQUESTED_RANKS=""
MPI_RANKS_SET=0
MPI_LAUNCHER="${MIMIR_MPI_LAUNCHER:-mpirun}"
declare -a MIMIR_ARGS=()

launcher_usage() {
	cat <<'EOF'
Usage: ./run_mimir.sh [launcher options] <mimir options>

Launcher options:
  --mpi-ranks N, --mpi N, --ddp-ranks N
      Start Mímir under mpirun -n N and split default OpenMP threads across ranks.
  --mpi-exec PATH
      MPI launcher executable (default: MIMIR_MPI_LAUNCHER or mpirun).
  --launcher-help
      Show this help.

Example:
  ./run_mimir.sh --mpi-ranks 2 --lua scripts/training/train_causal_lm.lua -- --conf configs/causal_lm.json --ddp --viz

The training script must also enable DDP (for example, with --ddp or
training.ddp=true in its configuration).
EOF
}

while (($#)); do
	case "$1" in
		--launcher-help)
			launcher_usage
			exit 0
			;;
		--mpi-ranks|--mpi|--ddp-ranks)
			if [ -z "${2:-}" ] || [[ "$2" == --* ]]; then
				echo "Missing rank count after $1" >&2
				exit 2
			fi
			if [ "$MPI_RANKS_SET" -eq 1 ]; then
				echo "MPI rank count was specified more than once" >&2
				exit 2
			fi
			MPI_REQUESTED_RANKS="$2"
			MPI_RANKS_SET=1
			shift 2
			;;
		--mpi-ranks=*|--mpi=*|--ddp-ranks=*)
			if [ "$MPI_RANKS_SET" -eq 1 ]; then
				echo "MPI rank count was specified more than once" >&2
				exit 2
			fi
			MPI_REQUESTED_RANKS="${1#*=}"
			MPI_RANKS_SET=1
			shift
			;;
		--mpi-exec)
			if [ -z "${2:-}" ] || [[ "$2" == --* ]]; then
				echo "Missing executable path after --mpi-exec" >&2
				exit 2
			fi
			MPI_LAUNCHER="$2"
			shift 2
			;;
		--mpi-exec=*)
			MPI_LAUNCHER="${1#*=}"
			shift
			;;
		*)
			MIMIR_ARGS+=("$1")
			shift
			;;
	esac
done

if [ "$MPI_RANKS_SET" -eq 1 ]; then
	case "$MPI_REQUESTED_RANKS" in
		''|*[!0-9]*|0|1)
			echo "DDP requires an integer MPI rank count of at least 2: $MPI_REQUESTED_RANKS" >&2
			exit 2
			;;
	esac
	MPI_REQUESTED_RANKS=$((10#$MPI_REQUESTED_RANKS))
	if [ "$MPI_REQUESTED_RANKS" -lt 2 ]; then
		echo "DDP requires at least 2 MPI ranks" >&2
		exit 2
	fi
	for mpi_size_var in OMPI_COMM_WORLD_SIZE PMI_SIZE PMIX_SIZE MV2_COMM_WORLD_SIZE; do
		if [ -n "${!mpi_size_var:-}" ] && [ "${!mpi_size_var}" -gt 1 ] 2>/dev/null; then
			echo "Refusing to start nested MPI: already inside a $mpi_size_var=${!mpi_size_var} process group" >&2
			exit 2
		fi
	done
	if ! command -v "$MPI_LAUNCHER" >/dev/null 2>&1; then
		echo "MPI launcher not found: $MPI_LAUNCHER" >&2
		exit 127
	fi
fi

# GPU runtime switches (1=enabled, 0=disabled), applied when compiled and available.
# Notebook/runtime MIMIR_DISABLE_* overrides remain authoritative for CPU/TPU runs.
MIMIR_GPU_CUDA="${MIMIR_GPU_CUDA:-0}"
MIMIR_GPU_VULKAN="${MIMIR_GPU_VULKAN:-1}"
MIMIR_GPU_OPENCL="${MIMIR_GPU_OPENCL:-1}"
for runtime in CUDA VULKAN OPENCL; do
    enable_key="MIMIR_GPU_${runtime}"
    disable_key="MIMIR_DISABLE_${runtime}"
    case "${!enable_key}" in
        0|1) ;;
        *) echo "$enable_key must be 0 or 1" >&2; exit 2 ;;
    esac
    if [ -z "${!disable_key}" ]; then
        export "$disable_key=$((1-${!enable_key}))"
    fi
done

export MIMIR_ALLOCATOR_LOG=0
export MIMIR_ALLOCATOR_LOG_VERBOSE=0

# Global execution planner (override possible via environment).
export MIMIR_ENABLE_PLANNER="${MIMIR_ENABLE_PLANNER:-1}"
export MIMIR_PLANNER_MODE="${MIMIR_PLANNER_MODE:-static}" # legacy | static | cost
export MIMIR_PLANNER_BUFFER_REUSE="${MIMIR_PLANNER_BUFFER_REUSE:-1}"
export MIMIR_PLANNER_FUSION="${MIMIR_PLANNER_FUSION:-1}"
export MIMIR_ENABLE_FUSION_TRAIN="${MIMIR_ENABLE_FUSION_TRAIN:-1}"
export MIMIR_RUNTIME_AUTOTUNE="${MIMIR_RUNTIME_AUTOTUNE:-1}"
export MIMIR_PLANNER_COST_MODEL="${MIMIR_PLANNER_COST_MODEL:-0}"
export MIMIR_PLANNER_DUMP="${MIMIR_PLANNER_DUMP:-0}"
export MIMIR_PLANNER_JSON="${MIMIR_PLANNER_JSON:-.mimir-spill/execution-plan.json}"
export MIMIR_PLANNER_DEVICE_RESIDENCY="${MIMIR_PLANNER_DEVICE_RESIDENCY:-0}"

# OpenCL Rusticl (AMD/Mesa): activer automatiquement sur iGPU AMD.
# Override possible: RUSTICL_ENABLE=... ./run_mimir.sh
if [ -z "${RUSTICL_ENABLE:-}" ]; then
	export RUSTICL_ENABLE=radeonsi
fi

# Configuration OpenMP (override possible via env)
NPROC=$(nproc 2>/dev/null || echo 4)
if [ "$MPI_RANKS_SET" -eq 1 ]; then
	MPI_WORLD_SIZE="$MPI_REQUESTED_RANKS"
else
	MPI_WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-${PMIX_SIZE:-${MV2_COMM_WORLD_SIZE:-${SLURM_NTASKS:-1}}}}}"
fi
case "$MPI_WORLD_SIZE" in
    ''|*[!0-9]*|0) MPI_WORLD_SIZE=1 ;;
esac
MPI_WORLD_SIZE=$((10#$MPI_WORLD_SIZE))
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    if [ "$MPI_WORLD_SIZE" -gt 1 ] 2>/dev/null; then
        OMP_NUM_THREADS=$((NPROC / MPI_WORLD_SIZE))
        if [ "$OMP_NUM_THREADS" -lt 1 ]; then OMP_NUM_THREADS=1; fi
    else
        OMP_NUM_THREADS=$NPROC
    fi
fi
export OMP_NUM_THREADS
export OMP_PROC_BIND=false
export OMP_PLACES=cores
export OMP_SCHEDULE="static"

# Limiter la rétention des gros buffers temporaires dans une arène glibc par worker.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# Désactiver nested parallelism
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

# CRITIQUE : Garder les threads en vie entre les régions parallèles
if [ -z "${OMP_WAIT_POLICY:-}" ]; then
	if [ "$MPI_WORLD_SIZE" -gt 1 ] 2>/dev/null; then
		export OMP_WAIT_POLICY=passive
	else
		export OMP_WAIT_POLICY=active
	fi
else
	export OMP_WAIT_POLICY
fi
export OMP_DYNAMIC=false           # Nombre de threads fixe
# Pinning: ne pas brider par défaut. Si l'utilisateur n'a rien défini,
# épingler sur les OMP_NUM_THREADS premiers CPU logiques hors MPI multi-rank.
if [ -z "${GOMP_CPU_AFFINITY:-}" ]; then
	if [ "$MPI_WORLD_SIZE" -le 1 ] 2>/dev/null; then
		if [ "$OMP_NUM_THREADS" -gt 1 ] 2>/dev/null; then
			export GOMP_CPU_AFFINITY="0-$((OMP_NUM_THREADS-1))"
		else
			export GOMP_CPU_AFFINITY="0"
		fi
	fi
fi

# ROCm guard: sur certaines stacks AMD (ex: gfx90c + rocBLAS incomplet),
# l'initialisation peut terminer en IOT/core dump. Par défaut on garde
# Vulkan/CPU stables, et on n'active ROCm qu'en opt-in explicite.
if [ -z "${MIMIR_DISABLE_ROCM:-}" ]; then
	if [ "${MIMIR_FORCE_ROCM:-0}" = "1" ]; then
		export MIMIR_DISABLE_ROCM=0
	else
		export MIMIR_DISABLE_ROCM=1
	fi
fi

# Sortie concise par défaut. MIMIR_LAUNCH_VERBOSE=1 restaure le diagnostic
# complet du lanceur; MIMIR_CONSOLE_VERBOSE=1 restaure toutes les lignes du
# framework (les fichiers logs/*.log restent toujours complets).
if [ "${MIMIR_LAUNCH_VERBOSE:-0}" = "1" ]; then
	echo "Configuration OpenMP:"
	echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS/$NPROC"
	echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
	echo "  OMP_PLACES=$OMP_PLACES"
	echo "  OMP_SCHEDULE=$OMP_SCHEDULE"
	echo "  OMP_NESTED=$OMP_NESTED"
	echo "  OMP_MAX_ACTIVE_LEVELS=$OMP_MAX_ACTIVE_LEVELS"
	echo "  OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
	echo "  OMP_DYNAMIC=$OMP_DYNAMIC"
	echo "  GOMP_CPU_AFFINITY=$GOMP_CPU_AFFINITY"
	echo "  MIMIR_DISABLE_ROCM=$MIMIR_DISABLE_ROCM"
	echo "  RUSTICL_ENABLE=$RUSTICL_ENABLE"
else
	echo "\n\n\t→\tMímir: threads=$OMP_NUM_THREADS/$NPROC mpi_ranks=$MPI_WORLD_SIZE rocm_disabled=$MIMIR_DISABLE_ROCM rusticl=$RUSTICL_ENABLE"
fi

# Lancer mimir, éventuellement sous MPI.
# Benchmarks may select another native Mímir entry point while retaining this
# exact runtime setup. Normal invocations keep the production executable.
MIMIR_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MIMIR_COMMAND="${MIMIR_EXECUTABLE:-$MIMIR_SCRIPT_DIR/bin/mimir}"
if [ "$MPI_RANKS_SET" -eq 1 ]; then
	exec "$MPI_LAUNCHER" -n "$MPI_REQUESTED_RANKS" "$MIMIR_COMMAND" "${MIMIR_ARGS[@]}"
else
	exec "$MIMIR_COMMAND" "${MIMIR_ARGS[@]}"
fi
