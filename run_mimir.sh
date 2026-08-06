#!/bin/bash

export MIMIR_ALLOCATOR_LOG=1
export MIMIR_ALLOCATOR_LOG_VERBOSE=1

# OpenCL Rusticl (AMD/Mesa): activer automatiquement sur iGPU AMD.
# Override possible: RUSTICL_ENABLE=... ./run_mimir.sh
if [ -z "${RUSTICL_ENABLE:-}" ]; then
	export RUSTICL_ENABLE=radeonsi
fi

# Configuration OpenMP (override possible via env)
NPROC=$(nproc 2>/dev/null || echo 4)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC}"
export OMP_PROC_BIND=false
export OMP_PLACES=cores
export OMP_SCHEDULE="static"

# Désactiver nested parallelism
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

# CRITIQUE : Garder les threads en vie entre les régions parallèles
export OMP_WAIT_POLICY=active     # Les threads restent actifs (pas de sleep)
export OMP_DYNAMIC=false           # Nombre de threads fixe
# Pinning: ne pas brider par défaut. Si l'utilisateur n'a rien défini,
# épingler sur les OMP_NUM_THREADS premiers CPU logiques.
if [ -z "${GOMP_CPU_AFFINITY:-}" ]; then
	if [ "$OMP_NUM_THREADS" -gt 1 ] 2>/dev/null; then
		export GOMP_CPU_AFFINITY="0-$((OMP_NUM_THREADS-1))"
	else
		export GOMP_CPU_AFFINITY="0"
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

# Afficher la configuration
echo "╔════════════════════════════════════════════════╗"
echo "║   LANCEMENT MÍMIR AVEC CONFIGURATION OPTIMALE  ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
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
echo "  MIMIR_DISABLE_ROCM=$MIMIR_DISABLE_ROCM (override: MIMIR_FORCE_ROCM=1)"
echo "  RUSTICL_ENABLE=$RUSTICL_ENABLE"
echo ""
echo ""
echo "Vérifier avec: top -p \$(pgrep mimir)"
echo "Ou: ps -eLf | grep mimir | wc -l  (doit montrer $OMP_NUM_THREADS)"
echo ""

# Lancer mimir
exec ./bin/mimir "$@"
