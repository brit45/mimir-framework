#!/bin/bash

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

# Afficher la configuration
echo "╔════════════════════════════════════════════════╗"
echo "║   LANCEMENT MÍMIR AVEC CONFIGURATION OPTIMALE  ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Configuration OpenMP:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "  OMP_PLACES=$OMP_PLACES"
echo "  OMP_SCHEDULE=$OMP_SCHEDULE"
echo ""
echo ""
echo "Vérifier avec: top -p \$(pgrep mimir)"
echo "Ou: ps -eLf | grep mimir | wc -l  (doit montrer ~7 lignes)"
echo ""

# Lancer mimir
exec ./bin/mimir "$@"
