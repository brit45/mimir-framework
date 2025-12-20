#!/bin/bash

# Configuration OpenMP optimale pour Mímir
#export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_SCHEDULE="static"

# Désactiver nested parallelism
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

# CRITIQUE : Garder les threads en vie entre les régions parallèles
export OMP_WAIT_POLICY=active     # Les threads restent actifs (pas de sleep)
export OMP_DYNAMIC=false           # Nombre de threads fixe
export GOMP_CPU_AFFINITY="0-5"    # Épingler sur les 6 premiers cœurs

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
echo "Utilisation CPU attendue: ~600% (6 threads × 100%)"
echo ""
echo "Vérifier avec: top -p \$(pgrep mimir)"
echo "Ou: ps -eLf | grep mimir | wc -l  (doit montrer ~7 lignes)"
echo ""

# Lancer mimir
exec ./bin/mimir "$@"
