#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# ============================================================
BENCH_CMD="${BENCH_CMD:-./run_mimir.sh --lua scripts/benchmarks/benchmark_stress.lua}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

# Files
BENCH_LOG="$LOG_DIR/bench.log"
TIME_LOG="$LOG_DIR/time.log"

PERF_BEFORE_DATA="${PERF_BEFORE_DATA:-$LOG_DIR/perf_before.data}" # optional
PERF_AFTER_DATA="$LOG_DIR/perf_after.data"
PERF_REPORT="$LOG_DIR/perf.report.log"
PERF_STAT="$LOG_DIR/perf.stat.log"
PERF_DIFF="$LOG_DIR/perf.diff.log"
PERF_TRACE="$LOG_DIR/perf.trace.log"

# perf sched
PERF_SCHED_DATA="$LOG_DIR/perf_sched.data"
PERF_SCHED_LAT="$LOG_DIR/perf.sched.latency.log"
PERF_SCHED_MAP="$LOG_DIR/perf.sched.map.log"
PERF_SCHED_TIMEHIST="$LOG_DIR/perf.sched.timehist.log"

# perf c2c
PERF_C2C_DATA="$LOG_DIR/perf_c2c.data"
PERF_C2C_REPORT="$LOG_DIR/perf.c2c.log"

# system monitors
PIDSTAT_CPU="$LOG_DIR/pidstat_cpu.log"
PIDSTAT_IO="$LOG_DIR/pidstat_io.log"
VMSTAT_LOG="$LOG_DIR/vmstat.log"
IOSTAT_LOG="$LOG_DIR/iostat.log"
NUMASTAT_LOG="$LOG_DIR/numastat.log"
PMAP_LOG="$LOG_DIR/pmap.log"

# perf record sampling
PERF_FREQ="${PERF_FREQ:-799}"
CALLGRAPH="${CALLGRAPH:-dwarf}"
STAT_EVENTS="${STAT_EVENTS:-cycles,instructions,branches,branch-misses,cache-references,cache-misses}"

# Feature flags (0/1)
ENABLE_C2C="${ENABLE_C2C:-1}"
ENABLE_SCHED="${ENABLE_SCHED:-1}"
ENABLE_TRACE="${ENABLE_TRACE:-1}"
ENABLE_SYSSTAT="${ENABLE_SYSSTAT:-1}"
ENABLE_NUMA="${ENABLE_NUMA:-1}"

# perf mem is fragile (IBS/PEBS). Keep OFF by default.
ENABLE_MEM="${ENABLE_MEM:-0}"
PERF_MEM_DATA="$LOG_DIR/perf_mem.data"
PERF_MEM_REPORT="$LOG_DIR/perf.mem.log"

# ============================================================
# Helpers
# ============================================================
log() { echo "[$(date +'%H:%M:%S')] $*"; }

kill_int() {
  local pid="$1"
  [[ -z "${pid:-}" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    sudo kill -INT "$pid" 2>/dev/null || true
  fi
}

wait_safely() {
  local pid="$1"
  [[ -z "${pid:-}" ]] && return 0
  wait "$pid" 2>/dev/null || true
}

# Find a descendant process by exact comm name (best effort, short polling)
find_descendant_by_comm() {
  local root_pid="$1"
  local target_comm="$2"

  local frontier=("$root_pid")
  for _ in $(seq 1 300); do
    local next=()
    for p in "${frontier[@]}"; do
      local kids
      kids="$(pgrep -P "$p" 2>/dev/null || true)"
      for k in $kids; do
        local comm
        comm="$(ps -p "$k" -o comm= 2>/dev/null || true)"
        if [[ "$comm" == "$target_comm" ]]; then
          echo "$k"
          return 0
        fi
        next+=("$k")
      done
    done
    frontier=("${next[@]}")
    sleep 0.05
  done
  return 1
}

# ============================================================
# Start bench (wrapper PID != mimir PID)
# ============================================================
log "Starting benchmark (with /usr/bin/time -v)..."
(
  /usr/bin/time -v $BENCH_CMD > "$BENCH_LOG" 2> "$TIME_LOG"
) &
WRAP_PID=$!
log "Wrapper PID: $WRAP_PID"

# Find real mimir pid
log "Searching for descendant process 'mimir'..."
MIMIR_PID="$(find_descendant_by_comm "$WRAP_PID" "mimir" || true)"

if [[ -z "$MIMIR_PID" ]]; then
  log "❌ Could not find mimir PID under wrapper PID=$WRAP_PID"
  log "Tip: verify your executable is really named 'mimir' (ps -eo comm,pid | grep mimir)"
  exit 1
fi

log "✅ MIMIR PID: $MIMIR_PID"

# ============================================================
# perf record
# ============================================================
log "Starting perf record..."
sudo perf record \
  -F "$PERF_FREQ" \
  -g \
  --call-graph "$CALLGRAPH" \
  -p "$MIMIR_PID" \
  -o "$PERF_AFTER_DATA" \
  -- sleep 999999 &
PERF_REC_PID=$!
log "perf record PID: $PERF_REC_PID"

# ============================================================
# perf stat
# ============================================================
log "Starting perf stat..."
sudo perf stat \
  -p "$MIMIR_PID" \
  -e "$STAT_EVENTS" \
  -- sleep 999999 \
  > "$PERF_STAT" 2>&1 &
PERF_STAT_PID=$!
log "perf stat PID: $PERF_STAT_PID"

# ============================================================
# perf sched
# ============================================================
PERF_SCHED_PID=""
if [[ "$ENABLE_SCHED" == "1" ]]; then
  log "Starting perf sched record..."
  sudo perf sched record -p "$MIMIR_PID" -o "$PERF_SCHED_DATA" -- sleep 999999 &
  PERF_SCHED_PID=$!
  log "perf sched PID: $PERF_SCHED_PID"
fi

# ============================================================
# perf c2c (VERY heavy)
# ============================================================
PERF_C2C_PID=""
if [[ "$ENABLE_C2C" == "1" ]]; then
  log "Starting perf c2c record..."
  sudo perf c2c record -p "$MIMIR_PID" -o "$PERF_C2C_DATA" -- sleep 999999 &
  PERF_C2C_PID=$!
  log "perf c2c PID: $PERF_C2C_PID"
fi

# ============================================================
# perf trace (syscalls)
# ============================================================
PERF_TRACE_PID=""
if [[ "$ENABLE_TRACE" == "1" ]]; then
  log "Starting perf trace..."
  sudo perf trace -p "$MIMIR_PID" > "$PERF_TRACE" 2>&1 &
  PERF_TRACE_PID=$!
  log "perf trace PID: $PERF_TRACE_PID"
fi

# ============================================================
# perf mem (fragile) - optional
# ============================================================
PERF_MEM_PID=""
if [[ "$ENABLE_MEM" == "1" ]]; then
  if perf list 2>/dev/null | grep -qiE 'ibs_op|pebs|mem-loads|mem-stores'; then
    log "Starting perf mem (fragile/heavy)..."
    sudo perf mem record -p "$MIMIR_PID" -o "$PERF_MEM_DATA" -- sleep 999999 &
    PERF_MEM_PID=$!
    log "perf mem PID: $PERF_MEM_PID"
  else
    log "perf mem not supported on this CPU/kernel, skipping."
  fi
fi

# ============================================================
# system monitors
# ============================================================
PIDSTAT_CPU_PID=""
PIDSTAT_IO_PID=""
VMSTAT_PID=""
IOSTAT_PID=""

if [[ "$ENABLE_SYSSTAT" == "1" ]]; then
  log "Starting pidstat/vmstat/iostat monitors..."
  pidstat -p "$MIMIR_PID" -u -r -w -t 1 > "$PIDSTAT_CPU" &
  PIDSTAT_CPU_PID=$!
  pidstat -p "$MIMIR_PID" -d 1 > "$PIDSTAT_IO" &
  PIDSTAT_IO_PID=$!
  vmstat 1 > "$VMSTAT_LOG" &
  VMSTAT_PID=$!
  iostat -xz 1 > "$IOSTAT_LOG" &
  IOSTAT_PID=$!
fi

# ============================================================
# NUMA + memory map snapshots
# ============================================================
if [[ "$ENABLE_NUMA" == "1" ]]; then
  sleep 2
  numastat -p "$MIMIR_PID" > "$NUMASTAT_LOG" 2>/dev/null || true
  pmap -x "$MIMIR_PID" > "$PMAP_LOG" 2>/dev/null || true
fi

# ============================================================
# Wait benchmark end (wrapper ends when bench ends)
# ============================================================
wait_safely "$WRAP_PID"
log "Benchmark finished."

# ============================================================
# Stop collectors
# ============================================================
log "Stopping collectors..."
kill_int "$PERF_REC_PID"
kill_int "$PERF_STAT_PID"
kill_int "$PERF_SCHED_PID"
kill_int "$PERF_C2C_PID"
kill_int "$PERF_MEM_PID"

# perf trace doesn't always react to -INT same way; try anyway
if [[ -n "${PERF_TRACE_PID:-}" ]]; then
  sudo kill -INT "$PERF_TRACE_PID" 2>/dev/null || true
fi

# stop monitors
kill "$PIDSTAT_CPU_PID" 2>/dev/null || true
kill "$PIDSTAT_IO_PID" 2>/dev/null || true
kill "$VMSTAT_PID" 2>/dev/null || true
kill "$IOSTAT_PID" 2>/dev/null || true

# wait for perf tools to flush
wait_safely "$PERF_REC_PID"
wait_safely "$PERF_STAT_PID"
wait_safely "$PERF_SCHED_PID"
wait_safely "$PERF_C2C_PID"
wait_safely "$PERF_MEM_PID"
wait_safely "$PERF_TRACE_PID"

sleep 1

# ============================================================
# Generate reports
# ============================================================
log "Generating perf report..."
sudo perf report --stdio -i "$PERF_AFTER_DATA" > "$PERF_REPORT"

if [[ "$ENABLE_SCHED" == "1" ]]; then
  log "Generating perf sched reports..."
  sudo perf sched latency  -i "$PERF_SCHED_DATA" > "$PERF_SCHED_LAT"
  sudo perf sched map      -i "$PERF_SCHED_DATA" > "$PERF_SCHED_MAP"
  sudo perf sched timehist -i "$PERF_SCHED_DATA" > "$PERF_SCHED_TIMEHIST"
fi

if [[ "$ENABLE_C2C" == "1" ]]; then
  log "Generating perf c2c report..."
  sudo perf c2c report -i "$PERF_C2C_DATA" > "$PERF_C2C_REPORT"
fi

if [[ "$ENABLE_MEM" == "1" && -f "$PERF_MEM_DATA" ]]; then
  log "Generating perf mem report..."
  sudo perf mem report -i "$PERF_MEM_DATA" > "$PERF_MEM_REPORT" || true
fi

# perf diff (if before exists)
if [[ -f "$PERF_BEFORE_DATA" ]]; then
  log "Generating perf diff..."
  sudo perf diff "$PERF_BEFORE_DATA" "$PERF_AFTER_DATA" > "$PERF_DIFF" || true
else
  log "perf diff skipped (no $PERF_BEFORE_DATA)."
fi

log "DONE. Logs in: $LOG_DIR"
log "  bench: $BENCH_LOG"
log "  time : $TIME_LOG"
log "  perf report: $PERF_REPORT"
log "  perf stat  : $PERF_STAT"

