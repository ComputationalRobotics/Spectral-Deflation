#!/bin/bash
# Build and run the Sec. 4.2 benchmark of Kang et al. (Mittelmann's G55mc, G59mc, G60mc, G60_mb),
# FP16 throughout (no switch to FP64), always the full 10000 iterations (no eta stop).   usage:
#   scripts/run.sh build                                          compile bin/admm and bin/pm_head_test
#   scripts/run.sh bench <instance> [baseline16|deflated16|both] [maxiter=10000] [sigma=1] [window=0.025] [gamma=1.1]
#       (gamma != 1.1 writes results/<instance>_<mode>_g<gamma>.txt)
#   scripts/run.sh baseline16|deflated16|baseline32 <instance> [maxiter] [sigma] [window]
#   scripts/run.sh synthetic                                      +/- head single-projection test
# instance = G55mc | G59mc | G60mc | G60_mb (data/<instance>.sdp).  Writes results/<instance>_<mode>.txt.
# The projection output is symmetrized, P <- (P + P^T)/2 (admm_mc argv[9] symmetrize, default 1;
# SYM=0 disables it).  DEFADMM_LANCZOS_SEED (default 20260916) makes the Lanczos start vectors
# reproducible; DEFADMM_LANCZOS_SEED="" restores the reference code's random seeding.
# Needs nvcc (sm_90: H100/H200) on the node.
set -e
: "${DEFADMM_LANCZOS_SEED=20260916}"; export DEFADMM_LANCZOS_SEED
SYM="${SYM:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PP="$ROOT/src/psd_projection"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.9.1-fasrc01 2>/dev/null || true
command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found"; exit 3; }
mkdir -p "$ROOT/bin" "$ROOT/results"
build() {
  nvcc -std=c++17 -arch=sm_90 -O2 "$ROOT/src/admm_mc.cu" \
    "$PP"/composite_FP32.cu "$PP"/composite_FP16.cu "$PP"/lanczos.cu "$PP"/utils.cu "$PP"/lobpcg.cu \
    -I "$ROOT/src" -lcublas -lcusolver -o "$ROOT/bin/admm"
  nvcc -std=c++17 -arch=sm_90 -O2 "$ROOT/test/pm_head_test.cu" \
    "$PP"/composite_FP32.cu "$PP"/composite_FP16.cu "$PP"/lanczos.cu "$PP"/utils.cu "$PP"/lobpcg.cu \
    -I "$ROOT/src" -lcublas -lcusolver -o "$ROOT/bin/pm_head_test"
  echo "BUILD_OK"
}
n_of() { head -1 "$ROOT/data/$1.sdp" | awk '{print $1}'; }
run_arm() {   # mode instance maxiter sigma window gamma
  local m=$1 inst=$2 it=${3:-10000} s=${4:-1} w=${5:-0.025} g=${6:-1.1} n; n=$(n_of "$inst")
  local tag=""; [ "$g" != "1.1" ] && tag="_g$g"
  echo "==== $inst $m (n=$n maxiter=$it sigma=$s window=$w gamma=$g symmetrize=$SYM lanczos_seed=$DEFADMM_LANCZOS_SEED) ===="
  "$ROOT/bin/admm" "$ROOT/data/$inst.sdp" "$m" "$it" "$s" 0 25 "$w" "$g" "$SYM" \
    | tee "$ROOT/results/${inst}_${m}${tag}.txt" | tail -2
}
synthetic() {
  echo "==== synthetic +/- head single-projection test ===="
  "$ROOT/bin/pm_head_test" 5000 80 60 3.0 40 200 30 150 7 | tee "$ROOT/results/synthetic_pm_head.txt"
  "$ROOT/bin/pm_head_test" 5000 80 0 3.0 40 200 30 150 7 | tee -a "$ROOT/results/synthetic_pm_head.txt"
}
case "${1:-build}" in
  build)      build ;;
  bench)      # bench <instance> <baseline16|deflated16|both> [maxiter=10000] [sigma=1] [window=0.025] [gamma=1.1]
              build; inst="${2:?instance}"; arm="${3:-both}"
              if [ "$arm" = both ]; then run_arm baseline16 "$inst" "${4:-10000}" "${5:-1}"; run_arm deflated16 "$inst" "${4:-10000}" "${5:-1}" "${6:-0.025}" "${7:-1.1}"
              else run_arm "$arm" "$inst" "${4:-10000}" "${5:-1}" "${6:-0.025}" "${7:-1.1}"; fi ;;
  baseline16|baseline32|deflated16)
              build; run_arm "$1" "${2:?instance}" "${3:-10000}" "${4:-1}" "${5:-0.025}" ;;
  synthetic)  build; synthetic ;;
  *) echo "unknown phase $1"; exit 2 ;;
esac
