#!/bin/bash
# Tracking-accuracy study (Appendix D): how well the warm-started block tracks the eigenpairs
# that the deflated projection removes.  Independent of the production benchmark: builds
# bin/track from tracking_mc.cu (a copy of ../src/admm_mc.cu plus FP64 checkpoints, see the
# header of tracking_mc.cu) and writes results/<instance>_deflated16_track[_it<maxiter>].txt.
# Nothing under ../src, ../scripts, ../results is touched.
#   run.sh build                                    compile bin/track
#   run.sh run <instance> [maxiter=10000] [sigma=1] [window=0.025] [gamma=1.1]
# instance = G55mc | G59mc | G60mc | G60_mb (../data/<instance>.sdp).  Env: DEFADMM_TRACK_EVERY
# (checkpoint period, default 100), DEFADMM_LANCZOS_SEED (default 20260916, as in the paper runs).
set -e
: "${DEFADMM_LANCZOS_SEED=20260916}"; export DEFADMM_LANCZOS_SEED
: "${DEFADMM_TRACK_EVERY=100}"; export DEFADMM_TRACK_EVERY
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"; PP="$ROOT/src/psd_projection"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.9.1-fasrc01 2>/dev/null || true
command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found"; exit 3; }
mkdir -p "$HERE/bin" "$HERE/results"
build() {
  nvcc -std=c++17 -arch=sm_90 -O2 "$HERE/tracking_mc.cu" \
    "$PP"/composite_FP32.cu "$PP"/composite_FP16.cu "$PP"/lanczos.cu "$PP"/utils.cu "$PP"/lobpcg.cu \
    -I "$ROOT/src" -lcublas -lcusolver -o "$HERE/bin/track"
  echo "BUILD_OK"
}
run() {   # instance maxiter sigma window gamma
  local inst=$1 it=${2:-10000} s=${3:-1} w=${4:-0.025} g=${5:-1.1}
  local tag=""; [ "$it" != "10000" ] && tag="_it$it"
  local out="$HERE/results/${inst}_deflated16_track${tag}.txt"
  echo "==== $inst deflated16 (maxiter=$it sigma=$s window=$w gamma=$g track_every=$DEFADMM_TRACK_EVERY lanczos_seed=$DEFADMM_LANCZOS_SEED) -> $out ===="
  "$HERE/bin/track" "$ROOT/data/$inst.sdp" deflated16 "$it" "$s" 0 25 "$w" "$g" 1 | tee "$out" | grep -E "^# track|^FINAL" | tail -4
}
case "${1:-build}" in
  build) build ;;
  run)   build; run "${2:?instance}" "${3:-10000}" "${4:-1}" "${5:-0.025}" "${6:-1.1}" ;;
  *) echo "unknown phase $1"; exit 2 ;;
esac
