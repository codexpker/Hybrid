#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=envs/timeradar/bin/python
out=results/left_memory_exposure_20261126_v1
mkdir -p "$out/logs"
job() {
    CUDA_VISIBLE_DEVICES=$1 "$py" -m hybrid_tsad.left_memory_update "$PWD" "$2" "$3" > "$out/logs/$2_$3.log" 2>&1
}
prepare() {
    if [[ ! -s "$out/$2/plan/audit.json" ]]; then
        "$py" -m hybrid_tsad.left_memory_exposure "$PWD" "$2"
    fi
    job "$1" "$2" check
}
prepare 0 PSM & a=$!
prepare 1 SWAT & b=$!
wait "$a"; wait "$b"
queue() { local g=$1 d=$2; shift 2; for arm in "$@"; do job "$g" "$d" "$arm"; done; }
queue 0 PSM restricted_A ordinary_B replay_C restricted_C & a=$!
queue 1 SWAT restricted_A ordinary_B replay_C restricted_C & b=$!
queue 2 PSM replay_B restricted_B ordinary_C & c=$!
queue 3 SWAT replay_B restricted_B ordinary_C & e=$!
rc=0
for p in "$a" "$b" "$c" "$e"; do wait "$p" || rc=1; done
exit "$rc"
