#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/left_local_reliability_20261126_v1
mkdir -p "$out/logs"
job() {
    CUDA_VISIBLE_DEVICES=$1 envs/timeradar/bin/python -m hybrid_tsad.left_reliability "$PWD" "$2" "$3" > "$out/logs/$2_$3.log" 2>&1
}
job 0 PSM prepare & a=$!
job 1 SWAT prepare & b=$!
wait "$a"; wait "$b"
queue() { local g=$1 d=$2 m=$3; for k in A B C; do job "$g" "$d" "${m}_$k"; done; }
queue 0 PSM window & a=$!
queue 1 SWAT window & b=$!
queue 2 PSM local & c=$!
queue 3 SWAT local & d=$!
rc=0
for p in "$a" "$b" "$c" "$d"; do wait "$p" || rc=1; done
exit "$rc"
