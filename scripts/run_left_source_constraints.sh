#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/left_source_constraints_20261126_v1
mkdir -p "$out/logs"
job() {
    local device=$1 domain=$2 arm=$3
    if [[ -f "$out/$domain/$arm/receipt.json" ]]; then return; fi
    CUDA_VISIBLE_DEVICES=$device envs/timeradar/bin/python -m hybrid_tsad.left_source_constraints "$PWD" "$domain" "$arm" > "$out/logs/${domain}_${arm}.log" 2>&1
}
job 0 PSM check & a=$!
job 1 SWAT check & b=$!
wait "$a"; wait "$b"
(job 0 PSM mean && job 0 PSM source_only) & a=$!
(job 1 SWAT mean && job 1 SWAT source_only) & b=$!
job 2 PSM separate & c=$!
job 3 SWAT separate & d=$!
rc=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || rc=1; done
exit "$rc"
