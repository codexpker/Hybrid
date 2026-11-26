#!/usr/bin/env bash
# Four independent lanes. Reuse only sealed endpoints and verify their hashes.
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=results/left_history_relation_20261126_v1/logs
mkdir -p "$out"
job() {
    CUDA_VISIBLE_DEVICES="$1" envs/timeradar/bin/python -u -m hybrid_tsad.left_history_training \
        "$PWD" "$2" "$3" > "$out/$2-$3.log" 2>&1
}
job 0 PSM joint & a=$!
(job 1 PSM continue && job 1 SMD joint) & b=$!
job 2 SWAT joint & c=$!
(job 3 SWAT continue && job 3 SMD continue) & d=$!
rc=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || rc=1; done
exit "$rc"
