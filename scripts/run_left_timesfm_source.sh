#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=results/left_timesfm_hidden_20261126_v1/logs
mkdir -p "$out"
run() {
    CUDA_VISIBLE_DEVICES="$1" timeout 14400 envs/timeradar/bin/python -u \
        -m hybrid_tsad.left_timesfm_training "$PWD" train "$2" > "$out/source-$2.log" 2>&1
}
run 1 PSM & p1=$!
run 2 SWAT & p2=$!
run 3 SMD & p3=$!
status=0
for p in "$p1" "$p2" "$p3"; do wait "$p" || status=1; done
exit "$status"
