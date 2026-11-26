#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/left_source_constraints_20261126_v1
run() {
    local domain=$1 device=$2
    for arm in mean separate source_only; do test -s "$out/$domain/$arm/receipt.json"; done
    CUDA_VISIBLE_DEVICES=$device /usr/bin/time -o "$out/$domain/eval-time.txt" -f '%e %U %S %M' \
        envs/timeradar/bin/python -m hybrid_tsad.left_source_constraints_eval "$PWD" "$domain" \
        > "$out/logs/${domain}_eval.log" 2>&1
}
run PSM 0 & a=$!
run SWAT 1 & b=$!
wait "$a"; wait "$b"
envs/timeradar/bin/python tmp/check_left_source_constraints.py "$PWD"
