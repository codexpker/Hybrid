#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${2:-1}
out=results/left_window_permutation_20261126_v1
mkdir -p "$out/logs"
for domain in PSM SWAT; do
    for k in A B C; do
        envs/timeradar/bin/python -u -m hybrid_tsad.left_reliability "$PWD" "$domain" "permuted_$k" > "$out/logs/${domain}_$k.log" 2>&1
    done
done
envs/timeradar/bin/python -m hybrid_tsad.left_permutation_eval "$PWD" bank > "$out/logs/banks.log" 2>&1
touch "$out/gpu_done"
envs/timeradar/bin/python -m hybrid_tsad.left_permutation_eval "$PWD" metrics > "$out/logs/metrics.log" 2>&1
touch "$out/metrics_done"
