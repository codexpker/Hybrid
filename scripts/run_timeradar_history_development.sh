#!/usr/bin/env bash
set -euo pipefail
cd "$1"
export PYTHONPATH="$PWD/tmp/timeradar_history_context/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=/dev/shm/timeradar_history_development_v1
py=envs/timeradar/bin/python
lane() {
    local gpu=$1 arm=$2 domain=$3
    local waited=0
    until [[ -s "$out/source/$arm/receipt.json" ]]; do
        sleep 5
        waited=$((waited+5))
        [[ $waited -lt 1800 ]]
    done
    export CUDA_VISIBLE_DEVICES=$gpu
    "$py" -u -m hybrid_tsad.timeradar_history_experiment U "$PWD" "$out" --arm "$arm" --domain "$domain"
    "$py" -u -m hybrid_tsad.timeradar_history_evaluation score "$PWD" "$out" --arm "$arm" --domain "$domain" --stage U
    "$py" -u -m hybrid_tsad.timeradar_history_evaluation score "$PWD" "$out" --arm "$arm" --domain "$domain" --stage Z
}
lane 2 native PSM > "$out/logs/native_PSM.log" 2>&1 & a=$!
lane 3 native SWAT > "$out/logs/native_SWAT.log" 2>&1 & b=$!
lane 0 history PSM > "$out/logs/history_PSM.log" 2>&1 & c=$!
lane 1 history SWAT > "$out/logs/history_SWAT.log" 2>&1 & d=$!
status=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || status=1; done
exit "$status"
