#!/usr/bin/env bash
set -euo pipefail
cd "$1"
out=/dev/shm/timeradar_history_development_v1
export PYTHONPATH="$PWD/tmp/timeradar_history_context/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=envs/timeradar/bin/python
ready() {
    local domain arm stage
    for domain in PSM SWAT; do
        for arm in native history; do
            [[ -s "$out/$domain/$arm/receipt.json" ]] || return 1
            for stage in Z U; do
                [[ -s "$out/$domain/scores/${stage}_${arm}/part0.json" ]] || return 1
            done
        done
    done
}
until ready; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 "$py" -m hybrid_tsad.timeradar_history_diagnostics "$PWD" "$out" > "$out/logs/final_diagnostics.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$py" -m hybrid_tsad.timeradar_history_evaluation evaluate "$PWD" "$out" > "$out/logs/evaluate.log" 2>&1
echo 'All fixed source/U score banks and metrics completed.'
