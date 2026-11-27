#!/usr/bin/env bash
# Wait for both complete development folds, then evaluate the prespecified expansion.
set -euo pipefail
root=$(realpath "$1")
runner=$(realpath "$2")
psm_worker=$3
swat_worker=$4
cd "$root"
output=results/left_holdout_history_20261127_v1
python=envs/timeradar/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

wait_fold() {
    local domain=$1 pid=$2 state
    until [[ -s "$output/public/$domain/$domain/METRICS.json" ]]; do
        state=$(ps -p "$pid" -o stat= || true)
        if [[ -z "$state" || "$state" == Z* ]]; then
            echo "$domain development pipeline exited before the metric seal" >&2
            exit 1
        fi
        sleep 10
    done
}
wait_fold PSM "$psm_worker"
wait_fold SWAT "$swat_worker"
CUDA_VISIBLE_DEVICES="" "$python" -u -m hybrid_tsad.left_holdout_evaluation "$root" PSM controlled-metrics
if CUDA_VISIBLE_DEVICES="" "$python" -m hybrid_tsad.left_holdout_report "$root" gate; then
    bash "$runner" "$root" SMD 0 ""
else
    status=$?
    if [[ "$status" != 42 ]]; then
        exit "$status"
    fi
fi
CUDA_VISIBLE_DEVICES="" "$python" -u -m hybrid_tsad.left_holdout_report "$root" report
