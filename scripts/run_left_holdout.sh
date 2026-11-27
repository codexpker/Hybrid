#!/usr/bin/env bash
# Join an already running source job or execute the complete fixed fold.
set -euo pipefail

root=$(realpath "$1")
domain=$2
gpu=$3
source_pid=$4
cd "$root"
python=envs/timeradar/bin/python
output=results/left_holdout_history_20261127_v1
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$output/logs"

if [[ "$domain" == SMD ]]; then
    "$python" -m hybrid_tsad.left_holdout_report "$root" gate
fi

if [[ -n "$source_pid" ]]; then
    until [[ -s "$output/$domain/Z/COMPLETE.json" ]]; do
        state=$(ps -p "$source_pid" -o stat= || true)
        if [[ -z "$state" || "$state" == Z* ]]; then
            echo "Source process exited without sealing the fold: $domain" >&2
            exit 1
        fi
        sleep 10
    done
else
    "$python" -u -m hybrid_tsad.left_holdout_training "$root" "$domain" source
fi

evaluate() {
    "$python" -u -m hybrid_tsad.left_holdout_evaluation "$root" "$domain" "$@"
}
normal() {
    "$python" -u -m hybrid_tsad.left_holdout_training "$root" "$domain" normal "$@"
}

if [[ "$domain" == SMD ]]; then
    "$python" -c 'from pathlib import Path; from hybrid_tsad.left_history_public import members; print(*members(Path.cwd(),"SMD"),sep="\n")' |
    while read -r entity; do
        normal --entity "$entity"
        evaluate score --stage Z --entity "$entity"
        evaluate score --stage N --entity "$entity"
        CUDA_VISIBLE_DEVICES="" evaluate metrics --entity "$entity"
    done
else
    evaluate score --stage Z
    evaluate controlled --stage Z
    normal
    evaluate score --stage N
    evaluate controlled --stage N
    CUDA_VISIBLE_DEVICES="" evaluate metrics
fi
