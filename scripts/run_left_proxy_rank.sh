#!/usr/bin/env bash
# Run the four independently recoverable source/replay lanes.
set -euo pipefail
cd "$(realpath "$1")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$(realpath "$2")"
out=results/left_proxy_rank_20261127_v1
py=envs/timeradar/bin/python

if [[ $# -eq 5 ]]; then
    domain=$3 arm=$4 gpu=$5
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ]]
    export CUDA_VISIBLE_DEVICES="$gpu"
    printf 'start %s %s %s GPU%s\n' "$(date -u +%FT%TZ)" "$domain" "$arm" "$gpu"
    nvidia-smi -i "$gpu" --query-gpu=index,uuid,name --format=csv,noheader
    trap 'status=$?; printf "end %s exit=%s\n" "$(date -u +%FT%TZ)" "$status"' EXIT
    run() { "$py" -u -m hybrid_tsad.left_proxy_rank "$PWD" "$domain" "$@" --arm "$arm"; }
    run Z
    run score --score-stage Z
    run U
    run score --score-stage U
    exit
fi

for domain in PSM SWAT; do
    test -s "$out/$domain/check_cpu.json"
    test -s "$out/$domain/check_cuda.json"
done
mkdir -p "$out/logs"
pids=()
script="$(realpath "$0")"
# The per-lane cut-off bounds the four success lanes together below 8 A40-hours.
for spec in PSM:C:0 PSM:M1:1 SWAT:C:2 SWAT:M1:3; do
    IFS=: read -r domain arm gpu <<< "$spec"
    timeout --kill-after=15s 6900 bash "$script" "$PWD" "$PYTHONPATH" "$domain" "$arm" "$gpu" \
        > "$out/logs/$domain-$arm.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
