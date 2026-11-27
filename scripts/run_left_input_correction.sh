#!/usr/bin/env bash
set -euo pipefail
cd "$(realpath "$1")"
export PYTHONPATH="$(realpath "$2")" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=results/left_input_correction_20261127_v1
py=envs/timeradar/bin/python

if [[ $# -gt 2 ]]; then
    gpu=$3 domain=$4
    shift 4
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ]]
    export CUDA_VISIBLE_DEVICES="$gpu"
    trap 'status=$?; printf "end %s exit=%s\n" "$(date -u +%FT%TZ)" "$status"' EXIT
    for method; do
        printf 'start %s %s %s GPU%s\n' "$(date -u +%FT%TZ)" "$domain" "$method" "$gpu"
        "$py" -u -m hybrid_tsad.left_input_correction "$PWD" "$domain" "$method"
    done
    exit
fi

mkdir -p "$out/logs"
for domain in PSM SWAT; do
    test -s "$out/$domain/check.json"
done
runner=$(realpath "$0")
timeout --kill-after=15s 6900 bash "$runner" "$PWD" "$PYTHONPATH" 0 PSM R0 R2 R4 R6 > "$out/logs/psm-0.log" 2>&1 & a=$!
timeout --kill-after=15s 6900 bash "$runner" "$PWD" "$PYTHONPATH" 1 PSM R1 R3 R5 R7 > "$out/logs/psm-1.log" 2>&1 & b=$!
timeout --kill-after=15s 6900 bash "$runner" "$PWD" "$PYTHONPATH" 2 SWAT R0 R2 R4 R6 > "$out/logs/swat-0.log" 2>&1 & c=$!
timeout --kill-after=15s 6900 bash "$runner" "$PWD" "$PYTHONPATH" 3 SWAT R1 R3 R5 R7 > "$out/logs/swat-1.log" 2>&1 & d=$!
rc=0
for p in "$a" "$b" "$c" "$d"; do wait "$p" || rc=1; done
exit "$rc"
