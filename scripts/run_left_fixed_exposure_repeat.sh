#!/usr/bin/env bash
set -euo pipefail
cd "$(realpath "$1")"
export PYTHONPATH="$(realpath "$2")" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out="$(realpath "$3")"
py=envs/timeradar/bin/python
run() { "$py" -u -m hybrid_tsad.left_fixed_exposure_repeat "$PWD" "$out" "$@"; }
if [[ $# -gt 3 ]]; then
    domain=$4 seed=$5 gpu=$6
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ]]
    export CUDA_VISIBLE_DEVICES="$gpu"
    printf 'start %s %s %s GPU%s\n' "$(date -u +%s.%N)" "$domain" "$seed" "$gpu"
    trap 'status=$?; printf "end %s exit=%s\n" "$(date -u +%s.%N)" "$status"' EXIT
    for c in A B C; do
        for m in R W; do
            run adapt --domain "$domain" --method "$m" --condition "$c" --seed "$seed"
            run score --domain "$domain" --method "$m" --condition "$c" --seed "$seed"
        done
    done
    exit
fi
test -s "$out/identity.json"
mkdir -p "$out/logs"
for domain in PSM SWAT; do
    test -s "$out/$domain/rng_probe.json"
done
runner=$(realpath "$0")
timeout --kill-after=10s 1720 bash "$runner" "$PWD" "$PYTHONPATH" "$out" PSM 2026 0 > "$out/logs/PSM-2026.log" 2>&1 & a=$!
timeout --kill-after=10s 1720 bash "$runner" "$PWD" "$PYTHONPATH" "$out" PSM 2027 1 > "$out/logs/PSM-2027.log" 2>&1 & b=$!
timeout --kill-after=10s 1720 bash "$runner" "$PWD" "$PYTHONPATH" "$out" SWAT 2026 2 > "$out/logs/SWAT-2026.log" 2>&1 & c=$!
timeout --kill-after=10s 1720 bash "$runner" "$PWD" "$PYTHONPATH" "$out" SWAT 2027 3 > "$out/logs/SWAT-2027.log" 2>&1 & d=$!
status=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || status=1; done
exit "$status"
