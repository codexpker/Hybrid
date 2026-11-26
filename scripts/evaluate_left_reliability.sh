#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/tsad_research}"
export PYTHONPATH="$PWD/tmp/trajectory_src/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=results/left_local_reliability_20261126_v1
for mode in bank metrics; do
    for d in PSM SWAT; do
        gpu=0; [[ $d == PSM ]] || gpu=1
        CUDA_VISIBLE_DEVICES=$gpu /usr/bin/time -f '%e %U %S %M' -o "$out/$d/time_$mode.txt" envs/timeradar/bin/python -m hybrid_tsad.left_reliability_eval "$PWD" "$d" "$mode" > "$out/logs/${d}_${mode}.log" 2>&1 &
        if [[ $d == PSM ]]; then p=$!; else s=$!; fi
    done
    wait "$p"; wait "$s"
done
envs/timeradar/bin/python tmp/check_left_reliability.py "$PWD"
