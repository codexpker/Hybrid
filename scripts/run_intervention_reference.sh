#!/usr/bin/env bash
# Run from the data workspace, with PYTHONPATH bound to this revision's src.
set -euo pipefail
w=$(pwd)
out=results/intervention_reference_20261126_v1
py=envs/timeradar/bin/python
metric=${1:?path to the pinned VUS implementation}
mkdir -p "$out"
jobs=()
launch() {
    local gpu=$1
    shift
    CUDA_VISIBLE_DEVICES="$gpu" "$py" -m hybrid_tsad.intervention_reference "$w" "$@" >"$out/$1-$2-$3.log" 2>&1 &
    jobs+=("$!")
}
join() {
    local p failed=0
    for p in "${jobs[@]}"; do wait "$p" || failed=1; done
    jobs=()
    return "$failed"
}
train() {
    if [[ ! -f "$out/$3/$2/receipt.json" ]]; then launch "$1" train "$2" "$3"; fi
}
bank() {
    if [[ ! -f "$out/$3/banks/$2.json" ]]; then launch "$1" score "$2" "$3"; fi
}
train 0 reconstruction source
train 1 pair source
join
bank 0 TEST PSM
bank 1 TEST SWAT
join
"$py" -m hybrid_tsad.intervention_reference_eval "$w" "$metric" TEST
train 0 replay PSM
train 1 absolute PSM
train 2 relative PSM
train 3 replay SWAT
join
train 0 absolute SWAT
train 1 relative SWAT
bank 2 QUERY PSM
bank 3 CAL PSM
join
bank 0 QUERY SWAT
bank 1 CAL SWAT
join
"$py" -m hybrid_tsad.intervention_reference_eval "$w" "$metric" QUERY

for domain in source PSM SWAT; do
    if [[ ! -f "$out/$domain/paired_check.json" ]]; then
        CUDA_VISIBLE_DEVICES=0 "$py" -m hybrid_tsad.intervention_reference_check "$w" "$domain"
    fi
done
repo_source=${PYTHONPATH%%:*}
"$py" "$repo_source/../scripts/collect_intervention_reference.py" "$w"
