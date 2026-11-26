#!/usr/bin/env bash
# Two workers per domain. Source and random_u may already have been produced.
set -eu
w=$(pwd)
d=$1
g=$2
second=$3
out=results/left_multisource_20261126_v1
py=envs/timeradar/bin/python
train() {
    local arm=$1 gpu=$2
    if [ ! -s "$out/$d/$arm/receipt.json" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$py" -u -m hybrid_tsad.left_multisource "$w" "$d" "$arm" >"$out/$d-$arm-worker.log" 2>&1
    fi
}
mkdir -p "$out"
train source "$g"
train N "$g" &
normal_pid=$!
train random_u "$second"
train ordinary "$second"
train replay "$second"
CUDA_VISIBLE_DEVICES="$second" "$py" -u -m hybrid_tsad.left_multisource_eval "$w" "$d" mixed
wait "$normal_pid"
CUDA_VISIBLE_DEVICES="$g" "$py" -u -m hybrid_tsad.left_multisource_eval "$w" "$d" public
