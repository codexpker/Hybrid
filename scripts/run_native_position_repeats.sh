#!/usr/bin/env bash
set -euo pipefail
cd /home/a312/tsad_research
export PYTHONPATH=preparations/native_position_aligned_20261124_v1/runtime
export CUDA_VISIBLE_DEVICES="$1"
shift
while (( $# )); do
  training_seed="$1"
  arm="$2"
  shift 2
  envs/timeradar_author_core_20261028_r2/bin/python -m hybrid_tsad.native_position_aligned_repeats train --seed "$training_seed" --arm "$arm"
done
