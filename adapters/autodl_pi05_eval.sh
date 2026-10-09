#!/usr/bin/env bash
set -euo pipefail

root="${1:?DEPTH_MODEL_ROOT path}"
family="${2:?dpv|rgbd4}"
train="${3:?d0|d1|d3}"
scene="${4:?easy|random}"
deploy="${5:?d0|d1|d3}"
gpu="${6:?physical GPU index}"
port="${7:?RPC port}"
batch="$root/evaluations/stack_matched_eval_20261008"

case "$train:$deploy:$scene" in
  d0:d0:easy|d0:d0:random|d0:d1:easy|d0:d1:random|d0:d3:easy|d0:d3:random|\
  d1:d0:easy|d1:d0:random|d1:d1:easy|d1:d1:random|d1:d3:easy|d1:d3:random|\
  d3:d0:easy|d3:d0:random|d3:d1:easy|d3:d1:random|d3:d3:easy|d3:d3:random) ;;
  *) echo "Unsupported train/deploy/scene combination" >&2; exit 2 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ ]] || exit 2
test -f "$batch/protocol.env"
export MATCHED_EVAL_BATCH="$batch"
export CUDA_VISIBLE_DEVICES="$gpu"

case "$family" in
  dpv)
    exec bash "$batch/bin/pi05_dpv_worker.sh" "$train" "$scene" "$deploy" "$gpu" "$port"
    ;;
  rgbd4)
    exec bash "$batch/bin/pi05_rgbd4_worker.sh" stack_blocks_two "$train" "$scene" "$deploy" "$gpu" "$port"
    ;;
  *) echo "Unsupported family: $family" >&2; exit 2 ;;
esac
