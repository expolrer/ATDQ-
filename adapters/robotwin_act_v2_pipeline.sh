#!/usr/bin/env bash
set -euo pipefail

root="${1:?DEPTH_MODEL_ROOT path}"
variant="${2:?ACT-v2 variant}"
task="${3:?RoboTwin task}"
config="${4:?RoboTwin task config}"
gpu="${5:?physical GPU index}"
tier="${6:?smoke|full}"
steps="${7:?training steps}"
rollouts="${8:?evaluation rollouts}"

case "$variant" in
  B0_RGB|B1_EARLY_RGBD|B2_GATED_DEPTH|B3_GATED_XYZMAP|B4_POINTCLOUD) ;;
  *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
esac
case "$tier" in smoke|full) ;; *) echo "Unsupported tier: $tier" >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[2-7]$ && "$steps" =~ ^[0-9]+$ && "$rollouts" =~ ^[0-9]+$ ]] || exit 2
[[ "$task" =~ ^[A-Za-z0-9_]+$ && "$config" =~ ^[A-Za-z0-9_]+$ ]] || exit 2

train_repo="$root/repos/rgbd-act-v2"
robotwin="$root/repos/RoboTwin"
python_train="$root/envs/dp3/bin/python"
python_eval="$root/envs/robotwin/bin/python"
run="$root/experiments/RGBDACTv2/$variant/$task/${tier}_seed0"
status="$run/status.txt"
checkpoint="$run/policy_step_$(printf '%06d' "$steps").pt"
isolation="$root/runtime/sapien-gpu-isolation/libsapien_gpu_redirect.so"
setting="${variant}-${tier}-seed0"
mkdir -p "$run"
exec 9>"$run/atdq.lock"
flock -n 9 || exit 75

test -x "$python_train"
test -x "$python_eval"
test -f "$train_repo/train_rgbd_act_v2.py"
test -f "$robotwin/script/eval_policy.py"
test -f "$robotwin/task_config/$config.yml"
test -r "$isolation"

fail() {
  printf '%s FAILED stage=%s gpu=%s tier=%s\n' "$(date --iso-8601=seconds)" "$1" "$gpu" "$tier" >"$status"
}
trap 'fail interrupted' INT TERM
trap 'fail runtime' ERR

if [[ ! -s "$checkpoint" || ! -s "$run/policy_best.pt" ]]; then
  if [[ "$tier" == smoke ]]; then
    validate_every=250; save_every=250; validation_batches=10; warmup_steps=100
  else
    validate_every=500; save_every=5000; validation_batches=50; warmup_steps=1000
  fi
  printf '%s TRAINING gpu=%s tier=%s steps=%s\n' "$(date --iso-8601=seconds)" "$gpu" "$tier" "$steps" >"$status"
  (
    cd "$train_repo"
    env CUDA_VISIBLE_DEVICES="$gpu" TORCH_HOME="$root/models/torch" PYTHONPATH="$train_repo" \
      "$python_train" train_rgbd_act_v2.py \
        --variant "$variant" --task "$task" --data-root "$root/datasets" \
        --output "$run" --steps "$steps" --batch-size 8 --workers 4 \
        --warmup-steps "$warmup_steps" --save-every "$save_every" \
        --validate-every "$validate_every" --validation-batches "$validation_batches" \
        --seed 0 --resume
  )
fi
test -s "$checkpoint"
test -s "$run/policy_best.pt"

seed_dir="$robotwin/eval_result/$task/ACT/$config/A0-seed0"
seed_file="$(find "$seed_dir" -type f -name valid_seeds.txt -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
test -n "$seed_file"
test -s "$seed_file"
printf '%s EVALUATING gpu=%s tier=%s rollouts=%s\n' "$(date --iso-8601=seconds)" "$gpu" "$tier" "$rollouts" >"$status"
(
  cd "$robotwin"
  env CUDA_VISIBLE_DEVICES="$gpu" ROBOTWIN_PHYSICAL_GPU="$gpu" \
    ROBOTWIN_DRM_RENDER_INDEX="$((128 + gpu))" LD_PRELOAD="$isolation" \
    TORCH_HOME="$root/models/torch" PYTHONPATH=".:$train_repo" \
    "$python_eval" script/eval_policy.py \
      --config policy/RGBDACTv2/deploy_policy.yml --overrides \
      --policy_name RGBDACTv2 --variant "$variant" --task_name "$task" \
      --task_config "$config" --ckpt_setting "$setting" \
      --checkpoint "$run/policy_best.pt" --seed 0 \
      --seed_list_path "$seed_file" --test_num "$rollouts"
)
result_dir="$robotwin/eval_result/$task/RGBDACTv2/$config/$setting"
result="$(find "$result_dir" -type f -name _result.txt -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
test -n "$result"
test -s "$result"
score="$(tail -n 1 "$result")"
printf '%s COMPLETE success_rate=%s gpu=%s tier=%s rollouts=%s result=%s\n' \
  "$(date --iso-8601=seconds)" "$score" "$gpu" "$tier" "$rollouts" "$result" >"$status"
trap - ERR INT TERM
