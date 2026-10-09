# Queue source mapping

This is a refactor of scheduling behavior observed on the two servers, not a
verbatim export of their work directories.

| Host | Inspected source | Behavior retained |
| --- | --- | --- |
| AutoDL | `depth-model/evaluations/stack_matched_eval_20261008/bin/dynamic_eval_queue.py` | idle-GPU dispatch, fixed matrix, result markers, bounded retry, status JSON |
| AutoDL | `.../bin/pi05_dpv_worker.sh`, `pi05_rgbd4_worker.sh`, `validate_run.py` | delegated model-specific smoke/formal evaluation and per-seed validation |
| AutoDL | `depth-model/scheduler/hanging_mug_act3_autodl_workers2_20260929/bin/*` and `hanging_mug_act3_new_depth_matrix_20260930/bin/*` | dependency/readiness and training resume patterns |
| 56 | `depth-model/scripts/continuous_rebalanced_worker.sh` and `continuous_rebalanced_supervisor.sh` | shared work, temporary GPU deadline, training followed by evaluation |
| 56 | `depth-model/scripts/rgbd_act_v2/continuous_v2_worker.sh` and `continuous_v2_supervisor.sh` | smoke-to-full stage barrier across five variants and three tasks |
| 56 | `depth-model/scripts/rgbd_act_v2/run_v2_train.sh` and `run_v2_eval.sh` | PyTorch resume checkpoint and RoboTwin invocation adapted in `adapters/robotwin_act_v2_pipeline.sh` |

Absolute server paths in the source are example configuration values here. The
ATDQ executable itself uses the repository location and `DEPTH_MODEL_ROOT`.
No private keys, passwords, models, datasets, environment directories, or saved
queue state are included.

## Evaluation validity boundary

ATDQ determines whether a **worker** finished and whether its configured result
artifacts are internally consistent. It does not infer that a policy is good or
that a simulator executed all actions. The old 56 A3 evaluation used
`ROBOTWIN_SKIP_PLANNERS=1`; this could suppress the Mplib TOPP planner along with
Curobo and produce a task-failure score without robot motion. ATDQ neither
imports that switch nor treats a zero-success result as proof of a valid run.

For pi0.5 on AutoDL, the upstream matched-eval worker validates seed order,
prompt identity, depth variant, policy RNG and initial arm motion. For RGBD-ACT-v2
on 56, add a model-specific movement/TOPP audit if the results will be used for
formal comparison. Do not mix these example protocols in a single architecture
ranking.
