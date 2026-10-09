# ATDQ-

Auto Training and Dynamic Queue for existing RoboTwin experiments. This repository
contains **scheduling and small workload adapters**, not model weights, datasets,
RoboTwin itself, or a replacement for a model-specific evaluation validity check.

The queue consolidates the patterns used on AutoDL and the 56 server: assign work
only to idle physical GPUs, keep training/evaluation stages ordered, resume after
restarting the coordinator, retry bounded failures, and preserve per-job logs.

## What it does

- Expands a JSON matrix into jobs; no YAML dependency or shell command interpolation.
- Checks `nvidia-smi` compute PIDs and memory, plus a cross-ATDQ GPU lock.
- Waits for required files, dependencies, and an entire preceding stage to finish.
- Accepts completion only when configured checkpoint/result checks pass. Exit 0 alone
  never counts as success.
- Saves attempts and `NEEDS_ATTENTION` markers. `PAUSED` prevents new launches;
  running jobs continue. A timezone-qualified deadline can temporarily add GPUs
  and then stop **only ATDQ-owned** children on those GPUs.
- Refuses to reuse a state directory after a config/project-root change.

## Requirements

Linux, Python 3.10+, `nvidia-smi`, `bash`, and `flock`. The example workloads also
require the existing model repositories, Python environments, checkpoints, seed
lists, and RoboTwin assets at `DEPTH_MODEL_ROOT`. No secrets belong in JSON configs.

```bash
git clone https://github.com/expolrer/ATDQ-.git
cd ATDQ-
python3 -m unittest discover -s tests -v
```

### AutoDL: fixed-seed pi0.5 matrix

The example wraps the already installed `stack_matched_eval_20261008` workers: two
architectures x three training depths x two scene modes x three deployment depths
= 36 independently resumable jobs. It does **not** download weights or generate
valid seeds. On the machine with the existing pipeline:

```bash
export DEPTH_MODEL_ROOT=/root/autodl-tmp/depth-model
python3 atdq.py validate examples/autodl-matched-pi05.json
python3 atdq.py run examples/autodl-matched-pi05.json --dry-run
python3 atdq.py run examples/autodl-matched-pi05.json
python3 atdq.py status examples/autodl-matched-pi05.json
```

The upstream AutoDL worker owns checkpoint loading, the fixed seed manifest,
prompt forwarding, D0/D1/D3 processing, first-action movement checks, per-seed
progress, and `FORMAL_COMPLETE`. ATDQ additionally verifies that the final JSON
has 100 unique seeds and a consistent success count. The worker and this queue
should **not** both be started under separate supervisors for the same batch.

### AutoDL: ACT3 resume training then hanging_mug evaluation

`examples/autodl-act3-train-eval.json` wraps the existing workers for D0/D1 ACT3
training at 6000 epochs and their six Easy D0/D1/D3 evaluations. It requires the
AutoDL `D1_READY` marker and the prepared 100-seed manifest; it does not create
a dataset or seed list. The train stage finishes before evaluation starts, and
the worker's `--resume` restores `training_last.pt` if interrupted.

```bash
export DEPTH_MODEL_ROOT=/root/autodl-tmp/depth-model
python3 atdq.py validate examples/autodl-act3-train-eval.json
python3 atdq.py run examples/autodl-act3-train-eval.json --dry-run
python3 atdq.py run examples/autodl-act3-train-eval.json
```

If the original AutoDL supervisor is still running this batch, stop that
supervisor before launching ATDQ; this repository does not control it.

### 56 server: RGBD-ACT-v2 training then RoboTwin evaluation

The sample expands five variants x three tasks x two tiers = 30 jobs. All 15 smoke
jobs must complete before full jobs start. Each job trains or resumes its PyTorch
checkpoint and then runs the configured RoboTwin evaluation.

```bash
export DEPTH_MODEL_ROOT=/ssd/hhw/depth-model
python3 atdq.py validate examples/server56-rgbd-act-v2.json
python3 atdq.py run examples/server56-rgbd-act-v2.json --dry-run
python3 atdq.py run examples/server56-rgbd-act-v2.json
```

To temporarily add GPU2/3, give a **future** Beijing-time deadline. These GPUs
are preempted at the deadline; GPU4-7 continue:

```bash
python3 atdq.py run examples/server56-rgbd-act-v2.json \
  --temporary-gpus 2,3 --until '2026-10-10T07:00:00+08:00'
```

The 56 adapter ports the observed `run_v2_train.sh`/`run_v2_eval.sh` arguments to
`DEPTH_MODEL_ROOT` instead of embedding `/ssd/hhw/...` in executable code. The
trainer defaults to `--resume` from `policy_last.pt`. The RoboTwin evaluator in
this example does **not** guarantee per-seed resume: if interrupted, that cell may
restart its evaluation. Verify 100 valid seeds, correct prompt/depth, Mplib TOPP,
and arm motion before reporting a scientific success rate. A result file alone
cannot prove the controller executed every predicted action.

## Operations

```bash
python3 atdq.py pause  examples/autodl-matched-pi05.json
python3 atdq.py resume examples/autodl-matched-pi05.json
python3 atdq.py retry  examples/autodl-matched-pi05.json JOB_ID
python3 atdq.py status examples/autodl-matched-pi05.json
```

`pause` does not kill a running job. A SIGTERM to ATDQ stops its own child process
groups and leaves their worker progress files for a later launch. When a job has
exhausted retries, inspect `state/jobs/JOB_ID/worker.log` and the upstream worker
state before issuing `retry`. On a changed config, choose a new `state_dir`; do not
delete old results to make the queue start.

`validate` and `--dry-run` never allocate a GPU or launch a worker. The state
directory defaults to `.state/` in this repository and is gitignored. The GPU
lock directory is `/tmp/atdq-gpu-locks`; other schedulers that do not honor it
are detected by `nvidia-smi` but cannot be made race-free by ATDQ alone.

## Scope and provenance

[Source mapping and protocol caveats](docs/ORIGINS.md) explain which production
scripts were inspected and why the older A3 `ROBOTWIN_SKIP_PLANNERS` setting is
not carried into this repository. The JSON files are **examples**: edit GPU IDs,
job matrix, checkpoints, seed configuration, and state path for a new deployment.
