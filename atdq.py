#!/usr/bin/env python3
"""Small, restartable GPU queue for existing RoboTwin training/evaluation workers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    pass


def env_expand(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"Set environment variable {name} before loading the config")
        return os.environ[name]

    return ENV_PATTERN.sub(replace, value)


def render(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return env_expand(value).format_map(context)
        except KeyError as error:
            raise ConfigError(f"Unknown placeholder {error} in {value!r}") from error
    if isinstance(value, list):
        return [render(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render(item, context) for key, item in value.items()}
    return value


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


@dataclass(frozen=True)
class Job:
    id: str
    stage: str
    command: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    requires: tuple[str, ...]
    completion: dict[str, Any]
    depends_on: tuple[str, ...]
    gpu_allow: tuple[int, ...] | None
    port_base: int
    context: dict[str, Any]

    def values(self, gpu: int = 0) -> dict[str, Any]:
        return {**self.context, "gpu": gpu, "port": self.port_base + gpu}


@dataclass(frozen=True)
class Config:
    source: Path
    name: str
    project_root: Path
    state_dir: Path
    lock_dir: Path
    gpus: tuple[int, ...]
    stages: tuple[str, ...]
    poll_seconds: int
    max_used_memory_mib: int
    max_attempts: int
    retry_delay_seconds: int
    jobs: tuple[Job, ...]
    fingerprint: str


def matrix_contexts(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    contexts = []
    for combination in itertools.product(*(matrix[key] for key in matrix)):
        context: dict[str, Any] = {}
        for key, choice in zip(matrix, combination):
            if isinstance(choice, dict):
                if "value" not in choice:
                    raise ConfigError(f"Matrix option {key} requires a value")
                extras = set(choice) - {"value"}
                if key in context or extras & context.keys() or key in extras:
                    raise ConfigError(f"Duplicate matrix placeholder in {key}")
                context[key] = choice["value"]
                context.update({name: choice[name] for name in extras})
            else:
                if key in context:
                    raise ConfigError(f"Duplicate matrix key {key}")
                context[key] = choice
        contexts.append(context)
    return contexts


def load_config(path: Path, project_root_override: str | None = None) -> Config:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ConfigError("Config version must be 1")
    root_text = project_root_override or env_expand(raw["project_root"])
    project_root = Path(root_text).expanduser().resolve()
    state_text = env_expand(raw.get("state_dir", f"../.state/{raw['name']}"))
    state_path = Path(state_text).expanduser()
    state_dir = (state_path if state_path.is_absolute() else path.parent / state_path).resolve()
    lock_dir = Path(env_expand(raw.get("gpu_lock_dir", "/tmp/atdq-gpu-locks"))).resolve()
    gpus = tuple(raw["gpus"])
    stages = tuple(raw["stages"])
    if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ConfigError("gpus must be unique nonnegative physical GPU indices")
    if not stages or len(set(stages)) != len(stages):
        raise ConfigError("stages must be a nonempty ordered list without duplicates")
    defaults = {
        "project_root": str(project_root),
        "repo_dir": str(Path(__file__).resolve().parent),
        "state_dir": str(state_dir),
    }
    jobs = []
    for template in raw["templates"]:
        for options in matrix_contexts(template.get("matrix", {})):
            context = {**defaults, **options}
            job_id = render(template["id"], context)
            if not ID_PATTERN.fullmatch(job_id):
                raise ConfigError(f"Invalid job ID {job_id!r}")
            context["job_id"] = job_id
            stage = render(template["stage"], context)
            if stage not in stages:
                raise ConfigError(f"Unknown stage {stage} in {job_id}")
            command = tuple(template["command"])
            if not command or any(not isinstance(arg, str) for arg in command):
                raise ConfigError(f"{job_id}: command must be an argv string list")
            allowed = template.get("gpu_allow")
            if allowed is not None and (not isinstance(allowed, list) or any(type(g) is not int for g in allowed)):
                raise ConfigError(f"{job_id}: gpu_allow must be a GPU index list")
            job = Job(
                id=job_id,
                stage=stage,
                command=command,
                cwd=template.get("cwd", "{project_root}"),
                env=template.get("env", {}),
                requires=tuple(template.get("requires", [])),
                completion=template["completion"],
                depends_on=tuple(render(template.get("depends_on", []), context)),
                gpu_allow=tuple(allowed) if allowed is not None else None,
                port_base=int(template.get("port_base", 0)),
                context=context,
            )
            # Catch typos in commands before GPUs become available.
            render(list(job.command), job.values())
            render(job.cwd, job.values())
            render(job.env, job.values())
            render(list(job.requires), job.values())
            render(job.completion, job.values())
            jobs.append(job)
    ids = [job.id for job in jobs]
    if len(ids) != len(set(ids)):
        raise ConfigError("Job IDs must be unique after matrix expansion")
    for job in jobs:
        if job.id in job.depends_on or any(dep not in ids for dep in job.depends_on):
            raise ConfigError(f"{job.id}: unknown or self dependency")
    fingerprint = hashlib.sha256(json.dumps(
        {"raw": raw, "project_root": str(project_root), "state_dir": str(state_dir)},
        sort_keys=True,
    ).encode()).hexdigest()
    return Config(
        source=path, name=raw["name"], project_root=project_root,
        state_dir=state_dir, lock_dir=lock_dir, gpus=gpus, stages=stages,
        poll_seconds=max(1, int(raw.get("poll_seconds", 30))),
        max_used_memory_mib=int(raw.get("max_used_memory_mib", 600)),
        max_attempts=max(1, int(raw.get("max_attempts", 3))),
        retry_delay_seconds=max(0, int(raw.get("retry_delay_seconds", 180))),
        jobs=tuple(jobs), fingerprint=fingerprint,
    )


def job_folder(config: Config, job: Job) -> Path:
    return config.state_dir / "jobs" / job.id


def complete(job: Job) -> bool:
    spec = render(job.completion, job.values())
    if any(not Path(path).is_file() for path in spec.get("files", [])):
        return False
    if any(not Path(path).is_file() or Path(path).stat().st_size == 0
           for path in spec.get("nonempty_files", [])):
        return False
    for item in spec.get("regex", []):
        path = Path(item["path"])
        if not path.is_file() or re.search(item["pattern"], path.read_text(errors="replace")) is None:
            return False
    if "json" in spec:
        item = spec["json"]
        path = Path(item["path"])
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if any(data.get(key) != value for key, value in item.get("equals", {}).items()):
            return False
        if "results_length" in item:
            results = data.get("results")
            if not isinstance(results, list) or len(results) != item["results_length"]:
                return False
            if not all(isinstance(result, dict) and isinstance(result.get("success"), bool)
                       for result in results):
                return False
            if item.get("successes_match_results") and data.get("successes") != sum(
                result["success"] for result in results
            ):
                return False
            if item.get("unique_seeds") and len({result.get("seed") for result in results}) != len(results):
                return False
    return True


def missing_requirements(job: Job) -> list[str]:
    return [path for path in render(list(job.requires), job.values()) if not Path(path).exists()]


def read_attempts(config: Config, job: Job) -> dict[str, Any]:
    path = job_folder(config, job) / "attempts.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"count": 0, "next_retry_at": 0}


def gpu_idle(gpu: int, max_used_memory_mib: int) -> bool:
    try:
        memory = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
        pids = subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
        return int(memory.strip().split()[0]) <= max_used_memory_mib and not pids.strip()
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return False


def parse_deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ConfigError("--until requires an explicit timezone, e.g. +08:00")
    return result.astimezone(timezone.utc)


def write_status(config: Config, active: dict[int, tuple[Job, subprocess.Popen, Any, Any]],
                 temporary: tuple[int, ...], deadline: datetime | None) -> dict[str, Any]:
    running = {job.id: gpu for gpu, (job, _, _, _) in active.items()}
    jobs = {}
    for job in config.jobs:
        folder = job_folder(config, job)
        if job.id in running:
            status = f"running:gpu{running[job.id]}"
        elif complete(job):
            status = "complete"
        elif (folder / "NEEDS_ATTENTION").exists():
            status = "needs_attention"
        elif missing := missing_requirements(job):
            status = f"waiting:required:{missing[0]}"
        elif any(not complete(next(item for item in config.jobs if item.id == dep)) for dep in job.depends_on):
            status = "waiting:dependency"
        elif job.stage != next((stage for stage in config.stages if any(
                (item.id in running or not complete(item)) and item.stage == stage
                for item in config.jobs)), None):
            status = "waiting:stage"
        elif read_attempts(config, job)["next_retry_at"] > time.time():
            status = "waiting:retry"
        else:
            status = "pending"
        jobs[job.id] = status
    report = {"updated_at": datetime.now(timezone.utc).isoformat(),
              "name": config.name, "fingerprint": config.fingerprint,
              "running": running, "jobs": jobs,
              "temporary_gpus": list(temporary),
              "temporary_until": deadline.isoformat() if deadline else None}
    atomic_json(config.state_dir / "status.json", report)
    return report


def ensure_manifest(config: Config) -> None:
    manifest = config.state_dir / "manifest.json"
    if manifest.is_file():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        if recorded.get("fingerprint") != config.fingerprint:
            raise ConfigError("State belongs to another config; use a new state_dir")
    else:
        atomic_json(manifest, {"name": config.name, "fingerprint": config.fingerprint,
                               "source": str(config.source), "jobs": [job.id for job in config.jobs]})


def terminate_group(process: subprocess.Popen, grace_seconds: int = 30) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run(config: Config, temporary: tuple[int, ...], deadline: datetime | None) -> None:
    if os.name != "posix":
        raise RuntimeError("GPU scheduling requires Linux (fcntl and process groups)")
    import fcntl

    if temporary and deadline is None:
        raise ConfigError("--temporary-gpus requires --until")
    if set(temporary) & set(config.gpus):
        raise ConfigError("Temporary GPUs must not overlap permanent GPUs")
    if deadline and deadline <= datetime.now(timezone.utc):
        raise ConfigError("The temporary GPU deadline has already passed")
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.lock_dir.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[Job, subprocess.Popen, Any, Any]] = {}
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with (config.state_dir / "queue.lock").open("a+") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_manifest(config)
        try:
            while not stop:
                for gpu, (job, process, stream, gpu_lock) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                    gpu_lock.close()
                    del active[gpu]
                    if code == 0 and complete(job):
                        print(f"COMPLETE job={job.id} gpu={gpu}", flush=True)
                        continue
                    if code == 75:
                        attempts = read_attempts(config, job)
                        attempts["next_retry_at"] = time.time() + 60
                    else:
                        attempts = read_attempts(config, job)
                        attempts["count"] += 1
                        attempts["next_retry_at"] = time.time() + config.retry_delay_seconds
                        if attempts["count"] >= config.max_attempts:
                            (job_folder(config, job) / "NEEDS_ATTENTION").write_text(
                                f"exit={code}; completion_verified={complete(job)}\n", encoding="utf-8")
                    atomic_json(job_folder(config, job) / "attempts.json", attempts)
                    print(f"FAILED job={job.id} gpu={gpu} exit={code}", flush=True)

                if deadline and datetime.now(timezone.utc) >= deadline:
                    for gpu in temporary:
                        if gpu in active:
                            job, process, stream, gpu_lock = active.pop(gpu)
                            terminate_group(process)
                            stream.close()
                            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                            gpu_lock.close()
                            print(f"DEADLINE_PREEMPT job={job.id} gpu={gpu}", flush=True)
                    temporary = ()

                report = write_status(config, active, temporary, deadline)
                if all(state == "complete" for state in report["jobs"].values()):
                    (config.state_dir / "ALL_COMPLETE").write_text(report["updated_at"] + "\n", encoding="utf-8")
                    break
                if not (config.state_dir / "PAUSED").exists():
                    running_ids = {job.id for job, _, _, _ in active.values()}
                    stage = next((stage for stage in config.stages if any(
                        (job.id in running_ids or not complete(job)) and job.stage == stage
                        for job in config.jobs)), None)
                    for gpu in (*config.gpus, *temporary):
                        if gpu in active:
                            continue
                        lock_path = config.lock_dir / f"gpu{gpu}.lock"
                        gpu_lock = lock_path.open("a+")
                        try:
                            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            gpu_lock.close()
                            continue
                        if not gpu_idle(gpu, config.max_used_memory_mib):
                            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                            gpu_lock.close()
                            continue
                        job = next((item for item in config.jobs if item.stage == stage
                            and (item.gpu_allow is None or gpu in item.gpu_allow)
                            and not complete(item)
                            and not (job_folder(config, item) / "NEEDS_ATTENTION").exists()
                            and not missing_requirements(item)
                            and all(complete(next(dep for dep in config.jobs if dep.id == name))
                                    for name in item.depends_on)
                            and read_attempts(config, item)["next_retry_at"] <= time.time()
                            and item.id not in (running.id for running, _, _, _ in active.values())), None)
                        if job is None:
                            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                            gpu_lock.close()
                            continue
                        folder = job_folder(config, job)
                        folder.mkdir(parents=True, exist_ok=True)
                        stream = (folder / "worker.log").open("a", encoding="utf-8")
                        values = job.values(gpu)
                        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                               **render(job.env, values)}
                        command = render(list(job.command), values)
                        try:
                            process = subprocess.Popen(command, cwd=render(job.cwd, values),
                                                       env=env, stdout=stream,
                                                       stderr=subprocess.STDOUT, start_new_session=True)
                        except OSError:
                            stream.close()
                            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                            gpu_lock.close()
                            raise
                        active[gpu] = (job, process, stream, gpu_lock)
                        print(f"START job={job.id} gpu={gpu} pid={process.pid}", flush=True)
                sleep_seconds = config.poll_seconds
                if temporary and deadline:
                    sleep_seconds = min(sleep_seconds, max(0.1, (
                        deadline - datetime.now(timezone.utc)).total_seconds()))
                time.sleep(sleep_seconds)
        finally:
            for gpu, (job, process, stream, gpu_lock) in active.items():
                terminate_group(process)
                stream.close()
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                gpu_lock.close()
                print(f"STOPPED job={job.id} gpu={gpu}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "status", "pause", "resume", "retry"))
    parser.add_argument("config", type=Path)
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("--project-root", help="Override project_root from the config")
    parser.add_argument("--temporary-gpus", default="", help="Comma-separated extra GPU indices")
    parser.add_argument("--until", help="ISO 8601 deadline with timezone for temporary GPUs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.project_root)
        if args.action == "validate" or args.dry_run:
            print(f"VALID name={config.name} jobs={len(config.jobs)} stages={','.join(config.stages)}")
            for job in config.jobs:
                print(f"{job.stage}\t{job.id}\tready={not missing_requirements(job)}\tcomplete={complete(job)}")
            return 0
        if args.action == "status":
            path = config.state_dir / "status.json"
            print(path.read_text(encoding="utf-8") if path.is_file() else "No status yet")
            return 0
        if args.action == "pause":
            config.state_dir.mkdir(parents=True, exist_ok=True)
            (config.state_dir / "PAUSED").touch()
            print("PAUSED: no new jobs will start; active jobs continue")
            return 0
        if args.action == "resume":
            (config.state_dir / "PAUSED").unlink(missing_ok=True)
            print("RESUMED")
            return 0
        if args.action == "retry":
            if not args.job_id or args.job_id not in {job.id for job in config.jobs}:
                raise ConfigError("retry requires a valid job_id")
            report_path = config.state_dir / "status.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if args.job_id in report.get("running", {}):
                    raise ConfigError("Cannot retry a job reported as running")
            folder = config.state_dir / "jobs" / args.job_id
            (folder / "NEEDS_ATTENTION").unlink(missing_ok=True)
            (folder / "attempts.json").unlink(missing_ok=True)
            print(f"RETRY_ENABLED job={args.job_id}")
            return 0
        temporary = tuple(int(value) for value in args.temporary_gpus.split(",") if value)
        run(config, temporary, parse_deadline(args.until))
        return 0
    except (ConfigError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ATDQ ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
