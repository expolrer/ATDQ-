import json
import os
from pathlib import Path
import tempfile
import unittest

from atdq import ConfigError, Job, complete, load_config, matrix_contexts, parse_deadline, render


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_autodl_matrix_expands_to_36_unique_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "examples/autodl-matched-pi05.json", directory)
            self.assertEqual(len(config.jobs), 36)
            self.assertEqual(len({job.id for job in config.jobs}), 36)
            self.assertEqual({job.stage for job in config.jobs}, {"formal"})
            self.assertEqual(config.gpus, (0, 1, 2))
            self.assertTrue(all("{gpu}" in job.command for job in config.jobs))

    def test_server56_matrix_keeps_stage_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "examples/server56-rgbd-act-v2.json", directory)
            self.assertEqual(len(config.jobs), 30)
            self.assertEqual(sum(job.stage == "smoke" for job in config.jobs), 15)
            self.assertEqual(sum(job.stage == "full" for job in config.jobs), 15)
            self.assertEqual(config.stages, ("smoke", "full"))
            resolved = render(config.jobs[0].completion, config.jobs[0].values())
            self.assertIn("policy_step_001000.pt", str(resolved))

    def test_autodl_train_then_eval_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "examples/autodl-act3-train-eval.json", directory)
            self.assertEqual(len(config.jobs), 8)
            self.assertEqual(config.stages, ("train", "eval"))
            self.assertEqual(sum(job.stage == "train" for job in config.jobs), 2)
            self.assertEqual(sum(job.stage == "eval" for job in config.jobs), 6)
            self.assertTrue(all(job.depends_on == (f"ACT3_train_{job.context['train']}",)
                                for job in config.jobs if job.stage == "eval"))

    def test_missing_environment_variable_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.environ.pop("DEPTH_MODEL_ROOT", None)
            try:
                with self.assertRaisesRegex(ConfigError, "DEPTH_MODEL_ROOT"):
                    load_config(ROOT / "examples/autodl-matched-pi05.json")
            finally:
                if missing is not None:
                    os.environ["DEPTH_MODEL_ROOT"] = missing

    def test_matrix_option_adds_named_fields(self):
        options = matrix_contexts({"tier": [{"value": "smoke", "steps": 1000}],
                                   "task": ["stack", "mug"]})
        self.assertEqual(options, [
            {"tier": "smoke", "steps": 1000, "task": "stack"},
            {"tier": "smoke", "steps": 1000, "task": "mug"},
        ])


class CompletionTests(unittest.TestCase):
    def test_json_requires_full_unique_consistent_results(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "FORMAL_COMPLETE").write_text("done\n")
            path = folder / "final_result.json"
            job = Job(
                id="example", stage="formal", command=("true",), cwd=directory,
                env={}, requires=(), depends_on=(), gpu_allow=None, port_base=0,
                context={"state_dir": directory},
                completion={"files": ["{state_dir}/FORMAL_COMPLETE"],
                            "json": {"path": "{state_dir}/final_result.json",
                                     "equals": {"completed": 2}, "results_length": 2,
                                     "successes_match_results": True, "unique_seeds": True}},
            )
            path.write_text(json.dumps({"completed": 2, "successes": 1, "results": [
                {"seed": 1, "success": True}, {"seed": 2, "success": False}]}))
            self.assertTrue(complete(job))
            path.write_text(json.dumps({"completed": 2, "successes": 2, "results": [
                {"seed": 1, "success": True}, {"seed": 1, "success": False}]}))
            self.assertFalse(complete(job))

    def test_deadline_requires_explicit_timezone(self):
        with self.assertRaises(ConfigError):
            parse_deadline("2026-10-10T07:00:00")
        self.assertEqual(parse_deadline("2026-10-10T07:00:00+08:00").hour, 23)


if __name__ == "__main__":
    unittest.main()
