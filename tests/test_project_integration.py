"""CPU integration checks: real CLI, branch coverage and source-only hand-in."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cli(*arguments, cwd=None):
    return subprocess.run([sys.executable, str(ROOT / "project.py"), *arguments],
                          cwd=cwd, capture_output=True, text=True, check=True)


class ProjectIntegrationTests(unittest.TestCase):
    def test_help_and_every_canonical_method_without_ml_imports(self):
        self.assertIn("original-resnet", cli("--help").stdout)
        for method in ("m1", "m2", "m3", "p2-only", "m4"):
            for action in ("train", "infer", "evaluate"):
                result = cli(action, "--dry-run", "--method", method,
                             "--root", "dataset with spaces", "--run", "run with spaces")
                plan = json.loads(result.stdout)
                for step in plan:
                    self.assertTrue(Path(step["command"][2]).is_file())
                    self.assertEqual(step["cwd"], str(ROOT))
                if method == "m3" and action == "train":
                    command = plan[0]["command"]
                    self.assertEqual(command[command.index("--batch") + 1], "8")
                if action == "evaluate":
                    self.assertTrue(plan[0]["command"][2].endswith("evaluate.py"))

    def test_process_paths_are_independent_of_calling_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = json.loads(cli("--dry-run", "train", "--method", "m4",
                                  "--root", folder, cwd=folder).stdout)
            self.assertTrue(Path(plan[0]["command"][2]).is_absolute())
            self.assertEqual(plan[0]["cwd"], str(ROOT))
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_native_checkpoints_and_original_variants_stay_separate(self):
        for variant in ("protocol", "baseline"):
            command = json.loads(cli("original-yolo", "resume", "--dry-run",
                                     "--root", "dataset", "--variant", variant).stdout)[0]["command"]
            self.assertIn("original-yolo", command[command.index("--out") + 1])
            self.assertIn(variant, command)
        normal = json.loads(cli("original-resnet", "train", "--dry-run", "--root", "dataset").stdout)
        robust = json.loads(cli("original-resnet", "train", "--dry-run", "--root", "dataset", "--robust").stdout)
        self.assertNotEqual(normal[0]["command"], robust[0]["command"])
        self.assertIn("--robust", robust[0]["command"])
        for action in ("benchmark", "warp"):
            command = json.loads(cli("original-resnet", action, "--dry-run", "--root", "dataset").stdout)[0]["command"]
            self.assertTrue(Path(command[2]).is_file())
            self.assertIn("original-resnet", command[command.index("--ckpt") + 1])

    def test_serial_plan_and_invalid_rf_timing(self):
        command = json.loads(cli("--dry-run", "run", "--root", "dataset").stdout)[0]["command"]
        self.assertIn("--allow-other-cuda", command)
        failed = subprocess.run([sys.executable, str(ROOT / "project.py"), "infer",
                                 "--dry-run", "--method", "m1", "--root", "dataset",
                                 "--benchmark", "--workers", "2"], capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("RF timing requires", failed.stderr)

    def test_archive_covers_both_original_branches_and_is_reproducible(self):
        spec = importlib.util.spec_from_file_location("submission_builder", ROOT / "tools/package_submission.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            archive, manifest, first_hash = module.build(ROOT, Path(folder))
            with zipfile.ZipFile(archive) as packed:
                names = set(packed.namelist())
                for entry in manifest["files"]:
                    self.assertEqual(hashlib.sha256(packed.read(entry["path"])).hexdigest(), entry["sha256"])
                for path in (*ROOT.glob("Yolo/training/**/*.py"),
                             *ROOT.glob("Yolo/evaluation/*.py"),
                             *ROOT.glob("Resnet/Submit/*.py")):
                    self.assertIn(path.relative_to(ROOT).as_posix(), names)
                self.assertFalse(any(Path(name).suffix.lower() in module.FORBIDDEN_SUFFIXES for name in names))
                self.assertIn("project.py", names)
            _, _, second_hash = module.build(ROOT, Path(folder))
            self.assertEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
