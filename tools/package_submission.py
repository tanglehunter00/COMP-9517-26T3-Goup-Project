"""Build a deterministic, English-only, source-only COMP9517 submission ZIP.

Run from any directory: python tools/package_submission.py
Only explicitly listed files are eligible; dataset, weights and image artefacts
cannot enter through a recursive working-directory scan. Uses stdlib only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "COMP9517_Group104_Code"
MAX_BYTES = 25_000_000
REQUIRED = (
    "README.md", "SUBMISSION.md", "THIRD_PARTY.md",
    "INTEGRATION.md", "project.py", "tests/test_project_integration.py",
    "Yolo/original_pipeline.py", "Yolo/Readme.md", "Resnet/Readme.md",
    "Yolo Improeved/Readme.md", "Random Forest/Submit/README.md",
    "requirements-submission.txt", "requirements-5090d.txt",
    "requirements.txt", "requirements-legacy-3090.txt",
    "run_serial_benchmark.py", "controlled_diagnostics.py",
    "robustness_benchmark.py", "tools/package_submission.py",
    "Yolo/train.py", "Resnet/pipeline.py", "Resnet/test_pipeline.py",
    "Yolo Improeved/requirements.txt",
    "Yolo Improeved/vendor/__init__.py",
    "Yolo Improeved/vendor/dysample.py",
    "Yolo Improeved/vendor/SOURCES.md",
    "Yolo Improeved/vendor/SHA256.json",
    "Yolo Improeved/vendor/DYSAMPLE_LICENSE",
    "Yolo Improeved/vendor/ULTRALYTICS_LICENSE",
    "tools/reporting/analyze_p2_ablation.py",
    "tools/reporting/requirements.txt",
    "tools/reporting/tests/test_p2_ablation.py",
    "reports/P2_ONLY_ANALYSIS.md",
    "reports/MAIN_comparison.csv", "reports/MAIN_comparison.json",
    "reports/CONTROLLED_conditions.csv", "reports/CONTROLLED_summary.json",
    "reports/ROBUSTNESS_comparison.csv", "reports/ROBUSTNESS_summary.json",
)
PATTERNS = (
    "Random Forest/Submit/*.py", "Random Forest/Submit/requirements.txt",
    "Yolo Improeved/*.py", "Yolo Improeved/configs/*.yaml",
    "Yolo Improeved/tests/*.py", "reports/epoch_logs/*.csv",
    "Yolo/training/**/*.py", "Yolo/evaluation/*.py",
    "Resnet/Submit/*.py", "Resnet/Submit/requirements.txt",
)
FORBIDDEN_PARTS = {".git", ".venv", "__pycache__", "node_modules", "trained",
                   "checkpoints", "images", "runs", "results"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".joblib", ".pkl", ".whl", ".pyc",
                      ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".pdf", ".docx",
                      ".mp4", ".zip", ".gz"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def inputs(root):
    """Return validated archive-relative paths and unmodified source bytes."""
    names = set(REQUIRED)
    for pattern in PATTERNS:
        matches = list(root.glob(pattern))
        if not matches:
            raise ValueError(f"Required file pattern has no matches: {pattern}")
        names.update(path.relative_to(root).as_posix() for path in matches)
    items = {}
    for name in sorted(names):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or non-regular required source: {name}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source escapes project root: {name}")
        parts = set(Path(name).parts)
        if parts & FORBIDDEN_PARTS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Forbidden submission artefact: {name}")
        data = path.read_bytes()
        content = data.decode("utf-8-sig")
        if CJK.search(name) or CJK.search(content):
            raise ValueError(f"Non-English CJK text in source submission: {name}")
        if path.suffix == ".py":
            ast.parse(content, filename=name)
        items[name] = data
    pins = json.loads(items["Yolo Improeved/vendor/SHA256.json"])
    digest = pins["vendor\\dysample.py"]
    actual = hashlib.sha256(
        items["Yolo Improeved/vendor/dysample.py"].replace(b"\r\n", b"\n")
    ).hexdigest()
    if actual != digest:
        raise ValueError("Vendored DySample differs from its pinned source hash")
    return items


def build(root, output):
    items = inputs(root)
    manifest = {
        "schema": 1,
        "description": "English COMP9517 Group 104 source submission",
        "files": [
            {"path": name, "bytes": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in items.items()
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{NAME}.zip"
    # Build and validate before replacing the previous deliverable.
    with tempfile.TemporaryDirectory(prefix="comp9517-") as temporary:
        candidate = Path(temporary) / archive.name
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as packed:
            for name, data in sorted({**items, "MANIFEST.json": manifest_bytes}.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 18, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                packed.writestr(info, data, compresslevel=9)
        with zipfile.ZipFile(candidate) as packed:
            if packed.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
        size = candidate.stat().st_size
        if size > MAX_BYTES:
            raise ValueError(f"Archive exceeds source ZIP limit: {size} bytes")
        archive.write_bytes(candidate.read_bytes())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / f"{NAME}.sha256").write_bytes(
        f"{checksum}  {archive.name}\n".encode("utf-8"))
    (output / f"{NAME}.manifest.json").write_bytes(manifest_bytes)
    return archive, manifest, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "submission")
    args = parser.parse_args()
    archive, manifest, checksum = build(ROOT, args.out.resolve())
    print(f"{archive}: {archive.stat().st_size:,} bytes; "
          f"{len(manifest['files'])} source files")
    print(f"SHA256 {checksum}")


if __name__ == "__main__":
    main()
