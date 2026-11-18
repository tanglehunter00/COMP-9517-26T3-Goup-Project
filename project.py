"""Unified, dependency-light command line for the complete bird-detection project.

Use --dry-run to print the exact subprocess plan without importing ML libraries,
reading the dataset, creating outputs, or starting training. Every implementation
runs in a separate process to isolate modules named model, data and evaluate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "Yolo Improeved"
RF = ROOT / "Random Forest" / "Submit"
METHODS = ("m1", "m2", "m3", "p2-only", "m4")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def command(name, help_text, dataset=True, run=False):
        q = sub.add_parser(name, help=help_text)
        q.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
        if dataset:
            q.add_argument("--root", required=True, type=Path,
                           help="Official dataset folder containing images/ and labels/")
        if run:
            q.add_argument("--run", type=Path, default=ROOT / "runs/integrated_seed42")
        return q

    command("assets", "Download and verify pinned initialization assets", dataset=False)
    command("audit", "Validate official split counts, labels and exact duplicates")
    q = command("train", "Train one matched-comparison configuration", run=True)
    q.add_argument("--method", choices=METHODS, required=True)
    q.add_argument("--resume", action="store_true", help="Resume a deep model's last.pt")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--workers", type=int, default=2, help="RF sample/mining workers")
    q = command("infer", "Export original-coordinate prediction CSV", run=True)
    q.add_argument("--method", choices=METHODS, required=True)
    q.add_argument("--split", choices=("val", "test"), default="val")
    q.add_argument("--weights", type=Path)
    q.add_argument("--out", type=Path, help="CSV path; defaults to the method run folder")
    q.add_argument("--benchmark", action="store_true")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--workers", type=int, default=1, help="RF image workers; use 1 for timing")
    q = command("evaluate", "Run the common COCO and operating-point evaluator", run=True)
    q.add_argument("--method", choices=METHODS, required=True)
    q.add_argument("--split", choices=("val", "test"), default="val")
    q.add_argument("--preds", type=Path)
    q.add_argument("--out", type=Path, help="Metrics JSON path")
    command("run", "Train all five methods, then evaluate and time them serially", run=True)
    command("diagnostics", "Run fixed-checkpoint validation interventions", run=True)
    q = command("robustness", "Run paired validation corruptions for all five methods", run=True)
    q.add_argument("--workers", type=int, default=8, help="RF image workers; not latency timing")
    q = command("p2-analysis", "Analyse saved predictions with paired image bootstrap", run=True)
    q.add_argument("--split", choices=("val", "test"), default="val")
    q.add_argument("--out", required=True, type=Path)
    q.add_argument("--label", required=True, help="Explicit experiment provenance")
    q.add_argument("--samples", type=int, default=5000)
    q.add_argument("--seed", type=int, default=20261118)
    q = command("original-yolo", "Portable adapter for the original YOLO branch", run=True)
    q.add_argument("action", choices=("train", "resume", "evaluate"))
    q.add_argument("--variant", choices=("protocol", "baseline"), default="protocol")
    q.add_argument("--weights", type=Path)
    q.add_argument("--split", choices=("val", "test"), default="val")
    q.add_argument("--device", default="0")
    q = command("original-resnet", "Original Faster R-CNN / robust / swirl experiments", run=True)
    q.add_argument("action", choices=("train", "benchmark", "warp"))
    q.add_argument("--robust", action="store_true", help="Separate non-protocol augmentation variant")
    q.add_argument("--weights", type=Path)
    q.add_argument("--split", choices=("val", "test"), default="val")
    return p


def build_plan(a):
    steps = []

    def add(script, *arguments, label=None, cwd=ROOT):
        script = Path(script)
        steps.append(dict(label=label or script.name, cwd=str(cwd),
                          command=[sys.executable, "-u", str(script), *map(str, arguments)]))

    dataset = a.root.resolve() if hasattr(a, "root") else None
    run = a.run.resolve() if hasattr(a, "run") else None
    folder = run / a.method if hasattr(a, "method") else None
    if a.command == "assets":
        add(SHARED / "download_assets.py")
    elif a.command == "audit":
        add(SHARED / "data.py", "--root", dataset)
    elif a.command == "train":
        if a.method == "m1":
            if a.resume:
                raise ValueError("--resume applies to deep last.pt checkpoints, not RF")
            add(RF / "build_samples.py", "--root", dataset, "--workers", a.workers,
                "--out", folder / "samples_round1.npz")
            add(RF / "train_rf.py", "--root", dataset, "--workers", a.workers,
                "--samples", folder / "samples_round1.npz", "--out_dir", folder / "models")
        else:
            args = ["--root", dataset, "--out", folder, "--device", a.device]
            if a.resume:
                args += ["--resume", folder / "last.pt"]
            if a.method == "m3":
                add(ROOT / "Resnet/pipeline.py", "train", "--batch", 8, *args)
            else:
                if a.method == "m2":
                    args += ["--baseline"]
                if a.method == "p2-only":
                    args += ["--p2-only"]
                add(SHARED / "train.py", *args)
    elif a.command == "infer":
        out = a.out.resolve() if a.out else folder / f"preds_{a.split}.csv"
        weights = a.weights.resolve() if a.weights else (
            folder / "models/rf_final.joblib" if a.method == "m1" else folder / "best.pt")
        args = ["--root", dataset, "--split", a.split, "--out", out]
        if a.method == "m1":
            if a.benchmark and a.workers != 1:
                raise ValueError("RF timing requires --workers 1")
            add(RF / "infer.py", *args, "--model", weights, "--workers", a.workers,
                "--repeats", 3 if a.benchmark else 1, "--warmup", 20 if a.benchmark else 0)
        else:
            args += ["--weights", weights, "--device", a.device]
            if a.benchmark:
                args += ["--benchmark"]
            if a.method == "m3":
                add(ROOT / "Resnet/pipeline.py", "infer", *args)
            else:
                add(SHARED / "infer.py", *args)
    elif a.command == "evaluate":
        preds = a.preds.resolve() if a.preds else folder / f"preds_{a.split}.csv"
        out = a.out.resolve() if a.out else folder / f"{a.split}_metrics.json"
        add(SHARED / "evaluate.py", "--root", dataset, "--split", a.split,
            "--preds", preds, "--out", out)
    elif a.command == "run":
        add(ROOT / "run_serial_benchmark.py", "--root", dataset, "--out", run,
            "--allow-other-cuda")
    elif a.command in ("diagnostics", "robustness"):
        args = ["--root", dataset, "--run", run]
        if a.command == "robustness":
            args += ["--rf-workers", a.workers]
        add(ROOT / ("controlled_diagnostics.py" if a.command == "diagnostics"
                    else "robustness_benchmark.py"), *args)
    elif a.command == "p2-analysis":
        add(ROOT / "tools/reporting/analyze_p2_ablation.py", "--root", dataset,
            "--results", run, "--split", a.split, "--out", a.out.resolve(),
            "--label", a.label, "--samples", a.samples, "--seed", a.seed)
    elif a.command == "original-yolo":
        args = [a.action, "--root", dataset, "--out", run / "original-yolo" / a.variant,
                "--variant", a.variant, "--split", a.split, "--device", a.device]
        if a.weights:
            args += ["--weights", a.weights.resolve()]
        add(ROOT / "Yolo/original_pipeline.py", *args)
    elif a.command == "original-resnet":
        base = ROOT / "Resnet/Submit"
        original = run / ("original-resnet-robust" if a.robust else "original-resnet")
        if a.action == "train":
            args = ["--data", dataset, "--out", original]
            if a.robust:
                args += ["--robust"]
            add(base / "train.py", *args)
        else:
            weights = a.weights.resolve() if a.weights else original / "best.pt"
            out = original / ("benchmark.json" if a.action == "benchmark" else "warp")
            add(base / ("benchmark.py" if a.action == "benchmark" else "warp_eval.py"),
                "--data", dataset, "--ckpt", weights, "--out", out, "--split", a.split)
    else:
        raise ValueError(a.command)
    return steps


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    for field in ("workers", "samples"):
        if hasattr(a, field) and getattr(a, field) <= 0:
            p.error(f"--{field} must be positive")
    try:
        plan = build_plan(a)
    except ValueError as error:
        p.error(str(error))
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return
    # Explicit paths and process isolation avoid cwd-sensitive sibling imports.
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if a.command in ("train", "infer", "evaluate"):
        (a.run.resolve() / a.method).mkdir(parents=True, exist_ok=True)
        if getattr(a, "out", None):
            a.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    if a.command == "original-resnet":
        (a.run.resolve() / ("original-resnet-robust" if a.robust else "original-resnet")).mkdir(parents=True, exist_ok=True)
    for step in plan:
        print(f"Starting: {step['label']}", flush=True)
        subprocess.run(step["command"], cwd=step["cwd"], env=env, check=True)


if __name__ == "__main__":
    main()
