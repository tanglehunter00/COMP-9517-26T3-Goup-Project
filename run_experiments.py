"""Resume the fixed experiments, then evaluate all selected best checkpoints.

Run from any directory: python run_experiments.py --root <official dataset>
Full checkpoints remain in ignored runs/; publishable inference-only weights and
auditable metrics are exported to trained/ and results/ after every run completes.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
SHARED = HERE/'Yolo Improeved'
sys.path.insert(0, str(SHARED))
from data import audit
from train import fingerprint


def execute(arguments, accelerated=True):
    env = os.environ.copy()
    env['COCO_BACKEND'] = 'faster' if accelerated else 'pycocotools'
    subprocess.run([sys.executable, *map(str, arguments)], cwd=HERE, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--runs-root', help='Optional checkpoint directory on a drive with sufficient free space')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    results = HERE/'results'; results.mkdir(exist_ok=True)
    report = dict(source='liugalhigh/beach-bird', audit=audit(root), fingerprint=fingerprint(root))
    (results/'dataset.json').write_text(json.dumps(report, indent=2))
    runs = {'m2': SHARED/'runs/m2', 'm4': SHARED/'runs/m4',
            'p2-only': SHARED/'runs/p2-only', 'm3': HERE/'Resnet/runs/m3'}
    if args.runs_root:
        storage = Path(args.runs_root).resolve(); storage.mkdir(parents=True, exist_ok=True)
        for method, source in list(runs.items()):
            destination = storage/method
            if source.exists() and not destination.exists():
                shutil.copytree(source, destination)
            runs[method] = destination
    # No test access until ALL training configurations have completed.
    for method, out in runs.items():
        last = out/'last.pt'
        checkpoint = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
        if checkpoint and (checkpoint['data_hash'] != report['fingerprint'] or checkpoint['smoke']):
            raise ValueError(f'{method}: existing checkpoint is for different data or a smoke run')
        if checkpoint and checkpoint['training_complete']:
            print(f'{method}: all 100 epochs already complete', flush=True)
            continue
        # The fixed retention policy needs about 8.3 GB for all full checkpoints.
        # Reserve the remaining archives and working copies before resuming.
        required = sum(max(0, 10-len(list(folder.glob('epoch*.pt')))) *
            (346_000_000 if name=='m3' else 114_000_000) for name,folder in runs.items())
        required += 1_500_000_000
        out.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(out).free < required:
            raise RuntimeError(f'Insufficient storage for fixed checkpoint retention: need {required} free bytes')
        command = ([HERE/'Resnet/pipeline.py', 'train'] if method=='m3' else [SHARED/'train.py'])
        command += ['--root', root, '--out', out]
        if method=='m2': command += ['--baseline']
        if method=='p2-only': command += ['--p2-only']
        if checkpoint: command += ['--resume', last]
        print(f'Training {method}', flush=True)
        execute(command)
    for method, out in runs.items():
        destination = results/method; destination.mkdir(exist_ok=True)
        for name in ['config.json', 'metrics.jsonl', 'transfer.json']:
            if (out/name).exists(): shutil.copy2(out/name, destination/name)
        for split in ['val', 'test']:
            csv = out/f'preds_{split}.csv'
            if method=='m3':
                command = [HERE/'Resnet/pipeline.py', 'infer']
            else:
                command = [SHARED/'infer.py']
            command += ['--root', root, '--weights', out/'best.pt', '--split', split, '--out', csv]
            if split=='test': command += ['--benchmark']
            # Reference evaluator for final published metrics (both architectures).
            execute(command, accelerated=False)
            execute([SHARED/'evaluate.py', '--root', root, '--split', split,
                     '--preds', csv, '--out', destination/f'{split}_metrics.json'], accelerated=False)
            if split=='test': shutil.copy2(csv.with_suffix('.timing.json'), destination/'timing.json')
            with csv.open('rb') as src, gzip.open(destination/f'preds_{split}.csv.gz', 'wb') as dst:
                shutil.copyfileobj(src, dst)
        selected = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        # Distribute compact inference weights; complete optimizer/RNG remains local.
        compact = {key:selected[key] for key in ['epoch','epochs','smoke','training_complete','data_hash']}
        compact.update({key:selected[key] for key in ['baseline','ablation'] if key in selected})
        compact['model'] = selected['model']
        compact['inference_only'] = True
        weights = HERE/'trained'/f'{method}-best.pt'; weights.parent.mkdir(exist_ok=True)
        torch.save(compact, weights)
        (destination/'weights.json').write_text(json.dumps(dict(file=weights.relative_to(HERE).as_posix(),
            sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), selected_epoch=selected['epoch']+1,
            precision='original FP32 checkpoint tensors; inference uses FP16 autocast',
            inference_only=True), indent=2))
    # Re-evaluate existing RF predictions with the same reference evaluator.
    destination = results/'m1'; destination.mkdir(exist_ok=True)
    for split in ['val','test']:
        execute([SHARED/'evaluate.py', '--root', root, '--split', split,
            '--preds', HERE/f'Random Forest/Submit/results/preds_{split}.csv',
            '--out', destination/f'{split}_metrics.json'], accelerated=False)
    print('All training and final evaluations complete.', flush=True)


if __name__=='__main__':
    main()
