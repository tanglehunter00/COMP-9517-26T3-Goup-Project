"""One-child-at-a-time, resumable RTX 5090 D benchmark with equal deep batches.

Run using the method's virtual environment. All fresh results remain in --out.
No historical results/trained paths are overwritten. A Windows file lock prevents
two queue owners, and every subprocess is waited for before another starts.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

import psutil
import torch

HERE = Path(__file__).resolve().parent
SHARED = HERE / 'Yolo Improeved'
RF = HERE / 'Random Forest/Submit'
sys.path.insert(0, str(SHARED))
from data import audit
from train import fingerprint

DEEP = ['m2', 'm3', 'p2-only', 'm4']
LOSS_COLUMNS = ['loss_box', 'loss_cls', 'loss_dfl', 'loss_classifier',
                'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def records(path):
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    if text and not text.endswith('\n'):
        lines = lines[:-1]  # An epoch line can still be in the middle of a write.
    return [json.loads(line) for line in lines if line.strip()]


def export_epochs(folder):
    metrics = records(folder / 'metrics.jsonl')
    if not metrics:
        return
    timing = {row['epoch']: row for row in records(folder / 'epoch_timing.jsonl')}
    rows = []
    for metric in metrics:
        parts = metric['losses']
        if isinstance(parts, list):
            parts = dict(zip(['box', 'cls', 'dfl'], parts, strict=True))
        loss = {key if key.startswith('loss_') else 'loss_' + key: value for key, value in parts.items()}
        row = dict(epoch=metric['epoch'], loss_total=sum(parts.values()), **loss,
                   training_s=metric['training_s'], validation_s=metric['validation_s'],
                   AP50=metric['AP50'], AP50_95=metric['AP50:95'],
                   lr=metric['lr'], updates=metric['updates'])
        row.update({key: timing[metric['epoch']][key] for key in ['checkpoint_s', 'epoch_wall_s']}
                   if metric['epoch'] in timing else {})
        rows.append(row)
    fields = ['epoch', 'loss_total', *LOSS_COLUMNS, 'training_s', 'validation_s',
              'checkpoint_s', 'epoch_wall_s', 'AP50', 'AP50_95', 'lr', 'updates']
    temporary = folder / 'epochs.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, folder / 'epochs.csv')


class SerialQueue:
    def __init__(self, root, out, allow_other_cuda=False):
        self.root, self.out = root, out
        self.allow_other_cuda = allow_other_cuda
        self.lock = None
        self.child = None
        self.status = {'status': 'starting', 'pid': os.getpid(), 'updated_at': now(), 'current': None}
        self.stages = {}

    def acquire(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / 'queue.lock').open('a+b')
        self.lock.seek(0)
        if self.lock.read(1) == b'':
            self.lock.write(b'0')
            self.lock.flush()
        self.lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Also protect against a child left alive after an earlier queue crashed.
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            if process.info['pid'] == os.getpid() or 'python' not in (process.info['name'] or '').lower():
                continue
            command = ' '.join(process.info['cmdline'] or [])
            if str(self.out).lower() in command.lower() and any(
                token in command for token in ['train.py', 'pipeline.py', 'train_rf.py', 'infer.py', 'build_samples.py']
            ):
                raise RuntimeError(f'Existing experiment process {process.pid}; refusing overlap')
        stage_file = self.out / 'stages.json'
        if stage_file.exists():
            self.stages = json.loads(stage_file.read_text(encoding='utf-8'))

    def update(self, **values):
        self.status.update(values)
        self.status['updated_at'] = now()
        for method in DEEP:
            export_epochs(self.out / method)
        self.status['epochs'] = {
            method: max((r['epoch'] for r in records(self.out / method / 'metrics.jsonl')), default=0)
            for method in DEEP
        }
        atomic_json(self.out / 'status.json', self.status)

    def execute(self, key, arguments, cwd=HERE, accelerated=False):
        if self.stages.get(key, {}).get('status') == 'complete':
            print(f'{now()} skip completed stage {key}', flush=True)
            return
        log_folder = self.out / 'logs'
        log_folder.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(COCO_BACKEND='faster' if accelerated else 'pycocotools',
                   PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command = [sys.executable, '-u', *map(str, arguments)]
        self.stages[key] = dict(status='running', started_at=now(), command=command, cwd=str(cwd))
        atomic_json(self.out / 'stages.json', self.stages)
        self.update(status='running', current=key)
        print(f'{now()} start {key}', flush=True)
        started = time.perf_counter()
        with (log_folder / f'{key}.log').open('a', encoding='utf-8') as log:
            self.child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.update(child_pid=self.child.pid)
            while self.child.poll() is None:
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.update()
            code = self.child.returncode
        self.status.pop('child_pid', None)
        self.child = None
        self.stages[key].update(status='complete' if code == 0 else 'failed',
                                finished_at=now(), wall_s=time.perf_counter()-started, returncode=code)
        atomic_json(self.out / 'stages.json', self.stages)
        self.update()
        if code:
            raise RuntimeError(f'{key} exited with code {code}; queue stopped. See logs/{key}.log')
        print(f'{now()} complete {key}', flush=True)

    def validate_training(self, method):
        folder = self.out / method
        history = records(folder / 'metrics.jsonl')
        config = json.loads((folder / 'config.json').read_text())
        checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert [r['epoch'] for r in history] == list(range(1, 101)), method
        assert config['train_batch'] == 8 and config['seed'] == 42 and config['epochs'] == 100
        assert config['accumulation'] == 2 and config['validation_batch'] == 8
        assert checkpoint['epoch'] == 99 and checkpoint['training_complete']
        assert history[-1]['updates'] == 8600
        assert all(math.isfinite(v) for row in history for v in (
            row['losses'].values() if isinstance(row['losses'], dict) else row['losses']))
        assert [r['epoch'] for r in records(folder/'epoch_timing.jsonl')] == list(range(1, 101))
        best = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        ranked = max(history, key=lambda row: (row['AP50'], row['AP50:95'], -row['epoch']))
        assert best['epoch'] + 1 == ranked['epoch'] and best['training_complete']
        export_epochs(folder)

    def train_deep(self, method):
        folder = self.out / method
        last = folder / 'last.pt'
        key = f'{method}_train'
        if last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved['epoch'] == 99 and saved['training_complete']:
                self.validate_training(method)
                return
            del saved
        command = [HERE/'Resnet/pipeline.py', 'train', '--batch', '8'] if method == 'm3' else [SHARED/'train.py']
        command += ['--root', self.root, '--out', folder, '--device', 'cuda:0']
        if method == 'm2':
            command += ['--baseline']
        if method == 'p2-only':
            command += ['--p2-only']
        if last.exists():
            command += ['--resume', last]
        self.execute(key, command, accelerated=True)
        self.validate_training(method)

    def train_rf(self):
        folder = self.out/'m1'
        folder.mkdir(exist_ok=True)
        self.execute('m1_samples', [RF/'build_samples.py', '--root', self.root,
                     '--workers', '2', '--out', folder/'samples_round1.npz'], cwd=folder)
        self.execute('m1_train', [RF/'train_rf.py', '--root', self.root,
                     '--workers', '2', '--samples', folder/'samples_round1.npz',
                     '--out_dir', folder/'models'], cwd=folder)

    def evaluate_method(self, method):
        folder = self.out/method
        for split in ['val', 'test']:
            csv_path = folder/f'preds_{split}.csv'
            if method == 'm1':
                command = [RF/'infer.py', '--model', folder/'models/rf_final.joblib', '--workers', '1',
                           '--repeats', '3' if split == 'test' else '1', '--warmup', '20' if split == 'test' else '0']
            else:
                command = [HERE/'Resnet/pipeline.py', 'infer'] if method == 'm3' else [SHARED/'infer.py']
                command += ['--weights', folder/'best.pt', '--device', 'cuda:0']
                if split == 'test':
                    command += ['--benchmark']
            command += ['--root', self.root, '--split', split, '--out', csv_path]
            self.execute(f'{method}_{split}_infer', command)
            self.execute(f'{method}_{split}_evaluate', [SHARED/'evaluate.py', '--root', self.root,
                         '--split', split, '--preds', csv_path, '--out', folder/f'{split}_metrics.json'])

    def summary(self):
        rows = []
        for method in ['m1', *DEEP]:
            folder = self.out/method
            metric = json.loads((folder/'test_metrics.json').read_text())
            timing = json.loads((folder/('preds_test_timing.json' if method=='m1' else 'preds_test.timing.json')).read_text())
            if method=='m1':
                history = []
                train_s = json.loads((folder/'models/train_log.json').read_text())['total_train_s']
                throughput = 3*timing['n_images']/sum(r['total_s'] for r in timing['repeats'])
                best_epoch = ''
            else:
                history = records(folder/'metrics.jsonl')
                train_s = sum(r['training_s'] for r in history)
                throughput = timing['images_per_second']
                ckpt = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
                best_epoch = ckpt['epoch']+1
            rows.append(dict(method=method, epochs=len(history) or '', train_batch=8 if history else '',
                        seed=42, best_epoch=best_epoch, AP50=metric['AP50'], AP50_95=metric['AP50:95'],
                        precision=metric['precision'], recall=metric['recall'], F1=metric['F1'],
                        training_s=train_s, validation_s=sum(r['validation_s'] for r in history) if history else '',
                        images_per_s=throughput, hardware='CPU' if method=='m1' else torch.cuda.get_device_name(0)))
        with (self.out/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as file:
            writer=csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        atomic_json(self.out/'comparison.json', rows)

    def run(self):
        self.acquire()
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('The full deep benchmark requires a CUDA GPU')
            if not self.allow_other_cuda and '5090' not in torch.cuda.get_device_name(0):
                raise RuntimeError('This run is explicitly for the RTX 5090 D')
            if shutil.disk_usage(self.out).free < 16*1024**3:
                raise RuntimeError('Need at least 16 GiB free for checkpoint retention')
            dataset = dict(audit=audit(self.root), fingerprint=fingerprint(self.root))
            plan = dict(dataset=str(self.root), **dataset, seed=42, epochs=100, train_batch=8,
                        accumulation=2, effective_batch=16, validation_batch=8, inference_batch=8,
                        training_order=[*DEEP, 'm1'], test_policy='After every training configuration is complete',
                        hardware=torch.cuda.get_device_name(0), torch=torch.__version__,
                        cpu_thread_limits=2, python=sys.version,
                        rf_policy='CPU, 2 forest fits with hard-negative mining, n_jobs=2; no epoch/batch',
                        loss_policy='Per-component training losses and their sum; loss definitions differ across architectures')
            manifest=self.out/'plan.json'
            if manifest.exists():
                previous=json.loads(manifest.read_text(encoding='utf-8'))
                for key in ['fingerprint','seed','epochs','train_batch','hardware','torch']:
                    if previous[key] != plan[key]:
                        raise ValueError(f'Existing queue differs in {key}')
            else:
                atomic_json(manifest, plan)
                snapshots=self.out/'source_snapshot'; snapshots.mkdir(exist_ok=True)
                sources = [Path(__file__), SHARED/'train.py', SHARED/'model.py', SHARED/'data.py',
                           SHARED/'infer.py', SHARED/'evaluate.py', HERE/'Resnet/pipeline.py',
                           *sorted((SHARED/'configs').glob('*.yaml')), *sorted(RF.glob('*.py'))]
                for source in sources:
                    target=snapshots/source.relative_to(HERE); target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
            self.update(status='running')
            for method in DEEP:
                self.train_deep(method)
            self.train_rf()
            for method in [*DEEP, 'm1']:
                self.evaluate_method(method)
            self.summary()
            self.update(status='complete', current=None, finished_at=now())
            print('All serial training, validation, test evaluations and benchmarks complete.', flush=True)
        except BaseException as error:
            if self.child is not None and self.child.poll() is None:
                # Wait for termination before releasing the single-owner lock.
                process=psutil.Process(self.child.pid)
                descendants=process.children(recursive=True)
                for child in descendants:
                    child.terminate()
                process.terminate()
                psutil.wait_procs([process,*descendants], timeout=10)
            self.update(status='failed', error=f'{type(error).__name__}: {error}', failed_at=now())
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--out',default=str(HERE/'runs/5090d_batch8_seed42_20261117'))
    parser.add_argument('--allow-other-cuda', action='store_true',
                        help='Run the same protocol on another CUDA GPU; record its actual name')
    args=parser.parse_args()
    SerialQueue(Path(args.root).resolve(), Path(args.out).resolve(), args.allow_other_cuda).run()
