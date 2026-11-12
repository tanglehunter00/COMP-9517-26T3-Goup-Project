"""Fixed M4/P2-only training loop. No EMA, early stopping, or automatic scaling."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from data import TrainDataset, audit, seed_worker
from model import HERE, build_model
from infer import predict_split, sync
from evaluate import ground_truth, evaluate, BACKEND


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(update, total, warmup):
    if update < warmup:
        return .001 * (.1 + .9*update/max(warmup-1, 1))
    phase = (update-warmup)/max(total-warmup-1, 1)
    return .00001 + (.001-.00001)*(1+math.cos(math.pi*phase))/2


def optimizer_for(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (no_decay if param.ndim == 1 or name.endswith('.bias') else decay).append(param)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': .0005},
                              {'params': no_decay, 'weight_decay': 0.}], lr=.001, betas=(.9, .999))


def fingerprint(root):
    digest = hashlib.sha256()
    for folder in ['images', 'labels']:
        for path in sorted((Path(root)/folder).rglob('*')):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def save_atomic(path, checkpoint):
    tmp = path.with_suffix('.tmp')
    torch.save(checkpoint, tmp); os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--weights', default=str(HERE/'checkpoints/yolo11s.pt'))
    p.add_argument('--out', default=str(HERE/'runs/m4')); p.add_argument('--resume')
    p.add_argument('--p2-only', action='store_true'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--baseline', action='store_true', help='M2: standard YOLO11s with the identical protocol')
    p.add_argument('--smoke', action='store_true', help='2 epochs, non-official image counts; outputs explicitly marked')
    p.add_argument('--stop-after-epoch', type=int, help='Save and stop at this epoch; resume keeps the original schedule')
    a = p.parse_args()
    if a.baseline and a.p2_only:
        p.error('--baseline and --p2-only are mutually exclusive')
    root = Path(a.root).resolve(); out = Path(a.out).resolve()
    if not out.is_relative_to(HERE):
        raise ValueError('Outputs must remain inside Yolo Improeved')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() and not a.resume:
        raise ValueError('Run already exists; use --resume or a new output folder')
    device = torch.device(a.device)
    if device.type != 'cuda' and not a.smoke:
        raise ValueError('Protocol requires CUDA FP16; use --smoke for CPU diagnostics')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_all(42); torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    report = audit(root, official=not a.smoke)
    data_hash = fingerprint(root)
    epochs = 2 if a.smoke else 100
    resume = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else None
    if a.resume and Path(a.resume).resolve() != out/'last.pt':
        raise ValueError('Resume must use last.pt in the same output folder (preserves selected best)')
    if resume and (resume['data_hash'] != data_hash or resume['ablation'] != a.p2_only or resume['epochs'] != epochs or resume.get('baseline', False) != a.baseline):
        raise ValueError('Resume data/model/schedule does not match')
    model, transfer = build_model(a.p2_only, None if resume else a.weights, baseline=a.baseline)
    hyp = get_cfg(overrides=dict(imgsz=640, batch=8, seed=42, workers=2,
                  hsv_h=.015, hsv_s=.4, hsv_v=.3, degrees=0., translate=.1, scale=.3,
                  shear=0., perspective=0., flipud=0., fliplr=.5, mosaic=.5,
                  mixup=0., copy_paste=0., cutmix=0., bgr=0., deterministic=True))
    model.args = hyp; model.to(device)
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., enabled=device.type == 'cuda')
    steps_per_epoch = math.ceil(report['train']['images']/16)
    total, warmup = epochs*steps_per_epoch, min(3, epochs)*steps_per_epoch
    # Smoke still exercises both scheduler stages.
    if a.smoke:
        warmup = steps_per_epoch
    start, updates, best = 0, 0, (-1., -1.)
    if resume:
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        start, updates, best = resume['epoch']+1, resume['updates'], tuple(resume['best'])
        random.setstate(resume['rng_python']); np.random.set_state(resume['rng_numpy'])
        torch.set_rng_state(resume['rng_torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(resume['rng_cuda'])
    config = dict(args=vars(a), data=report, data_hash=data_hash, epochs=epochs, seed=42,
                  train_batch=8, accumulation=2, validation_batch=8, precision='FP16' if device.type=='cuda' else 'FP32',
                  ema=False, hyp=vars(hyp), architecture=model.yaml, torch=torch.__version__,
                  optimizer='AdamW', weight_decay={'weights': .0005, 'bias_and_norm': 0.},
                  amp_initial_scale=128.,
                  scheduler={'total_updates': total, 'warmup_updates': warmup},
                  device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                  nondeterminism='warn_only=True; CUDA grid_sample backward may be nondeterministic',
                  smoke=a.smoke, evaluator=BACKEND, official_hardware=device.type=='cuda' and 'T4' in torch.cuda.get_device_name(device))
    (out/'config.json').write_text(json.dumps(config, indent=2, default=str))
    if not resume:
        (out/'transfer.json').write_text(json.dumps(transfer, indent=2))
    gt = ground_truth(root, 'val')
    for epoch in range(start, epochs):
        # Epoch-specific seeds and a fresh dataset also make interrupted epoch replay reproducible.
        seed_all(42+epoch)
        epoch_hyp = get_cfg(overrides=vars(hyp))
        if epoch >= 90:
            epoch_hyp.mosaic = 0.
        dataset = TrainDataset(root, epoch_hyp)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2,
                            collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(42+epoch), drop_last=False,
                            pin_memory=device.type=='cuda')
        model.train(); optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sync(device); t = time.perf_counter(); loss_sum = np.zeros(3); count = 0
        for i, batch in enumerate(loader):
            n = len(batch['img'])
            for key in ['img', 'cls', 'bboxes', 'batch_idx']:
                batch[key] = batch[key].to(device, non_blocking=True)
            batch['img'] = batch['img'].float()/255
            # Ultralytics returns a loss multiplied by batch size. Divide the entire
            # accumulation group by its actual image count, including the tail.
            group_images = min(16, len(dataset)-(i//2)*16)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                loss, parts = model(batch)
                scalar = loss.sum()/group_images
            if not torch.isfinite(scalar):
                raise RuntimeError('Non-finite loss; fixed configuration failed')
            scaler.scale(scalar).backward()
            loss_sum += parts.detach().float().cpu().numpy()*n; count += n
            if i%2 == 1 or i == len(loader)-1:
                lr = lr_at(updates, total, warmup)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite gradient; fixed configuration failed')
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); updates += 1
        sync(device); train_s = time.perf_counter()-t
        peak = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
        t = time.perf_counter(); model.eval()
        raw = predict_split(model, root, 'val', device)
        mapping = {im['stem']: im['id'] for im in gt['images']}
        rows = [dict(image_id=mapping[r['image_id']], category_id=0,
                     bbox=[r['x'],r['y'],r['width'],r['height']], score=r['score']) for r in raw]
        metrics = evaluate(gt, rows); sync(device); val_s = time.perf_counter()-t
        key = (metrics['AP50'] if metrics['AP50'] is not None else -1.,
               metrics['AP50:95'] if metrics['AP50:95'] is not None else -1.)
        improved = key > best
        if improved:
            best = key
        log = dict(epoch=epoch+1, losses=(loss_sum/count).tolist(), lr=lr, updates=updates,
                   training_s=train_s, validation_s=val_s, peak_training_allocated_bytes=peak, **metrics)
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(log, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in log.items() if k!='density'}), flush=True)
        ckpt = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    scheduler={'updates': updates, 'total': total, 'warmup': warmup},
                    epoch=epoch, epochs=epochs, updates=updates, best=best, ablation=a.p2_only,
                    data_hash=data_hash, smoke=a.smoke, baseline=a.baseline, training_complete=epoch==epochs-1,
                    rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        save_atomic(out/'last.pt', ckpt)
        if improved:
            save_atomic(out/'best.pt', ckpt)
        if (epoch+1)%10 == 0:
            save_atomic(out/f'epoch{epoch+1:03}.pt', ckpt)
        if a.stop_after_epoch and epoch+1 >= a.stop_after_epoch and epoch+1 < epochs:
            print('Saved interruption checkpoint; resume from last.pt to finish the unchanged schedule')
            return
    if (out/'best.pt').exists():
        selected = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        selected['training_complete'] = True
        save_atomic(out/'best.pt', selected)


if __name__ == '__main__':
    main()
