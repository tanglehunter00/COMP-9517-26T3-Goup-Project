"""M3 Faster R-CNN V2: fixed training, original-coordinate export and timing."""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / 'Yolo Improeved'
sys.path.insert(0, str(SHARED))
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data import audit, image_paths, labels, letterbox, read_image, seed_worker
from evaluate import evaluate, ground_truth, BACKEND
from infer import benchmark, sync
from train import fingerprint, save_atomic, seed_all, reconcile_log


def freeze_bn(model):
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def build_model(pretrained=True):
    if torchvision.__version__.split('+')[0] != '0.21.0':
        raise RuntimeError('M3 requires torchvision==0.21.0 (paired with torch==2.6.0)')
    model = fasterrcnn_resnet50_fpn_v2(
        weights='COCO_V1' if pretrained else None, weights_backbone=None,
        trainable_backbone_layers=3 if pretrained else 5,
        min_size=640, max_size=640, rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000, rpn_nms_thresh=.7,
        box_batch_size_per_image=512, box_positive_fraction=.25,
        box_score_thresh=.001, box_nms_thresh=.6, box_detections_per_img=300)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2)
    # Explicit freezing also applies to checkpoint reload without COCO weights.
    for name, parameter in model.backbone.body.named_parameters():
        if not name.startswith(('layer2.', 'layer3.', 'layer4.')):
            parameter.requires_grad_(False)
    freeze_bn(model)
    return model


def prepare(image):
    canvas, geometry = letterbox(image)
    tensor = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))).float()/255
    return tensor, geometry


class Birds(Dataset):
    def __init__(self, root):
        self.root = root
        self.paths = image_paths(root, 'train')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor, (scale, px, py, w, h) = prepare(read_image(path))
        lab = labels(self.root, 'train', path)
        boxes = np.empty((len(lab), 4), dtype=np.float32)
        boxes[:, 0] = (lab[:, 1]-lab[:, 3]/2)*w*scale+px
        boxes[:, 2] = (lab[:, 1]+lab[:, 3]/2)*w*scale+px
        boxes[:, 1] = (lab[:, 2]-lab[:, 4]/2)*h*scale+py
        boxes[:, 3] = (lab[:, 2]+lab[:, 4]/2)*h*scale+py
        if random.random() < .5:
            tensor = tensor.flip(-1)
            boxes[:, [0, 2]] = 640-boxes[:, [2, 0]]
        return tensor, dict(boxes=torch.from_numpy(boxes), labels=torch.ones(len(boxes), dtype=torch.int64))


def collate(batch):
    return tuple(zip(*batch))


def lr_at(update, total, warmup):
    if update < warmup:
        return .0025*(.1+.9*update/max(warmup-1, 1))
    phase = (update-warmup)/max(total-warmup-1, 1)
    return .000025+(.0025-.000025)*(1+math.cos(math.pi*phase))/2


def update_group(model, batches, optimizer, scaler, device):
    """Retry AMP overflows without dropping an update or repeating BN/RNG effects."""
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type=='cuda' else []
    count = sum(len(images) for images, _ in batches)
    for attempt in range(16):
        if attempt:
            with torch.no_grad():
                for buffer, saved in buffers: buffer.copy_(saved)
            torch.set_rng_state(cpu_rng)
            if device.type=='cuda': torch.cuda.set_rng_state_all(cuda_rng)
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for images, targets in batches:
            n = len(images)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                parts = model(images, targets); loss = sum(parts.values())*n/count
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; fixed configuration failed')
            scaler.scale(loss).backward()
            for key, value in parts.items(): losses[key] = losses.get(key, 0.)+float(value.detach())*n
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=False)
        if torch.isfinite(norm):
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            return losses, count, attempt
        if not scaler.is_enabled() or all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Non-finite gradient norm unrelated to AMP overflow')
        previous_scale = scaler.get_scale()
        # unscale_ detected non-finite gradients, so step skips the optimizer.
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() >= previous_scale:
            raise RuntimeError('AMP failed to back off after gradient overflow')
    raise RuntimeError('AMP overflow persisted after 16 retries; fixed configuration failed')


@torch.inference_mode()
def predict_batch(model, images, device):
    processed = [prepare(image) for image in images]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
        outputs = model([tensor.to(device) for tensor, _ in processed])
    result = []
    for output, (_, (scale, px, py, w, h)) in zip(outputs, processed):
        boxes = output['boxes'].float().cpu().numpy()
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]]-px)/scale, 0, w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]]-py)/scale, 0, h)
        scores = output['scores'].float().cpu().numpy()
        valid = (boxes[:, 2]>boxes[:, 0]) & (boxes[:, 3]>boxes[:, 1])
        result.append(np.column_stack([boxes, scores, np.zeros(len(scores))])[valid])
    return result


def predictions(model, root, split, device):
    paths = image_paths(root, split)
    rows = []
    for i in range(0, len(paths), 8):
        batch = paths[i:i+8]
        for path, boxes in zip(batch, predict_batch(model, [read_image(p) for p in batch], device)):
            for x1, y1, x2, y2, score, _ in boxes:
                rows.append(dict(image_id=path.stem, category_id=0, x=float(x1), y=float(y1),
                                 width=float(x2-x1), height=float(y2-y1), score=float(score)))
    return rows


def metrics_for(gt, rows):
    mapping = {im['stem']: im['id'] for im in gt['images']}
    return evaluate(gt, [dict(image_id=mapping[r['image_id']], category_id=0,
        bbox=[r['x'], r['y'], r['width'], r['height']], score=r['score']) for r in rows])


def train(args):
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:
        raise ValueError('Run exists; resume last.pt or choose a new output directory')
    if args.resume and Path(args.resume).resolve() != out/'last.pt':
        raise ValueError('Resume requires last.pt in the same output directory')
    device = torch.device(args.device)
    if device.type != 'cuda' and not args.smoke:
        raise ValueError('Full training requires CUDA FP16')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_all(42); torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    report = audit(args.root, official=not args.smoke); data_hash = fingerprint(args.root)
    epochs = 2 if args.smoke else 100
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if ckpt and (ckpt['data_hash'] != data_hash or ckpt['epochs'] != epochs or ckpt['smoke'] != args.smoke):
        raise ValueError('Resume data or schedule mismatch')
    model = build_model(pretrained=not bool(ckpt)).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                               lr=.0025, momentum=.9, weight_decay=.0001, nesterov=False)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., enabled=device.type=='cuda')
    dataset = Birds(args.root); steps = math.ceil(len(dataset)/4)
    total, warmup = epochs*steps, steps
    start, updates, best = 0, 0, (-1., -1.)
    if ckpt:
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start, updates, best = ckpt['epoch']+1, ckpt['updates'], tuple(ckpt['best'])
        reconcile_log(out, start)
        random.setstate(ckpt['rng_python']); np.random.set_state(ckpt['rng_numpy'])
        torch.set_rng_state(ckpt['rng_torch'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
    config = dict(args=vars(args), data=report, data_hash=data_hash, epochs=epochs,
        torch=torch.__version__, torchvision=torchvision.__version__, seed=42, train_batch=2,
        accumulation=2, validation_batch=8, workers=2, precision='FP16' if device.type=='cuda' else 'FP32',
        optimizer='SGD', momentum=.9, nesterov=False, weight_decay=.0001,
        amp_overflow_policy='back off and replay same accumulation group; restore buffers and RNG; max 16 retries',
        scheduler=dict(total_updates=total, warmup_updates=warmup),
        device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        official_hardware=device.type=='cuda' and 'T4' in torch.cuda.get_device_name(device),
        smoke=args.smoke, evaluator=BACKEND, nondeterminism='warn_only=True; CUDA roi_align backward may be nondeterministic', architecture=str(model))
    (out/'config.json').write_text(json.dumps(config, indent=2))
    gt = ground_truth(args.root, 'val')
    for epoch in range(start, epochs):
        seed_all(42+epoch)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42+epoch),
            pin_memory=device.type=='cuda', drop_last=False)
        model.train(); freeze_bn(model); optimizer.zero_grad(set_to_none=True)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        sync(device); began = time.perf_counter(); losses = {}; count = 0; batches = []; amp_retries = 0
        for i, (images, targets) in enumerate(loader):
            images = [im.to(device) for im in images]
            targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
            batches.append((images, targets))
            if i%2==1 or i==len(loader)-1:
                lr = lr_at(updates, total, warmup)
                for group in optimizer.param_groups: group['lr'] = lr
                group_losses, n, retries = update_group(model, batches, optimizer, scaler, device)
                for k,v in group_losses.items(): losses[k] = losses.get(k, 0.)+v
                count += n; amp_retries += retries; batches.clear(); updates += 1
        sync(device); training_s = time.perf_counter()-began
        peak = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
        model.eval(); began = time.perf_counter()
        metrics = metrics_for(gt, predictions(model, args.root, 'val', device)); sync(device)
        key = tuple(metrics[k] if metrics[k] is not None else -1. for k in ['AP50', 'AP50:95'])
        improved = key>best
        if improved: best=key
        log = dict(epoch=epoch+1, losses={k:v/count for k,v in losses.items()}, lr=lr, updates=updates,
            amp_retries=amp_retries, amp_scale=scaler.get_scale(),
            training_s=training_s, validation_s=time.perf_counter()-began, peak_training_allocated_bytes=peak, **metrics)
        with (out/'metrics.jsonl').open('a') as file: file.write(json.dumps(log, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in log.items() if k!='density'}), flush=True)
        state = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch, epochs=epochs,
            scheduler=dict(updates=updates, total=total, warmup=warmup), updates=updates, best=best,
            data_hash=data_hash, smoke=args.smoke, training_complete=epoch==epochs-1,
            rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        save_atomic(out/'last.pt', state)
        if improved: save_atomic(out/'best.pt', state)
        if (epoch+1)%10==0: save_atomic(out/f'epoch{epoch+1:03}.pt', state)
        if args.stop_after_epoch and epoch+1>=args.stop_after_epoch and epoch+1<epochs: return
    selected = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    selected['training_complete']=True; save_atomic(out/'best.pt', selected)


def infer(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    if ckpt['smoke'] and not args.allow_smoke: raise ValueError('Diagnostic checkpoint requires --allow-smoke')
    if args.split=='test' and not ckpt['training_complete']: raise ValueError('Complete training before testing')
    model = build_model(False).to(device); model.load_state_dict(ckpt['model']); model.eval()
    rows = predictions(model, args.root, args.split, device)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['image_id','category_id','x','y','width','height','score'])
        writer.writeheader(); writer.writerows(rows)
    out.with_suffix('.metrics.json').write_text(json.dumps(metrics_for(ground_truth(args.root, args.split), rows), indent=2))
    if args.benchmark:
        if args.split!='test': raise ValueError('Benchmark uses test split')
        timing = benchmark(model, [read_image(p) for p in image_paths(args.root, args.split)], device, predict_batch)
        out.with_suffix('.timing.json').write_text(json.dumps(timing, indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['train','infer'])
    parser.add_argument('--root', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--weights'); parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--benchmark', action='store_true'); parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    (train if args.mode=='train' else infer)(args)
