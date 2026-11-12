"""M4 only. Register DySample in this process, without editing Ultralytics or M2."""
from pathlib import Path
import re
import torch
from ultralytics import YOLO, __version__
from ultralytics.cfg import get_cfg
from ultralytics.nn import tasks
from vendor.dysample import DySample

HERE = Path(__file__).resolve().parent


def build_model(ablation=False, weights=None, baseline=False):
    if __version__ != '8.3.221':
        raise RuntimeError('This implementation requires ultralytics==8.3.221')
    tasks.DySample = DySample
    cfg = 'yolo11s.yaml' if baseline else HERE / 'configs' / ('yolo11s-p2.yaml' if ablation else 'yolo11s-p2-dysample.yaml')
    model = tasks.DetectionModel(str(cfg), ch=3, nc=1, verbose=False)
    model.args = get_cfg()
    model.names = {0: 'bird'}
    report = {'source': str(weights), 'loaded': [], 'skipped': []}
    if weights:
        source = YOLO(str(weights)).model.state_dict()
        target = model.state_dict()
        for key, value in source.items():
            mapped = key
            # Existing backbone/neck 0..22 retain their original indices.
            # Old P3/P4/P5 detector branches become branches 1/2/3 in layer 26.
            if key.startswith('model.23.') and not baseline:
                mapped = key.replace('model.23.', 'model.26.', 1)
                mapped = re.sub(r'(cv[23]\.)([012])\.',
                                lambda m: m[1] + str(int(m[2]) + 1) + '.', mapped)
            if mapped in target and target[mapped].shape == value.shape:
                target[mapped] = value
                report['loaded'].append([key, mapped])
            else:
                report['skipped'].append(key)
        model.load_state_dict(target)
    if model.stride.tolist() != ([8., 16., 32.] if baseline else [4., 8., 16., 32.]):
        raise RuntimeError(f'Unexpected detection strides: {model.stride}')
    return model, report


def load_model(path, device):
    # Only load trusted project checkpoints (optimizer/RNG state needs full pickle).
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model, _ = build_model(ckpt['ablation'], baseline=ckpt.get('baseline', False))
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), ckpt
