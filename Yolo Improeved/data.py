"""Strict YOLO dataset audit; no label rewriting or external cache files."""
from pathlib import Path
import hashlib
import random
import cv2
import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset


def image_paths(root, split):
    paths = sorted(p for p in (Path(root) / 'images' / split).glob('*')
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if not paths:
        raise ValueError(f'No images: {root}/images/{split}')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Image stems must be unique within a split')
    return paths


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    return image


def labels(root, split, path):
    file = Path(root) / 'labels' / split / (Path(path).stem + '.txt')
    if not file.is_file():
        raise ValueError(f'Missing label file (not treated as a negative): {file}')
    if not file.read_text().strip():
        return np.empty((0, 5), dtype=np.float32)
    a = np.loadtxt(file, ndmin=2, dtype=np.float32)
    if a.shape[1] != 5 or not np.isfinite(a).all():
        raise ValueError(f'Invalid YOLO labels: {file}')
    if np.any(a[:, 0] != 0) or np.any(a[:, 3:] <= 0):
        raise ValueError(f'Expected class 0 and positive box sizes: {file}')
    lo, hi = a[:, 1:3] - a[:, 3:] / 2, a[:, 1:3] + a[:, 3:] / 2
    if np.any(lo < -1e-6) or np.any(hi > 1 + 1e-6):
        raise ValueError(f'Out-of-bounds box: {file}')
    return a


def audit(root, official=True):
    result, hashes = {}, {}
    for split, expected in [('train', 1369), ('val', 291), ('test', 240)]:
        paths = image_paths(root, split)
        n = 0
        for p in paths:
            read_image(p)
            n += len(labels(root, split, p))
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest][0] != split:
                raise ValueError(f'Cross-split duplicate: {hashes[digest]} and {split}/{p.name}')
            hashes[digest] = (split, p.name)
        result[split] = {'images': len(paths), 'boxes': n}
        if official and len(paths) != expected:
            raise ValueError(f'{split}: expected {expected} images, found {len(paths)}')
    return result


class TrainDataset(YOLODataset):
    def __init__(self, root, hyp):
        self.project_root = Path(root)
        super().__init__(img_path=str(self.project_root / 'images/train'), imgsz=640,
                         batch_size=8, augment=True, hyp=hyp, rect=False, cache=False,
                         stride=32, data={'names': {0: 'bird'}, 'nc': 1, 'channels': 3})

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        # Ultralytics conditionally adds blur/CLAHE when albumentations is installed.
        # Remove it so the resolved pipeline is independent of optional packages.
        transforms.transforms = [t for t in transforms.transforms if t.__class__.__name__ != 'Albumentations']
        return transforms

    def get_labels(self):
        out = []
        for file in self.im_files:
            image = read_image(file)
            a = labels(self.project_root, 'train', file)
            out.append(dict(im_file=file, shape=image.shape[:2], cls=a[:, :1],
                            bboxes=a[:, 1:], segments=[], keypoints=None,
                            normalized=True, bbox_format='xywh'))
        return out


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def letterbox(image):
    h, w = image.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (640 - nw) // 2, (640 - nh) // 2
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(image, (nw, nh))
    return canvas, (scale, px, py, w, h)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
