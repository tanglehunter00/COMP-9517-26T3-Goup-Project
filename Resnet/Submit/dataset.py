import io
import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id):
    """Deterministic per-worker seeds derived from the loader generator, so they change every epoch
    (a fixed 42 + worker_id would replay the same flip sequence each epoch)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BeachBirdDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        transform=None,
        target_size=640,
        pad_color=(114, 114, 114),
        corrupt=False,
    ):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            label_dir (str): Path to the directory containing YOLO .txt annotation files.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (int): Target canvas dimension (640x640).
            pad_color (tuple): RGB padding color, default (114, 114, 114).
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        self.pad_color = pad_color
        self.corrupt = corrupt  # evaluation only: fixed, seeded distortion per image

        # Get all valid image filenames (sorted for determinism)
        valid_extensions = (".jpg", ".jpeg", ".png")
        self.image_files = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
        )
        print(f"Loaded {len(self.image_files)} images from {img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, file_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        if self.corrupt:
            image = corrupt_image(image, idx)

        # Aspect-preserving resize and padding to target_size x target_size
        image, scale, pad_w, pad_h = self._resize_and_pad(image)

        # Load corresponding YOLO label file (.txt)
        base_name, _ = os.path.splitext(file_name)
        label_path = os.path.join(self.label_dir, base_name + ".txt")

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: x_center, y_center, width, height (normalized 0-1)
                        x_center, y_center, w_norm, h_norm = map(float, parts[1:5])

                        # Convert normalized YOLO to absolute pixel coordinates
                        box_w = w_norm * orig_width
                        box_h = h_norm * orig_height
                        box_x_center = x_center * orig_width
                        box_y_center = y_center * orig_height

                        x_min = box_x_center - box_w / 2
                        y_min = box_y_center - box_h / 2
                        x_max = box_x_center + box_w / 2
                        y_max = box_y_center + box_h / 2

                        # Scale and pad coordinates to match 640x640 canvas
                        x_min_new = x_min * scale + pad_w
                        y_min_new = y_min * scale + pad_h
                        x_max_new = x_max * scale + pad_w
                        y_max_new = y_max * scale + pad_h

                        # Clip boxes to canvas boundaries
                        x_min_new = max(0.0, min(float(self.target_size), x_min_new))
                        y_min_new = max(0.0, min(float(self.target_size), y_min_new))
                        x_max_new = max(0.0, min(float(self.target_size), x_max_new))
                        y_max_new = max(0.0, min(float(self.target_size), y_max_new))

                        if x_max_new > x_min_new and y_max_new > y_min_new:
                            boxes.append([x_min_new, y_min_new, x_max_new, y_max_new])
                            # Single foreground class: 'bird' (mapped to class ID 1, background is 0)
                            labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Apply augmentations (e.g., horizontal flip with p=0.5)
        if self.transform is not None:
            image, boxes = self.transform(image, boxes)

        # Format target dictionary required by torchvision Faster R-CNN
        target = {
            "boxes": boxes
            if len(boxes) > 0
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": labels
            if len(labels) > 0
            else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            # letterbox params to map predictions back to original-image pixels
            "meta": torch.tensor([scale, pad_w, pad_h, orig_width, orig_height], dtype=torch.float64),
        }

        # uint8 [C, H, W]; converted to float [0, 1] on the GPU (4x less host-to-device traffic)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)

        return image_tensor, target

    def _resize_and_pad(self, image):
        orig_width, orig_height = image.size
        scale = min(self.target_size / orig_width, self.target_size / orig_height)
        new_width = int(round(orig_width * scale))
        new_height = int(round(orig_height * scale))

        # Resize image maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Create padded canvas
        new_image = Image.new(
            "RGB", (self.target_size, self.target_size), self.pad_color
        )
        pad_w = (self.target_size - new_width) // 2
        pad_h = (self.target_size - new_height) // 2
        new_image.paste(image, (pad_w, pad_h))

        return new_image, scale, pad_w, pad_h


class HorizontalFlipTransform:
    """Applies horizontal flip augmentation with a given probability (M3 spec: 0.5)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(boxes) > 0:
                # Flip x coordinates: x_min_new = width - x_max, x_max_new = width - x_min
                width = image.width
                old_x_min = boxes[:, 0].clone()
                old_x_max = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x_max
                boxes[:, 2] = width - old_x_min
        return image, boxes


def _blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))


def _noise(image, sigma, rng):
    a = np.asarray(image, dtype=np.float32)
    return Image.fromarray(np.clip(a + rng.normal(0, sigma, a.shape), 0, 255).astype(np.uint8))


def _jpeg(image, quality):
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class PhotometricDistort:
    """--robust run only (non-protocol): random colour jitter, blur, sensor noise and JPEG artefacts.
    Pixel-only, so boxes are unchanged. Uses the per-worker seeded `random`, so runs stay reproducible."""

    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, image, boxes):
        if random.random() < 0.8:
            for enhance in random.sample([ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color], 3):
                image = enhance(image).enhance(random.uniform(0.6, 1.4))
        if random.random() < self.p:
            image = _blur(image, random.uniform(0.5, 2.0))
        if random.random() < self.p:
            image = _noise(image, random.uniform(3, 15), np.random.default_rng(random.getrandbits(32)))
        if random.random() < self.p:
            image = _jpeg(image, random.randint(20, 70))
        return image, boxes


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


CORRUPTIONS = ("blur", "noise", "jpeg", "dark", "low_contrast")


def corrupt_image(image, idx):
    """Fixed distorted copy for robustness evaluation: image idx always gets the same corruption
    (cycling through CORRUPTIONS) at a fixed moderate severity, applied to the original image."""
    rng = np.random.default_rng(10_000 + idx)
    kind = CORRUPTIONS[idx % len(CORRUPTIONS)]
    if kind == "blur":
        return _blur(image, 1.5)
    if kind == "noise":
        return _noise(image, 12, rng)
    if kind == "jpeg":
        return _jpeg(image, 25)
    if kind == "dark":
        return ImageEnhance.Brightness(image).enhance(0.45)
    return ImageEnhance.Contrast(image).enhance(0.45)


def collate_fn(batch):
    """Custom collate function for object detection batching."""
    return tuple(zip(*batch))


def make_loader(root, split, batch_size, train=False, generator=None, robust=False, corrupt=False):
    """Loader for <root>/images/<split> + <root>/labels/<split>. Protocol: 2 workers, no dropped batches,
    train shuffled with a seeded generator and horizontal flip 0.5; val/test unaugmented.
    robust=True adds PhotometricDistort to training; corrupt=True gives the fixed distorted eval copy."""
    transform = None
    if train:
        flip = HorizontalFlipTransform(p=0.5)
        transform = Compose(flip, PhotometricDistort()) if robust else flip
    dataset = BeachBirdDataset(
        os.path.join(root, "images", split), os.path.join(root, "labels", split),
        transform=transform, corrupt=corrupt)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2,
                      worker_init_fn=seed_worker, generator=generator, collate_fn=collate_fn,
                      drop_last=False, pin_memory=torch.cuda.is_available())
