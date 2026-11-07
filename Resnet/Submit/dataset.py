import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id):
    """Ensures deterministic behavior across DataLoader workers."""
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class BeachBirdDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        transform=None,
        target_size=640,
        pad_color=(114, 114, 114),
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
        }

        # Convert PIL image to Tensor [C, H, W] normalized to [0, 1]
        image_tensor = torch.tensor(np.array(image, dtype=np.float32) / 255.0).permute(
            2, 0, 1
        )

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


def collate_fn(batch):
    """Custom collate function for object detection batching."""
    return tuple(zip(*batch))


def get_dataloaders(train_dir, train_anns, val_dir, val_anns, batch_size=2):
    """
    Constructs train and validation DataLoaders adhering to the fixed training protocol (Seed 42, 2 workers).
    """
    random.seed(42)

    train_transform = HorizontalFlipTransform(p=0.5)
    val_transform = None  # No augmentations for validation

    train_dataset = BeachBirdDataset(train_dir, train_anns, transform=train_transform)
    val_dataset = BeachBirdDataset(val_dir, val_anns, transform=val_transform)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader
