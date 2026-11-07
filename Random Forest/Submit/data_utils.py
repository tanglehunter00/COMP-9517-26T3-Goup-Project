"""Image / label loading helpers shared by all M1 scripts."""
from pathlib import Path
import cv2
import numpy as np
from config import CANVAS, PAD_VALUE, split_dirs


def imread(path):
    """cv2.imread replacement that also works for non-ASCII Windows paths (e.g. Chinese folder names)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def list_images(root, split):
    img_dir, _ = split_dirs(root, split)
    paths = sorted(img_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"No .jpg images found in '{img_dir}'. --root must be the folder that directly "
            f"contains 'images' and 'labels' (check the path and for an extra nested folder).")
    return paths


def read_yolo_labels(label_path):
    """Read a YOLO txt file -> float array (n, 5): class, xc, yc, w, h (normalised).
    A missing or empty file gives an empty (0, 5) array (negative image)."""
    p = Path(label_path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    arr = np.loadtxt(p, dtype=np.float64, ndmin=2)
    return arr.reshape(-1, 5)


def letterbox(img, canvas=CANVAS, pad_value=PAD_VALUE):
    """Resize keeping aspect ratio so the image fits canvas x canvas, then pad
    (centred) with pad_value. Returns canvas image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    s = min(canvas / w, canvas / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((canvas, canvas, 3), pad_value, dtype=np.uint8)
    px, py = (canvas - nw) // 2, (canvas - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, s, (px, py)


def yolo_to_canvas_xyxy(labels, orig_w, orig_h, scale, pad):
    """Normalised YOLO boxes -> pixel xyxy boxes in the letterboxed canvas."""
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    xc, yc, w, h = (labels[:, i] for i in range(1, 5))
    x1 = (xc - w / 2) * orig_w * scale + pad[0]
    y1 = (yc - h / 2) * orig_h * scale + pad[1]
    x2 = (xc + w / 2) * orig_w * scale + pad[0]
    y2 = (yc + h / 2) * orig_h * scale + pad[1]
    return np.stack([x1, y1, x2, y2], axis=1)


def load_canvas_and_boxes(img_path, root, split):
    """Load one image as a BGR canvas plus its GT boxes (xyxy, canvas pixels)."""
    img = imread(img_path)
    if img is None:
        raise IOError(f"cannot read {img_path}")
    _, lab_dir = split_dirs(root, split)
    labels = read_yolo_labels(lab_dir / (Path(img_path).stem + ".txt"))
    h, w = img.shape[:2]
    canvas, s, pad = letterbox(img)
    return canvas, yolo_to_canvas_xyxy(labels, w, h, s, pad)


def iou_one_to_many(box, boxes):
    """IoU of one xyxy box against an (n, 4) array."""
    if len(boxes) == 0:
        return np.zeros(0)
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)
