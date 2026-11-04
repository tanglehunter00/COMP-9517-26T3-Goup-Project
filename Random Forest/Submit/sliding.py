"""Sliding-window machinery: image pyramid, window generation, BATCHED feature extraction
(numerically equivalent to features.patch_features, but ~50x faster), IoU and greedy NMS.
Batching is a pure speed optimisation; it does not change the protocol."""
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from config import (CANVAS, PATCH, PYRAMID_SCALES, WINDOW_SHAPES, STRIDE, HSV_BINS,
                    HOG_PARAMS, BATCH_WINDOWS)
from data_utils import iou_one_to_many

_CELL = HOG_PARAMS["pixels_per_cell"][0]
_NC = PATCH // _CELL                       # cells per side (8)
_ORI = HOG_PARAMS["orientations"]


_H_LUT = (np.arange(256) * HSV_BINS // 180).clip(0, HSV_BINS - 1).astype(np.uint8)   # hue 0-179 -> 8 bins


def _batched_hog(gray):
    """gray: (N,64,64) float32. Same algorithm as skimage.feature.hog with the protocol
    parameters (zero-gradient border, hard orientation bins, no spatial interpolation, L2-Hys)."""
    N = gray.shape[0]
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    ang[gy == 0] = 0.0                                  # horizontal gradients -> bin 0 (exactly like 180 % 180)
    a = ang * np.float32(_ORI / np.pi)
    a = np.where(a < 0, a + _ORI, a)                    # unsigned orientation in [0, 9)
    b = np.minimum(a.astype(np.int32), _ORI - 1)
    cell = (np.arange(PATCH)[:, None] // _CELL * _NC + np.arange(PATCH)[None, :] // _CELL)[None]
    idx = (np.arange(N, dtype=np.int32)[:, None, None] * (_NC * _NC) + cell) * _ORI + b
    hist = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=N * _NC * _NC * _ORI)
    hist = (hist / (_CELL * _CELL)).reshape(N, _NC, _NC, _ORI)
    nb = _NC - 1
    blocks = np.stack([hist[:, r:r + nb, c:c + nb, :] for r in (0, 1) for c in (0, 1)], axis=3)
    blocks = blocks.reshape(N, nb, nb, 4 * _ORI)                   # (2,2,orient) row-major
    eps = 1e-5
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps ** 2)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps ** 2)
    return blocks.reshape(N, -1)


def batched_features(patches):
    """patches: (N,64,64,3) uint8 BGR -> (N,1788) float32 (HOG + 3x8 HSV histograms)."""
    N = patches.shape[0]
    flat = np.ascontiguousarray(patches).reshape(N * PATCH, PATCH, 3)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).reshape(N, PATCH, PATCH).astype(np.float32)
    f_hog = _batched_hog(gray)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV).reshape(N, PATCH * PATCH, 3)
    bins = np.stack([_H_LUT[hsv[..., 0]], hsv[..., 1] >> 5, hsv[..., 2] >> 5], axis=1)   # (N,3,4096) uint8
    off = (np.arange(N, dtype=np.int32)[:, None, None] * 3 + np.arange(3, dtype=np.int32)[None, :, None]) * HSV_BINS
    cnt = np.bincount((bins.astype(np.int32) + off).ravel(), minlength=N * 3 * HSV_BINS)
    f_col = cnt.reshape(N, 3 * HSV_BINS) / float(PATCH * PATCH)     # per-channel L1 normalisation
    return np.concatenate([f_hog, f_col], axis=1).astype(np.float32)


def _level_image(canvas, li):
    L = int(round(CANVAS * PYRAMID_SCALES[li]))
    return canvas if L == CANVAS else cv2.resize(canvas, (L, L), interpolation=cv2.INTER_AREA)


def patches_for_ids(canvas, ids):
    """Regenerate the 64x64 patches of windows given by ids rows (level, shape, row, col)."""
    out, cache = [], {}
    for li, si, r, c in ids:
        if li not in cache:
            cache[li] = _level_image(canvas, int(li))
        w, h = WINDOW_SHAPES[int(si)]
        p = cache[li][r * STRIDE:r * STRIDE + h, c * STRIDE:c * STRIDE + w]
        if (h, w) != (PATCH, PATCH):
            p = cv2.resize(p, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
        out.append(p)
    return np.stack(out)


def iter_window_batches(canvas, batch=BATCH_WINDOWS):
    """Yield (patches64 (N,64,64,3) uint8, boxes (N,4) float32 xyxy in canvas pixels,
    ids (N,4) int32 = level, shape, row, col) over every pyramid level x window shape x stride-8 position."""
    for li, s in enumerate(PYRAMID_SCALES):
        L = int(round(CANVAS * s))
        level = _level_image(canvas, li)
        ratio = CANVAS / L
        for si, (w, h) in enumerate(WINDOW_SHAPES):
            if L < w or L < h:
                continue
            view = sliding_window_view(level, (h, w), axis=(0, 1))[::STRIDE, ::STRIDE]  # (nh,nw,3,h,w)
            nh, nw = view.shape[:2]
            rows = max(1, batch // nw)
            for r0 in range(0, nh, rows):
                r1 = min(nh, r0 + rows)
                chunk = np.ascontiguousarray(view[r0:r1].transpose(0, 1, 3, 4, 2)).reshape(-1, h, w, 3)
                if (h, w) != (PATCH, PATCH):
                    chunk = np.stack([cv2.resize(p, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
                                      for p in chunk])
                ys, xs = np.meshgrid(np.arange(r0, r1) * STRIDE, np.arange(nw) * STRIDE, indexing="ij")
                x1 = xs.ravel() * ratio; y1 = ys.ravel() * ratio
                boxes = np.stack([x1, y1, x1 + w * ratio, y1 + h * ratio], axis=1).astype(np.float32)
                ids = np.stack([np.full(len(boxes), li), np.full(len(boxes), si),
                                ys.ravel() // STRIDE, xs.ravel() // STRIDE], axis=1).astype(np.int32)
                yield chunk, boxes, ids


def iter_features(canvas, batch=BATCH_WINDOWS):
    for patches, boxes, ids in iter_window_batches(canvas, batch):
        yield batched_features(patches), boxes, ids


def iou_matrix(boxes, gts):
    """(N,4) x (G,4) -> (N,G) IoU."""
    if len(gts) == 0:
        return np.zeros((len(boxes), 0))
    b = boxes[:, None, :].astype(np.float64); g = gts[None, :, :]
    iw = np.clip(np.minimum(b[..., 2], g[..., 2]) - np.maximum(b[..., 0], g[..., 0]), 0, None)
    ih = np.clip(np.minimum(b[..., 3], g[..., 3]) - np.maximum(b[..., 1], g[..., 1]), 0, None)
    inter = iw * ih
    ab = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]); ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / np.maximum(ab + ag - inter, 1e-9)


def hard_nms(boxes, scores, iou_thr, max_keep):
    """Greedy hard NMS; returns indices of kept boxes (highest score first)."""
    order = np.argsort(-scores, kind="stable")
    boxes = boxes[order]
    idx, keep = np.arange(len(order)), []
    while idx.size and len(keep) < max_keep:
        i = idx[0]; keep.append(order[i])
        if idx.size == 1:
            break
        ious = iou_one_to_many(boxes[i], boxes[idx[1:]])
        idx = idx[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)
