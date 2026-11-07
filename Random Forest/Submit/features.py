"""Feature extraction: HOG (grayscale) + HSV colour histogram for a 64x64 patch."""
import cv2
import numpy as np
from skimage.feature import hog
from config import HOG_PARAMS, HSV_BINS, PATCH


def patch_features(patch_bgr):
    """patch_bgr: uint8 BGR image, any size. Returns a float32 vector (1788-d)."""
    if patch_bgr.shape[:2] != (PATCH, PATCH):
        patch_bgr = cv2.resize(patch_bgr, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    f_hog = hog(gray, **HOG_PARAMS)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    ranges = [(0, 180), (0, 256), (0, 256)]      # OpenCV hue range is 0-179
    hists = []
    for c, r in enumerate(ranges):
        h, _ = np.histogram(hsv[:, :, c], bins=HSV_BINS, range=r)
        h = h.astype(np.float64)
        hists.append(h / max(h.sum(), 1e-9))     # L1 normalisation
    return np.concatenate([f_hog, *hists]).astype(np.float32)


def crop_window(canvas, x1, y1, x2, y2):
    """Crop an (integer-rounded, border-clipped) window from the canvas."""
    H, W = canvas.shape[:2]
    x1, y1 = max(int(round(x1)), 0), max(int(round(y1)), 0)
    x2, y2 = min(int(round(x2)), W), min(int(round(y2)), H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return canvas[y1:y2, x1:x2]
