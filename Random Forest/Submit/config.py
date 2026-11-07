"""Shared constants for M1 (multi-scale sliding window + HOG/colour + Random Forest).
All values follow the fixed group protocol; do not tune them."""
from pathlib import Path

SEED = 42
CANVAS = 640                                  # images are scaled to a 640x640 canvas (pad value 114)
PAD_VALUE = 114
PATCH = 64                                    # every window is resized to 64x64 before feature extraction
CONTEXT = 0.10                                # positives: box expanded by 10% on each side
PYRAMID_SCALES = [1.0, 0.8, 0.64, 0.512]      # image pyramid ratios
WINDOW_SHAPES = [(32, 32), (48, 32), (32, 48), (64, 64)]   # (w, h) in pyramid-level pixels
STRIDE = 8
N_NEGATIVES = 20_000                          # random background windows (IoU < 0.1 with every GT box)
NEG_IOU_MAX = 0.1
SPLITS = ("train", "val", "test")

# HOG: grayscale, 9 orientations, 8x8 px cells, 2x2 cells/block, L2-Hys
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)
HSV_BINS = 8                                  # per-channel histogram bins, L1-normalised


def split_dirs(root, split):
    """Return (image_dir, label_dir) of a split. `root` is the dataset folder
    that contains images/ and labels/."""
    root = Path(root)
    return root / "images" / split, root / "labels" / split

# ---- Random Forest (protocol; everything else is the scikit-learn default) ----
RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_leaf=2, max_features="sqrt",
                 class_weight="balanced", bootstrap=True, random_state=SEED, n_jobs=2)
N_HARD_NEG = 5_000                            # max hard negatives mined after round 1

# ---- Inference / post-processing ----
SCORE_THR = 0.001                             # fixed pre-NMS score threshold
NMS_IOU = 0.6                                 # greedy hard NMS
MAX_DETS = 300                                # max boxes kept per image
BATCH_WINDOWS = 512                           # windows per feature batch (memory only, no effect on results)
