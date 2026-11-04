"""Detection and hard-negative mining for one 640x640 canvas image."""
import numpy as np
from config import SCORE_THR, NMS_IOU, MAX_DETS, NEG_IOU_MAX, N_HARD_NEG
from sliding import iter_features, iou_matrix, hard_nms


def detect(model, canvas):
    """All windows -> RF score -> keep score >= SCORE_THR -> greedy hard NMS (IoU 0.6) -> max 300.
    Returns boxes (n,4) xyxy in canvas pixels and scores (n,)."""
    bs, ss = [], []
    for feats, boxes, _ in iter_features(canvas):
        p = model.predict_proba(feats)[:, 1]
        m = p >= SCORE_THR
        if m.any():
            bs.append(boxes[m]); ss.append(p[m])
    if not bs:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes, scores = np.concatenate(bs), np.concatenate(ss)
    keep = hard_nms(boxes, scores, NMS_IOU, MAX_DETS)
    return boxes[keep], scores[keep]


def mine_image(model, canvas, gts, k=N_HARD_NEG):
    """Highest-scoring windows whose IoU with EVERY GT box is < 0.1 (top k of this image).
    Returns scores (m,) and ids (m,4)."""
    sc, idl = [], []
    for feats, boxes, ids in iter_features(canvas):
        neg = iou_matrix(boxes, gts).max(axis=1, initial=0.0) < NEG_IOU_MAX
        if not neg.any():
            continue
        sc.append(model.predict_proba(feats[neg])[:, 1]); idl.append(ids[neg])
    if not sc:
        return np.zeros(0, np.float32), np.zeros((0, 4), np.int32)
    sc, idl = np.concatenate(sc), np.concatenate(idl)
    if len(sc) > k:
        top = np.argpartition(-sc, k - 1)[:k]
        sc, idl = sc[top], idl[top]
    return sc.astype(np.float32), idl
