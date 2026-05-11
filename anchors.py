"""
anchors.py — SSD default (anchor) box generation + encoding / decoding utilities.

All boxes are represented in [cx, cy, w, h] format, normalized to [0, 1].
"""

import numpy as np
import tensorflow as tf
from config import (
    FEATURE_MAP_SIZES, ANCHOR_SCALES, ANCHOR_ASPECT_RATIOS,
    NUM_ANCHORS, IOU_MATCH_THRESH, ENCODE_VARIANCES
)


# ─────────────────────────────── Generation ──────────────────────────────────

def generate_anchors() -> np.ndarray:
    """
    Generate all 8732 SSD default boxes.

    For each feature map level k and each spatial cell (i, j):
      - One square anchor at scale s_k
      - One square anchor at the "additional" scale  s'_k = sqrt(s_k * s_{k+1})
      - For each extra aspect ratio ar in ANCHOR_ASPECT_RATIOS[k]:
            w = s_k * sqrt(ar),   h = s_k / sqrt(ar)
            w = s_k / sqrt(ar),   h = s_k * sqrt(ar)   (inverse)

    Returns:
        np.ndarray of shape [8732, 4], dtype float32, clipped to [0, 1].
    """
    anchors = []

    for k, (f_k, ars) in enumerate(zip(FEATURE_MAP_SIZES, ANCHOR_ASPECT_RATIOS)):
        s_k  = ANCHOR_SCALES[k]
        s_k1 = ANCHOR_SCALES[k + 1]

        for i in range(f_k):
            for j in range(f_k):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # Square anchor at scale s_k
                anchors.append([cx, cy, s_k, s_k])

                # Square anchor at the interpolated scale
                s_prime = float(np.sqrt(s_k * s_k1))
                anchors.append([cx, cy, s_prime, s_prime])

                # Non-square anchors for each aspect ratio
                for ar in ars:
                    w = s_k * np.sqrt(ar)
                    h = s_k / np.sqrt(ar)
                    anchors.append([cx, cy, w, h])       # ar
                    anchors.append([cx, cy, h, w])       # 1/ar

    anchors = np.array(anchors, dtype=np.float32)
    anchors = np.clip(anchors, 0.0, 1.0)

    assert len(anchors) == NUM_ANCHORS, (
        f"Expected {NUM_ANCHORS} anchors, generated {len(anchors)}. "
        "Check FEATURE_MAP_SIZES and ANCHOR_ASPECT_RATIOS in config.py."
    )
    return anchors   # [8732, 4]


# ──────────────────────────── IoU (numpy) ────────────────────────────────────

def compute_iou_np(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of [cx, cy, w, h] boxes.

    Args:
        boxes_a: [M, 4]
        boxes_b: [N, 4]

    Returns:
        iou: [M, N]
    """
    def to_xyxy(b):
        return np.stack([
            b[..., 0] - b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] + b[..., 3] / 2,
        ], axis=-1)

    a = to_xyxy(boxes_a)[:, np.newaxis, :]   # [M, 1, 4]
    b = to_xyxy(boxes_b)[np.newaxis, :, :]   # [1, N, 4]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_area = (np.maximum(0.0, inter_x2 - inter_x1) *
                  np.maximum(0.0, inter_y2 - inter_y1))

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])  # [M, 1]
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # [1, N]

    union = area_a + area_b - inter_area + 1e-10
    return inter_area / union   # [M, N]


# ───────────────────────────── Encoding ──────────────────────────────────────

def encode_offsets(gt_boxes: np.ndarray,
                   anchors: np.ndarray,
                   variances=ENCODE_VARIANCES) -> np.ndarray:
    """
    Encode GT boxes relative to matched anchors (SSD offset encoding).

    Both arrays are in [cx, cy, w, h] normalized format.
    t_cx = (gt_cx - a_cx) / (var0 * a_w)
    t_cy = (gt_cy - a_cy) / (var0 * a_h)
    t_w  = log(gt_w  / a_w) / var1
    t_h  = log(gt_h  / a_h) / var1
    """
    cx = (gt_boxes[:, 0] - anchors[:, 0]) / (variances[0] * anchors[:, 2])
    cy = (gt_boxes[:, 1] - anchors[:, 1]) / (variances[0] * anchors[:, 3])
    w  = np.log(np.maximum(gt_boxes[:, 2], 1e-10) / anchors[:, 2]) / variances[1]
    h  = np.log(np.maximum(gt_boxes[:, 3], 1e-10) / anchors[:, 3]) / variances[1]
    return np.stack([cx, cy, w, h], axis=-1).astype(np.float32)


def encode_boxes(gt_boxes: np.ndarray,
                 gt_labels: np.ndarray,
                 anchors: np.ndarray,
                 iou_threshold: float = IOU_MATCH_THRESH):
    """
    Match ground-truth boxes to anchors and produce training targets.

    Algorithm:
      1. Compute IoU between every anchor and every GT box.
      2. Assign each anchor to the GT with highest IoU (if >= threshold).
      3. For each GT, guarantee its best-matching anchor is assigned (bipartite).
      4. Encode offsets only for positive (matched) anchors.

    Args:
        gt_boxes:       [N, 4] float32, [cx, cy, w, h] normalized
        gt_labels:      [N]    int32,   1-based class labels
        anchors:        [8732, 4]
        iou_threshold:  float

    Returns:
        loc_targets:    [8732, 4]  encoded offsets  (only valid for positives)
        cls_targets:    [8732]     int32; 0 = background
    """
    num_anchors = len(anchors)

    if len(gt_boxes) == 0:
        return (np.zeros((num_anchors, 4), dtype=np.float32),
                np.zeros(num_anchors,      dtype=np.int32))

    iou = compute_iou_np(anchors, gt_boxes)   # [8732, N]

    best_gt_idx = np.argmax(iou, axis=1)       # [8732]  best GT for each anchor
    best_gt_iou = np.max(iou, axis=1)          # [8732]

    # Guarantee: every GT gets at least its best-matching anchor
    best_anchor_per_gt = np.argmax(iou, axis=0)  # [N]
    for gt_i, anc_i in enumerate(best_anchor_per_gt):
        best_gt_idx[anc_i] = gt_i
        best_gt_iou[anc_i] = 1.0              # force positive

    # Assign class labels
    cls_targets = np.zeros(num_anchors, dtype=np.int32)
    pos_mask = best_gt_iou >= iou_threshold
    cls_targets[pos_mask] = gt_labels[best_gt_idx[pos_mask]]

    # Encode localization targets (zeros for negatives; ignored in loss)
    matched_gt   = gt_boxes[best_gt_idx]       # [8732, 4]
    loc_targets  = encode_offsets(matched_gt, anchors)
    loc_targets[~pos_mask] = 0.0

    return loc_targets, cls_targets


# ────────────────────────────── Decoding ─────────────────────────────────────

def decode_offsets(loc_pred, anchors_np: np.ndarray,
                   variances=ENCODE_VARIANCES):
    """
    Decode network predictions back to [cx, cy, w, h] boxes.

    Args:
        loc_pred:   [N, 4]  predicted offsets  (tf.Tensor or np.ndarray)
        anchors_np: [N, 4]  anchor boxes       (np.ndarray)

    Returns:
        boxes: tf.Tensor [N, 4] in [cx, cy, w, h], clipped to [0, 1]
    """
    anchors = tf.constant(anchors_np, dtype=tf.float32)

    cx = loc_pred[:, 0] * variances[0] * anchors[:, 2] + anchors[:, 0]
    cy = loc_pred[:, 1] * variances[0] * anchors[:, 3] + anchors[:, 1]
    w  = tf.exp(loc_pred[:, 2] * variances[1]) * anchors[:, 2]
    h  = tf.exp(loc_pred[:, 3] * variances[1]) * anchors[:, 3]

    boxes = tf.stack([cx, cy, w, h], axis=-1)
    return tf.clip_by_value(boxes, 0.0, 1.0)
