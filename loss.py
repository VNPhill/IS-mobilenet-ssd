"""
loss.py — SSD Multibox Loss.

Components:
  1. Localization loss  — Smooth L1 on positive anchors only
  2. Classification loss — Cross-entropy with Hard Negative Mining
                           (neg : pos = NEG_POS_RATIO : 1 by default 3:1)

Reference: Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016.
"""

import tensorflow as tf
from config import NEG_POS_RATIO


# ──────────────────────────── Smooth L1 ──────────────────────────────────────

def smooth_l1(pred: tf.Tensor, target: tf.Tensor,
              beta: float = 1.0) -> tf.Tensor:
    """
    Element-wise Smooth L1 (Huber) loss.

    L(x) = 0.5 * x^2 / beta          if |x| < beta
            |x| - 0.5 * beta          otherwise

    Args:
        pred, target: tensors of the same shape
        beta:         transition point between quadratic / linear regions

    Returns:
        element-wise loss tensor (same shape as inputs)
    """
    diff = tf.abs(pred - target)
    loss = tf.where(diff < beta,
                    0.5 * diff ** 2 / beta,
                    diff - 0.5 * beta)
    return loss


# ──────────────────────────── SSD Loss Class ─────────────────────────────────

class SSDLoss:
    """
    SSD Multibox loss callable.

    Usage:
        criterion = SSDLoss()
        total, cls_l, loc_l = criterion(cls_pred, loc_pred,
                                        cls_targets, loc_targets)
    """

    def __init__(self, neg_pos_ratio: int = NEG_POS_RATIO,
                 loc_weight: float = 1.0):
        """
        Args:
            neg_pos_ratio: maximum ratio of negatives to positives (hard mining)
            loc_weight:    weight applied to localization loss term (alpha)
        """
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_weight    = loc_weight

    def __call__(self,
                 cls_pred:    tf.Tensor,
                 loc_pred:    tf.Tensor,
                 cls_targets: tf.Tensor,
                 loc_targets: tf.Tensor):
        """
        Compute the SSD loss for a batch.

        Args:
            cls_pred:    [B, 8732, num_classes]  raw logits
            loc_pred:    [B, 8732, 4]            predicted offsets
            cls_targets: [B, 8732]               int32, 0 = background
            loc_targets: [B, 8732, 4]            ground-truth encoded offsets

        Returns:
            total_loss, cls_loss, loc_loss  — all scalar tensors
        """
        num_anchors = tf.shape(cls_pred)[1]

        # ── Positive mask ────────────────────────────────────────────────────
        pos_mask  = cls_targets > 0                                  # [B,8732]
        pos_float = tf.cast(pos_mask, tf.float32)
        num_pos   = tf.reduce_sum(pos_float, axis=1)                 # [B]

        # ── 1. Localization Loss (positive anchors only) ──────────────────────
        loc_l_per_anchor = tf.reduce_sum(
            smooth_l1(loc_pred, loc_targets), axis=-1)               # [B,8732]
        loc_loss_per_img = tf.reduce_sum(
            loc_l_per_anchor * pos_float, axis=1)                    # [B]

        # ── 2. Classification Loss with Hard Negative Mining ─────────────────

        # Softmax cross-entropy for every anchor
        cls_loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=cls_targets, logits=cls_pred)                     # [B,8732]

        # Zero out positives so they are never selected as hard negatives
        neg_loss = cls_loss_all * tf.cast(~pos_mask, tf.float32)     # [B,8732]

        # Number of negatives to keep: min(3 * num_pos, total_anchors - num_pos)
        num_neg = tf.minimum(
            tf.cast(self.neg_pos_ratio, tf.float32) * num_pos,
            tf.cast(num_anchors, tf.float32) - num_pos
        )                                                             # [B]
        num_neg = tf.cast(num_neg, tf.int32)

        # Rank each anchor by its negative loss (descending → rank 0 = highest loss)
        # argsort gives ordered indices; argsort of argsort gives rank
        sorted_idx = tf.argsort(neg_loss, axis=1,
                                direction='DESCENDING')              # [B,8732]
        rank       = tf.argsort(sorted_idx, axis=1)                  # [B,8732]

        # Build binary mask: keep only top-K negatives
        hard_neg_mask = tf.cast(
            tf.cast(rank, tf.float32) < tf.cast(num_neg[:, tf.newaxis], tf.float32),
            tf.float32
        )                                                             # [B,8732]

        # Total classification loss = positives + hard negatives
        cls_loss_per_img = tf.reduce_sum(
            cls_loss_all * (pos_float + hard_neg_mask), axis=1)      # [B]

        # ── Normalize by number of positive matches ───────────────────────────
        normalizer = tf.maximum(num_pos, 1.0)                        # [B]

        loc_loss_norm = tf.reduce_sum(loc_loss_per_img / normalizer)
        cls_loss_norm = tf.reduce_sum(cls_loss_per_img / normalizer)

        total_loss = cls_loss_norm + self.loc_weight * loc_loss_norm
        return total_loss, cls_loss_norm, loc_loss_norm
