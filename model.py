"""
model.py — MobileNetV1-SSD architecture defined from scratch.

Structure:
  ┌─ MobileNetV1 Backbone ──────────────────────────────────────────────────┐
  │  conv0  (3×3, stride 2) →  150×150×32                                   │
  │  dw1..5 (depthwise-sep) →  38×38×256   ← feat_38 (4 anchors/cell)      │
  │  dw6..11                →  19×19×512   ← feat_19 (6 anchors/cell)      │
  │  dw12..13               →  10×10×1024  ← feat_10 (6 anchors/cell)      │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ Extra SSD Layers ───────────────────────────────────────────────────────┐
  │  extra1 (1×1 + 3×3 s2) →  5×5×512     ← feat_5  (6 anchors/cell)      │
  │  extra2 (1×1 + 3×3 s2) →  3×3×256     ← feat_3  (4 anchors/cell)      │
  │  extra3 (1×1 + 3×3 valid)→ 1×1×256   ← feat_1  (4 anchors/cell)      │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ Prediction Heads (per feature map) ────────────────────────────────────┐
  │  cls_head:  Conv2D → [B, H×W×A, num_classes]                            │
  │  loc_head:  Conv2D → [B, H×W×A, 4]                                      │
  └─────────────────────────────────────────────────────────────────────────┘
  Final outputs:
    cls_out: [B, 8732, num_classes]   (raw logits)
    loc_out: [B, 8732, 4]             (encoded offsets)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from config import NUM_CLASSES_WITH_BG, ANCHORS_PER_CELL, FEATURE_MAP_SIZES


# ──────────────────────────── Building Blocks ────────────────────────────────

def _conv_bn_relu6(x, filters, kernel_size=3, stride=1,
                   padding='same', name=''):
    """Standard conv → BN → ReLU6."""
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, padding=padding,
        use_bias=False, name=f'{name}_conv'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(6.0, name=f'{name}_relu')(x)
    return x


def _dw_sep_conv(x, pointwise_filters, stride=1, name=''):
    """
    MobileNet depthwise-separable block:
        DW Conv2D (3×3) → BN → ReLU6 → PW Conv2D (1×1) → BN → ReLU6
    """
    # Depthwise convolution (channel-wise spatial filtering)
    x = layers.DepthwiseConv2D(
        3, strides=stride, padding='same',
        use_bias=False, name=f'{name}_dw'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_dw_bn')(x)
    x = layers.ReLU(6.0, name=f'{name}_dw_relu')(x)

    # Pointwise convolution (cross-channel combination)
    x = layers.Conv2D(
        pointwise_filters, 1, padding='same',
        use_bias=False, name=f'{name}_pw'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_pw_bn')(x)
    x = layers.ReLU(6.0, name=f'{name}_pw_relu')(x)
    return x


def _prediction_head(feature, num_anchors: int, num_classes: int, name=''):
    """
    SSD detection head: two parallel 3×3 convolutions (no BN/activation).

    Returns:
        cls_pred: [B, H, W, num_anchors * num_classes]   raw logits
        loc_pred: [B, H, W, num_anchors * 4]             box offsets
    """
    cls_pred = layers.Conv2D(
        num_anchors * num_classes, 3,
        padding='same', name=f'{name}_cls'
    )(feature)
    loc_pred = layers.Conv2D(
        num_anchors * 4, 3,
        padding='same', name=f'{name}_loc'
    )(feature)
    return cls_pred, loc_pred


# ─────────────────────────── Model Builder ───────────────────────────────────

def build_mobilenet_ssd(num_classes: int = NUM_CLASSES_WITH_BG,
                        width: float = 1.0) -> Model:
    """
    Build the full MobileNetV1-SSD model.

    Args:
        num_classes: total classes including background (default: 21 for VOC)
        width:       MobileNet width multiplier (1.0 = full model)

    Returns:
        tf.keras.Model with two outputs:
            cls_out  [B, 8732, num_classes]   logits
            loc_out  [B, 8732, 4]             encoded box offsets
    """
    def f(n: int) -> int:
        """Apply width multiplier and clamp to minimum 1."""
        return max(1, int(n * width))

    # ── Input ────────────────────────────────────────────────────────────────
    inp = layers.Input(shape=(300, 300, 3), name='input_image')

    # ── MobileNetV1 Backbone ─────────────────────────────────────────────────
    # Stage 0:  300×300 → 150×150
    x = _conv_bn_relu6(inp, f(32), stride=2, name='conv0')

    # Stage 1:  150×150
    x = _dw_sep_conv(x, f(64), stride=1, name='dw1')

    # Stage 2:  150×150 → 75×75
    x = _dw_sep_conv(x, f(128), stride=2, name='dw2')
    x = _dw_sep_conv(x, f(128), stride=1, name='dw3')

    # Stage 3:  75×75 → 38×38
    x = _dw_sep_conv(x, f(256), stride=2, name='dw4')
    x = _dw_sep_conv(x, f(256), stride=1, name='dw5')
    feat_38 = x                                         # 38×38 × 256

    # Stage 4:  38×38 → 19×19  (5 extra 1-stride blocks for more capacity)
    x = _dw_sep_conv(x, f(512), stride=2, name='dw6')
    x = _dw_sep_conv(x, f(512), stride=1, name='dw7')
    x = _dw_sep_conv(x, f(512), stride=1, name='dw8')
    x = _dw_sep_conv(x, f(512), stride=1, name='dw9')
    x = _dw_sep_conv(x, f(512), stride=1, name='dw10')
    x = _dw_sep_conv(x, f(512), stride=1, name='dw11')
    feat_19 = x                                         # 19×19 × 512

    # Stage 5:  19×19 → 10×10
    x = _dw_sep_conv(x, f(1024), stride=2, name='dw12')
    x = _dw_sep_conv(x, f(1024), stride=1, name='dw13')
    feat_10 = x                                         # 10×10 × 1024

    # ── Extra SSD Layers ─────────────────────────────────────────────────────
    # 10×10 → 5×5
    x = _conv_bn_relu6(feat_10, f(256), kernel_size=1, name='extra1_1x1')
    x = _conv_bn_relu6(x,       f(512), stride=2,      name='extra1_3x3')
    feat_5 = x                                          # 5×5  × 512

    # 5×5 → 3×3
    x = _conv_bn_relu6(feat_5, f(128), kernel_size=1, name='extra2_1x1')
    x = _conv_bn_relu6(x,      f(256), stride=2,      name='extra2_3x3')
    feat_3 = x                                          # 3×3  × 256

    # 3×3 → 1×1  (use 'valid' padding so 3×3 kernel fully contracts 3×3 map)
    x = _conv_bn_relu6(feat_3, f(128), kernel_size=1, name='extra3_1x1')
    x = _conv_bn_relu6(x,      f(256), padding='valid', name='extra3_3x3')
    feat_1 = x                                          # 1×1  × 256

    # ── Detection Heads ──────────────────────────────────────────────────────
    features     = [feat_38, feat_19, feat_10, feat_5, feat_3, feat_1]
    cls_outputs  = []
    loc_outputs  = []

    for i, (feat, n_anchors, fmap_size) in enumerate(
            zip(features, ANCHORS_PER_CELL, FEATURE_MAP_SIZES)):

        cls_p, loc_p = _prediction_head(feat, n_anchors, num_classes,
                                        name=f'head{i}')

        # Reshape [B, H, W, A*C] → [B, H*W*A, C]
        n_boxes = fmap_size * fmap_size * n_anchors
        cls_p = layers.Reshape(
            (n_boxes, num_classes), name=f'cls_reshape{i}'
        )(cls_p)
        loc_p = layers.Reshape(
            (n_boxes, 4), name=f'loc_reshape{i}'
        )(loc_p)

        cls_outputs.append(cls_p)
        loc_outputs.append(loc_p)

    # Concatenate across all feature maps → [B, 8732, *]
    cls_out = layers.Concatenate(axis=1, name='cls_out')(cls_outputs)
    loc_out = layers.Concatenate(axis=1, name='loc_out')(loc_outputs)

    return Model(inputs=inp, outputs=[cls_out, loc_out],
                 name='MobileNetV1_SSD')
