"""
config.py — All hyperparameters and constants for MobileNet-SSD training.

Architecture: MobileNetV1 backbone + SSD multi-scale detection heads
Dataset:      Pascal VOC 2007 + 2012  (20 classes)
Total anchors: 8732 (standard SSD configuration)
"""

# ─────────────────────────── Dataset / Classes ───────────────────────────────

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird',   'boat',        'bottle',
    'bus',       'car',     'cat',    'chair',        'cow',
    'diningtable','dog',    'horse',  'motorbike',    'person',
    'pottedplant','sheep',  'sofa',   'train',        'tvmonitor'
]

NUM_CLASSES        = len(VOC_CLASSES)          # 20 foreground classes
NUM_CLASSES_WITH_BG = NUM_CLASSES + 1          # 21 (index 0 = background)

# Maps class name → integer label (background = 0, VOC classes = 1..20)
CLASS_TO_IDX = {cls: idx + 1 for idx, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {idx + 1: cls for idx, cls in enumerate(VOC_CLASSES)}

# ──────────────────────────── SSD Anchor Config ──────────────────────────────

# Feature map spatial sizes at each detection scale
FEATURE_MAP_SIZES = [38, 19, 10, 5, 3, 1]

# Number of anchor boxes per spatial cell at each scale
#   38x38 → 4  (aspect ratios: 1, 1', 2, 1/2)
#   19x19 → 6  (aspect ratios: 1, 1', 2, 1/2, 3, 1/3)
#   ...
ANCHORS_PER_CELL = [4, 6, 6, 6, 4, 4]

# Total anchors: 5776 + 2166 + 600 + 150 + 36 + 4 = 8732
NUM_ANCHORS = sum(a * f * f for a, f in zip(ANCHORS_PER_CELL, FEATURE_MAP_SIZES))

# Anchor scales (one extra value is used for the "additional" scale at each level)
# s_k = scale at level k, s'_k = sqrt(s_k * s_{k+1}) for additional square anchor
ANCHOR_SCALES = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075]

# Extra aspect ratios per level (code also adds the inverse 1/ar automatically)
ANCHOR_ASPECT_RATIOS = [
    [2],       # 38x38 → adds 2 and 0.5  → total 4 anchors/cell
    [2, 3],    # 19x19 → adds 2,0.5,3,1/3 → total 6 anchors/cell
    [2, 3],    # 10x10
    [2, 3],    # 5x5
    [2],       # 3x3
    [2],       # 1x1
]

# Box encoding variance (standard SSD values)
ENCODE_VARIANCES = (0.1, 0.2)

# ─────────────────────────── Training Hyperparameters ────────────────────────

INPUT_SIZE     = 300            # SSD-300 input resolution
BATCH_SIZE     = 16
NUM_EPOCHS     = 500

LR_INIT        = 1e-2           # Initial SGD learning rate
LR_STEPS       = [100, 200, 300, 400]     # Epoch milestones: divide LR by 10x at each
MOMENTUM       = 0.9
WEIGHT_DECAY   = 5e-4           # L2 regularization weight

NEG_POS_RATIO  = 3              # Hard negative mining: keep 3× more negatives
IOU_MATCH_THRESH = 0.5          # IoU threshold to assign anchor as positive

# ─────────────────────────── Paths ───────────────────────────────────────────

DATA_DIR        = 'data/VOCdevkit'
CHECKPOINT_DIR  = 'checkpoints'
LOG_DIR         = 'logs'
TFLITE_PATH     = 'outputs/model.tflite'
