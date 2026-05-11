"""
dataset.py — Pascal VOC dataset pipeline for SSD training.

Supports:
  • VOC 2007 trainval + VOC 2012 trainval  → training split (16551 images)
  • VOC 2007 test                           → evaluation split (4952 images)

XML annotations are parsed, boxes converted to [cx, cy, w, h] normalized,
then encoded into SSD anchor targets via anchors.encode_boxes().
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from config import (
    DATA_DIR, INPUT_SIZE, BATCH_SIZE,
    CLASS_TO_IDX, NUM_ANCHORS
)
from anchors import generate_anchors, encode_boxes

# Pre-compute anchors once at module load time
_ANCHORS = generate_anchors()          # [8732, 4]


# ─────────────────────────── Annotation Parsing ──────────────────────────────

def parse_voc_xml(xml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a Pascal VOC XML annotation file.

    Returns:
        boxes:  [N, 4] float32  [cx, cy, w, h] normalized to [0, 1]
        labels: [N]    int32    1-based class indices (0 = background, unused here)
    """
    tree   = ET.parse(xml_path)
    root   = tree.getroot()
    size   = root.find('size')
    width  = float(size.find('width').text)
    height = float(size.find('height').text)

    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name not in CLASS_TO_IDX:
            continue                           # ignore unlisted classes

        # Skip difficult instances during training
        diff_el   = obj.find('difficult')
        difficult = int(diff_el.text) if diff_el is not None else 0
        if difficult:
            continue

        bb   = obj.find('bndbox')
        xmin = float(bb.find('xmin').text) / width
        ymin = float(bb.find('ymin').text) / height
        xmax = float(bb.find('xmax').text) / width
        ymax = float(bb.find('ymax').text) / height

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w  = xmax - xmin
        h  = ymax - ymin

        # Clamp to valid range
        cx, cy = np.clip(cx, 0, 1), np.clip(cy, 0, 1)
        w,  h  = np.clip(w,  0, 1), np.clip(h,  0, 1)

        boxes.append([cx, cy, w, h])
        labels.append(CLASS_TO_IDX[name])

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return (np.array(boxes,  dtype=np.float32),
            np.array(labels, dtype=np.int32))


def get_image_ids(voc_root: str, year: str, split: str) -> List[str]:
    """Read image IDs from VOC ImageSets/Main/<split>.txt"""
    txt = os.path.join(voc_root, f'VOC{year}', 'ImageSets', 'Main',
                       f'{split}.txt')
    with open(txt) as f:
        return [line.strip() for line in f if line.strip()]


# ─────────────────────────── Data Augmentation ───────────────────────────────

def _random_horizontal_flip(img: tf.Tensor,
                              boxes: np.ndarray) -> Tuple[tf.Tensor, np.ndarray]:
    """Flip image and adjust box cx coordinates."""
    if np.random.random() < 0.5:
        img = tf.image.flip_left_right(img)
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]   # flip cx
    return img, boxes


def _photo_metric_distortion(img: tf.Tensor) -> tf.Tensor:
    """Random brightness, contrast, saturation, hue adjustments."""
    img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.1)
    return tf.clip_by_value(img, 0.0, 255.0)


def _random_crop(img: tf.Tensor,
                 boxes: np.ndarray,
                 labels: np.ndarray,
                 min_iou: float = 0.5):
    """
    SSD-style random crop: sample a patch with IoU >= min_iou with all boxes.
    Falls back to the full image if no valid crop is found quickly.
    """
    if len(boxes) == 0:
        return img, boxes, labels

    h, w = img.shape[0], img.shape[1]

    for _ in range(50):
        scale = np.random.uniform(0.3, 1.0)
        aspect = np.random.uniform(0.5, 2.0)
        new_h  = int(h * scale)
        new_w  = int(w * min(1.0, scale * aspect))
        new_h  = max(1, min(new_h, h))
        new_w  = max(1, min(new_w, w))

        top  = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # Crop coordinates normalized
        c_x1, c_y1 = left / w, top / h
        c_x2, c_y2 = (left + new_w) / w, (top + new_h) / h

        # Check overlap with existing boxes (in [x1,y1,x2,y2])
        bx1 = boxes[:, 0] - boxes[:, 2] / 2
        by1 = boxes[:, 1] - boxes[:, 3] / 2
        bx2 = boxes[:, 0] + boxes[:, 2] / 2
        by2 = boxes[:, 1] + boxes[:, 3] / 2

        ix1 = np.maximum(bx1, c_x1)
        iy1 = np.maximum(by1, c_y1)
        ix2 = np.minimum(bx2, c_x2)
        iy2 = np.minimum(by2, c_y2)

        inter = (np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1))
        area  = (bx2 - bx1) * (by2 - by1)
        iou   = inter / (area + 1e-10)

        if np.all(iou >= min_iou):
            # Crop the image
            img_cropped = img[top:top + new_h, left:left + new_w]

            # Adjust boxes to crop coordinates
            cx = (boxes[:, 0] - c_x1) / (c_x2 - c_x1)
            cy = (boxes[:, 1] - c_y1) / (c_y2 - c_y1)
            bw = boxes[:, 2] / (c_x2 - c_x1)
            bh = boxes[:, 3] / (c_y2 - c_y1)

            new_boxes = np.stack([cx, cy, bw, bh], axis=-1)
            # Keep only boxes whose center falls inside the crop
            valid = ((cx > 0) & (cx < 1) & (cy > 0) & (cy < 1))
            if valid.any():
                return img_cropped, new_boxes[valid], labels[valid]

    return img, boxes, labels


# ─────────────────────────── Dataset Class ───────────────────────────────────

class VOCDataset:
    """
    Pascal VOC dataset.  Loads images + annotations, applies optional
    augmentation, then encodes targets against SSD anchors.
    """

    def __init__(self, voc_root: str,
                 samples: List[Tuple[str, str]],
                 augment: bool = False):
        """
        Args:
            voc_root: path to  data/VOCdevkit/
            samples:  list of (year, image_id)  e.g. [('2007', '000032'), ...]
            augment:  enable SSD-style data augmentation during training
        """
        self.voc_root = voc_root
        self.samples  = samples
        self.augment  = augment

    def _load_image(self, year: str, img_id: str) -> tf.Tensor:
        path = os.path.join(self.voc_root, f'VOC{year}',
                            'JPEGImages', f'{img_id}.jpg')
        raw  = tf.io.read_file(path)
        img  = tf.image.decode_jpeg(raw, channels=3)      # uint8
        return img

    def _preprocess(self, img: tf.Tensor) -> tf.Tensor:
        """Resize and normalize to [-1, 1] (matches MobileNet convention)."""
        img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img

    def load_sample(self, year: str, img_id: str):
        """
        Load one (image, loc_targets, cls_targets) tuple.

        Returns:
            img:         [300, 300, 3]  float32, normalized to [-1, 1]
            loc_targets: [8732, 4]      float32
            cls_targets: [8732]         int32
        """
        ann_path = os.path.join(self.voc_root, f'VOC{year}',
                                'Annotations', f'{img_id}.xml')
        gt_boxes, gt_labels = parse_voc_xml(ann_path)

        img = self._load_image(year, img_id)

        if self.augment:
            img_np = img.numpy().astype(np.float32)      # keep uint8 range

            # Photo metric distortion (on float image 0-255)
            img = _photo_metric_distortion(
                tf.cast(img, tf.float32))

            # Random horizontal flip
            img, gt_boxes = _random_horizontal_flip(img, gt_boxes)

            # Random crop
            img_np2 = img.numpy().astype(np.uint8)
            img_t   = tf.cast(img_np2, tf.float32)
            img, gt_boxes, gt_labels = _random_crop(
                img.numpy(), gt_boxes, gt_labels)
            img = tf.constant(img, dtype=tf.float32)

        img = self._preprocess(img)

        # Encode GT to anchor targets
        loc_targets, cls_targets = encode_boxes(gt_boxes, gt_labels, _ANCHORS)
        return img, loc_targets, cls_targets

    # ── tf.data.Dataset ──────────────────────────────────────────────────────

    def as_tf_dataset(self) -> tf.data.Dataset:
        """Build a tf.data.Dataset from a Python generator over samples."""

        def _generator():
            for year, img_id in self.samples:
                try:
                    img, loc_t, cls_t = self.load_sample(year, img_id)
                    yield img, loc_t, cls_t
                except Exception as exc:
                    # Log and skip corrupted / missing samples
                    print(f"[Dataset] Skipping {year}/{img_id}: {exc}")
                    continue

        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(NUM_ANCHORS, 4),            dtype=tf.float32),
                tf.TensorSpec(shape=(NUM_ANCHORS,),              dtype=tf.int32),
            ),
        )


# ─────────────────────────── Public API ──────────────────────────────────────

def build_dataset(split: str = 'train',
                  batch_size: int = BATCH_SIZE,
                  voc_root: str = DATA_DIR) -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data.Dataset.

    Training split  : VOC 2007 trainval + VOC 2012 trainval  (≈16 551 images)
    Eval split      : VOC 2007 test                           (≈4 952 images)

    Args:
        split:      'train' or 'val'
        batch_size: number of samples per batch
        voc_root:   path to VOCdevkit/

    Returns:
        tf.data.Dataset yielding (images, loc_targets, cls_targets)
    """
    is_train = (split == 'train')
    samples  = []

    if is_train:
        for year in ['2007', '2012']:
            ids = get_image_ids(voc_root, year, 'trainval')
            samples += [(year, i) for i in ids]
    else:
        ids = get_image_ids(voc_root, '2007', 'test')
        samples = [('2007', i) for i in ids]

    print(f"[Dataset] '{split}' split: {len(samples)} images")

    ds = VOCDataset(voc_root, samples, augment=is_train).as_tf_dataset()

    if is_train:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
