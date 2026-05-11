"""
evaluate.py — Evaluation on Pascal VOC 2007 test set.

Computes mean Average Precision (mAP) following the VOC 2010+ protocol
(area under the precision-recall curve, not 11-point interpolation).

Run:
    python evaluate.py

Expected output:
    aeroplane       AP=0.XXXX
    bicycle         AP=0.XXXX
    ...
    mAP@0.50: 0.XXXX
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   
from collections import defaultdict
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from model   import build_mobilenet_ssd
from dataset import get_image_ids, parse_voc_xml
from anchors import generate_anchors, decode_offsets
from config  import (
    DATA_DIR, INPUT_SIZE, CHECKPOINT_DIR,
    NUM_CLASSES, NUM_CLASSES_WITH_BG,
    VOC_CLASSES, IDX_TO_CLASS,
)
from utils import Tee

_ANCHORS = generate_anchors()     # [8732, 4]

def setup_results_dir():
    base_dir = "results"
    pr_dir   = os.path.join(base_dir, "pr_curves")
    sample_dir = os.path.join(base_dir, "samples")

    os.makedirs(pr_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    return base_dir, pr_dir, sample_dir

def draw_detections(image, boxes, scores, labels, save_path):
    """
    image: [H, W, 3] uint8
    boxes: normalized [x1, y1, x2, y2]
    """
    h, w, _ = image.shape

    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    ax = plt.gca()

    # bright distinct colors
    colors = plt.cm.get_cmap('tab10', 10)

    for box, score, label in zip(boxes, scores, labels):
        if score < 0.2:
            continue

        x1, y1, x2, y2 = box
        x1 *= w; x2 *= w
        y1 *= h; y2 *= h

        color = colors(label % 10)  # cycle colors

        # thicker, visible box
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=3   # thicker = easier to see
        )
        ax.add_patch(rect)

        cls_name = VOC_CLASSES[label - 1]

        # visible text with background
        ax.text(
            x1,
            y1 - 5,
            f"{cls_name} {score:.2f}",
            fontsize=9,
            color='white',
            bbox=dict(
                facecolor=color,
                alpha=0.8,
                pad=2,
                edgecolor='none'
            )
        )

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

# ──────────────────────────── Post-processing ────────────────────────────────

def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def postprocess(cls_logits: np.ndarray,
                loc_offsets,
                anchors: np.ndarray,
                conf_threshold: float = 0.01,
                nms_iou: float = 0.45,
                max_dets: int = 200
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode predictions, apply confidence threshold, per-class NMS.

    Args:
        cls_logits:  [8732, num_classes] raw logits
        loc_offsets: [8732, 4]           predicted encoded offsets (Tensor or ndarray)
        anchors:     [8732, 4]           anchor boxes [cx, cy, w, h]

    Returns:
        det_boxes:   [K, 4] float32  [x1, y1, x2, y2] normalized
        det_scores:  [K]    float32
        det_labels:  [K]    int32    (1-based class index)
    """
    probs = tf.nn.softmax(cls_logits, axis=-1).numpy()   # [8732, 21]

    # Decode box offsets → [cx, cy, w, h] → [x1, y1, x2, y2]
    boxes_cxcywh = decode_offsets(loc_offsets, anchors).numpy()  # [8732, 4]
    boxes_xyxy   = np.clip(_cxcywh_to_xyxy(boxes_cxcywh), 0.0, 1.0)

    all_boxes, all_scores, all_labels = [], [], []

    for cls_idx in range(1, NUM_CLASSES_WITH_BG):       # skip background (0)
        scores = probs[:, cls_idx]
        mask   = scores > conf_threshold
        if not np.any(mask):
            continue

        filtered_boxes  = boxes_xyxy[mask].astype(np.float32)
        filtered_scores = scores[mask].astype(np.float32)

        # TF non-maximum suppression
        keep = tf.image.non_max_suppression(
            filtered_boxes, filtered_scores,
            max_output_size=max_dets,
            iou_threshold=nms_iou,
        ).numpy()

        all_boxes.extend(filtered_boxes[keep].tolist())
        all_scores.extend(filtered_scores[keep].tolist())
        all_labels.extend([cls_idx] * len(keep))

    if len(all_boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros(0,     dtype=np.float32),
                np.zeros(0,     dtype=np.int32))

    return (np.array(all_boxes,  dtype=np.float32),
            np.array(all_scores, dtype=np.float32),
            np.array(all_labels, dtype=np.int32))


# ──────────────────────────── IoU (scalar) ───────────────────────────────────

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-10)


# ──────────────────────────── AP Computation ─────────────────────────────────

def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute VOC 2010+ Average Precision (area under smoothed PR curve).
    Precision is max-smoothed backwards, then area is integrated.
    """
    mrec = np.concatenate(([0.0], recall,    [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Smooth precision backwards
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate
    change_pts = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[change_pts + 1] - mrec[change_pts]) * mpre[change_pts + 1])
    return float(ap)


# ──────────────────────────── Main mAP loop ──────────────────────────────────

def compute_map(model: tf.keras.Model,
                voc_root: str = DATA_DIR,
                iou_threshold: float = 0.5) -> float:
    """
    Run full-image inference on VOC 2007 test and compute mAP.

    Args:
        model:         trained Keras model
        voc_root:      path to VOCdevkit/
        iou_threshold: IoU threshold for a detection to be counted as TP

    Returns:
        mAP scalar
    """
    ids = get_image_ids(voc_root, '2007', 'test')
    print(f"[Eval] Evaluating on {len(ids)} test images …")

    _, _, sample_dir = setup_results_dir()

    NUM_SAMPLES = 10
    sample_ids = set(ids[:NUM_SAMPLES]) 
        
    # Accumulators: per-class lists of (score, img_id, box)
    detections    = defaultdict(list)   # cls_idx → [(score, img_id, box)]
    ground_truths = defaultdict(list)   # cls_idx → [(img_id, box)]

    for img_id in tqdm(ids, desc="Evaluating images", leave=False):
        img_path = os.path.join(voc_root, 'VOC2007', 'JPEGImages',
                                f'{img_id}.jpg')
        ann_path = os.path.join(voc_root, 'VOC2007', 'Annotations',
                                f'{img_id}.xml')

        # ── Preprocess ───────────────────────────────────────────────────────
        raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        img = img[tf.newaxis]                              # [1, 300, 300, 3]

        # ── Inference ────────────────────────────────────────────────────────
        cls_pred, loc_pred = model(img, training=False)
        cls_pred = cls_pred[0].numpy()                     # [8732, 21]
        loc_pred = loc_pred[0]                             # Tensor [8732, 4]

        # ── Decode + NMS ─────────────────────────────────────────────────────
        det_boxes, det_scores, det_labels = postprocess(
            cls_pred, loc_pred, _ANCHORS)

        # ── Save sample detections ─────────────────────
        if img_id in sample_ids:
            raw_img = tf.io.read_file(img_path)
            vis_img = tf.image.decode_jpeg(raw_img, channels=3).numpy().astype(np.uint8)

            save_path = os.path.join(sample_dir, f"{img_id}.png")

            draw_detections(
                vis_img,
                det_boxes,
                det_scores,
                det_labels,
                save_path
            )
        
        for box, score, label in zip(det_boxes, det_scores, det_labels):
            detections[int(label)].append((float(score), img_id, box))

        # ── Ground-truth ─────────────────────────────────────────────────────
        gt_boxes, gt_labels = parse_voc_xml(ann_path)
        for box, label in zip(gt_boxes, gt_labels):
            # Convert [cx, cy, w, h] → [x1, y1, x2, y2]
            xyxy = np.array([
                box[0] - box[2] / 2, box[1] - box[3] / 2,
                box[0] + box[2] / 2, box[1] + box[3] / 2,
            ], dtype=np.float32)
            ground_truths[int(label)].append((img_id, xyxy))

    # ── Per-class AP ─────────────────────────────────────────────────────────
    aps = []
    pr_curves = {} 
    print()
    for cls_idx in tqdm(range(1, NUM_CLASSES_WITH_BG), desc="Computing AP", leave=False):
        cls_name = VOC_CLASSES[cls_idx - 1]
        cls_dets = sorted(detections[cls_idx], key=lambda x: -x[0])
        cls_gts  = ground_truths[cls_idx]

        if not cls_gts:
            print(f"  {cls_name:<15s}  (no GT, skipped)")
            continue

        # Group GTs by image for fast lookup
        gt_by_img = defaultdict(list)
        for (img_id, box) in cls_gts:
            gt_by_img[img_id].append({'box': box, 'matched': False})

        num_gt = len(cls_gts)
        tp = np.zeros(len(cls_dets), dtype=np.float32)
        fp = np.zeros(len(cls_dets), dtype=np.float32)

        for d, (score, img_id, det_box) in enumerate(cls_dets):
            gts = gt_by_img.get(img_id, [])
            if not gts:
                fp[d] = 1
                continue

            ious     = [_iou_xyxy(det_box, g['box']) for g in gts]
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]

            if best_iou >= iou_threshold and not gts[best_idx]['matched']:
                tp[d] = 1
                gts[best_idx]['matched'] = True
            else:
                fp[d] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall    = cum_tp / (num_gt + 1e-10)
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)

        pr_curves[cls_name] = (recall, precision)
        
        ap = _voc_ap(recall, precision)
        aps.append(ap)
        print(f"  {cls_name:<15s}  AP={ap:.4f}")

    mAP = float(np.mean(aps)) if aps else 0.0
    print(f"\n  mAP@{iou_threshold:.2f}: {mAP:.4f}")
    
    results_txt = os.path.join("results", "map_results.txt")

    with open(results_txt, "w") as f:
        f.write(f"mAP@{iou_threshold:.2f}: {mAP:.4f}\n")
        for i, ap in enumerate(aps):
            f.write(f"{VOC_CLASSES[i]:<15s} AP={ap:.4f}\n")
            
    # ── Save PR curves ─────────────────────────────
    _, pr_dir = setup_results_dir()

    for cls_name, (recall, precision) in pr_curves.items():
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {cls_name}")
        plt.grid()

        save_path = os.path.join(pr_dir, f"{cls_name}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"[Eval] PR curves saved to: {pr_dir}")

    return mAP


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == '__main__':
    model = build_mobilenet_ssd(num_classes=NUM_CLASSES_WITH_BG)

    # ckpt = os.path.join(CHECKPOINT_DIR, 'best_val.weights.h5')
    ckpt = os.path.join(CHECKPOINT_DIR, 'epoch_500.weights.h5')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run  python train.py  first."
        )

    base_dir, _, _ = setup_results_dir()
    log_path = os.path.join(base_dir, "eval.log")

    tee = Tee(log_path)

    os.sys.stdout = tee
    os.sys.stderr = tee 

    print(f"[Eval] Logging to {log_path}")

    print(f"[Eval] Loading weights from {ckpt}")
    model.load_weights(ckpt)

    # ── Run evaluation ────────────────────────────
    compute_map(model, voc_root=DATA_DIR)