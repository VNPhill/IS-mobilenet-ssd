import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model   import build_mobilenet_ssd
from anchors import generate_anchors, decode_offsets
from config  import (
    INPUT_SIZE, CHECKPOINT_DIR,
    NUM_CLASSES_WITH_BG,
    VOC_CLASSES
)

# ───────────────────────────────────────────────
# Setup
# ───────────────────────────────────────────────

_ANCHORS = generate_anchors()

def setup_inference_dir():
    base_dir = "results"
    infer_dir = os.path.join(base_dir, "inference")
    os.makedirs(infer_dir, exist_ok=True)
    return infer_dir


# ───────────────────────────────────────────────
# Utils
# ───────────────────────────────────────────────

def _cxcywh_to_xyxy(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def postprocess(cls_logits, loc_offsets,
                conf_threshold=0.2,
                nms_iou=0.45,
                max_dets=100):

    probs = tf.nn.softmax(cls_logits, axis=-1).numpy()
    boxes_cxcywh = decode_offsets(loc_offsets, _ANCHORS).numpy()
    boxes_xyxy = np.clip(_cxcywh_to_xyxy(boxes_cxcywh), 0.0, 1.0)

    all_boxes, all_scores, all_labels = [], [], []

    for cls_idx in range(1, NUM_CLASSES_WITH_BG):
        scores = probs[:, cls_idx]
        mask = scores > conf_threshold
        if not np.any(mask):
            continue

        b = boxes_xyxy[mask]
        s = scores[mask]

        keep = tf.image.non_max_suppression(
            b, s,
            max_output_size=max_dets,
            iou_threshold=nms_iou
        ).numpy()

        all_boxes.extend(b[keep])
        all_scores.extend(s[keep])
        all_labels.extend([cls_idx] * len(keep))

    if len(all_boxes) == 0:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int32)
        )

    return (
        np.array(all_boxes, np.float32),
        np.array(all_scores, np.float32),
        np.array(all_labels, np.int32)
    )


# ───────────────────────────────────────────────
# Drawing
# ───────────────────────────────────────────────

def draw_and_save(image, boxes, scores, labels, save_path):
    h, w, _ = image.shape

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    ax = plt.gca()

    colors = plt.cm.get_cmap('tab10', 10)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 *= w; x2 *= w
        y1 *= h; y2 *= h

        color = colors(label % 10)

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=3
        )
        ax.add_patch(rect)

        cls_name = VOC_CLASSES[label - 1]

        ax.text(
            x1,
            y1 - 5,
            f"{cls_name} {score:.2f}",
            color='white',
            fontsize=9,
            bbox=dict(facecolor=color, alpha=0.8, pad=2)
        )

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# ───────────────────────────────────────────────
# Inference
# ───────────────────────────────────────────────

def run_inference(image_path):
    # Load model
    model = build_mobilenet_ssd(num_classes=NUM_CLASSES_WITH_BG)

    ckpt = os.path.join(CHECKPOINT_DIR, 'epoch_480.weights.h5')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[Infer] Loading weights from {ckpt}")
    model.load_weights(ckpt)

    # Read image
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    orig = img.numpy().astype(np.uint8)

    img_resized = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
    img_resized = tf.cast(img_resized, tf.float32) / 127.5 - 1.0
    img_resized = img_resized[tf.newaxis]

    # Forward pass
    cls_pred, loc_pred = model(img_resized, training=False)
    cls_pred = cls_pred[0].numpy()
    loc_pred = loc_pred[0]

    boxes, scores, labels = postprocess(cls_pred, loc_pred)

    # Save output
    infer_dir = setup_inference_dir()

    filename = os.path.basename(image_path)
    save_path = os.path.join(infer_dir, filename)

    draw_and_save(orig, boxes, scores, labels, save_path)

    print(f"[Infer] Saved result to: {save_path}")


# ───────────────────────────────────────────────

if __name__ == "__main__":
    # 🔴 change this path to your image
    image_path = "inference_input/2 people.jpg"

    run_inference(image_path)
    
    