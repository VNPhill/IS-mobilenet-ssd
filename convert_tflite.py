"""
convert_tflite.py — Convert trained MobileNet-SSD to TFLite with post-processing.

Output format (identical to original pretrained coco_ssd_mobilenet_v1):
  Output 0 → detection_boxes    [1, MAX_DET, 4]   float32  [ymin,xmin,ymax,xmax]
  Output 1 → detection_classes  [1, MAX_DET]       float32  0-based class index
  Output 2 → detection_scores   [1, MAX_DET]       float32  confidence in [0,1]
  Output 3 → num_detections     [1]                float32  actual count

Run:
    python convert_tflite.py --checkpoint checkpoints/best_model.weights.h5
    python convert_tflite.py --checkpoint checkpoints/best_model.weights.h5 --quantize
"""

import os
import argparse
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from model   import build_mobilenet_ssd
from anchors import generate_anchors
from config  import (
    CHECKPOINT_DIR, INPUT_SIZE,
    NUM_CLASSES_WITH_BG, ENCODE_VARIANCES, DATA_DIR,
)

os.makedirs('outputs', exist_ok=True)
_ANCHORS_NP = generate_anchors()   # [8732, 4]


# ─────────────────────────── Post-processing Wrapper ─────────────────────────

class SSDPostProcess(tf.Module):
    """
    Wraps the raw SSD Keras model with:
        decode offsets → softmax → NMS → pad to fixed MAX_DET

    Outputs the same 4-tensor format as the original pretrained SSD model.
    """

    def __init__(self, base_model, anchors_np: np.ndarray,
                 max_det: int = 10,
                 conf_thresh: float = 0.3,
                 nms_thresh:  float = 0.45):
        super().__init__()
        self.base        = base_model
        self.anchors     = tf.constant(anchors_np, dtype=tf.float32)
        self.max_det     = max_det
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.var_xy      = tf.constant(ENCODE_VARIANCES[0], dtype=tf.float32)
        self.var_wh      = tf.constant(ENCODE_VARIANCES[1], dtype=tf.float32)

    def _decode_boxes(self, loc_offsets):
        """Decode SSD offsets → [ymin, xmin, ymax, xmax] normalized."""
        a   = self.anchors
        cx  = loc_offsets[:, 0] * self.var_xy * a[:, 2] + a[:, 0]
        cy  = loc_offsets[:, 1] * self.var_xy * a[:, 3] + a[:, 1]
        w   = tf.exp(loc_offsets[:, 2] * self.var_wh) * a[:, 2]
        h   = tf.exp(loc_offsets[:, 3] * self.var_wh) * a[:, 3]
        xmin = tf.clip_by_value(cx - w / 2, 0.0, 1.0)
        ymin = tf.clip_by_value(cy - h / 2, 0.0, 1.0)
        xmax = tf.clip_by_value(cx + w / 2, 0.0, 1.0)
        ymax = tf.clip_by_value(cy + h / 2, 0.0, 1.0)
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)   # [8732, 4]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, INPUT_SIZE, INPUT_SIZE, 3], dtype=tf.float32)
    ])
    def __call__(self, x):
        cls_logits, loc_offsets = self.base(x, training=False)
        cls_logits  = cls_logits[0]    # [8732, num_classes]
        loc_offsets = loc_offsets[0]   # [8732, 4]

        boxes       = self._decode_boxes(loc_offsets)            # [8732, 4]
        probs       = tf.nn.softmax(cls_logits, axis=-1)
        fg_probs    = probs[:, 1:]                               # skip background
        best_scores = tf.reduce_max(fg_probs, axis=-1)           # [8732]
        best_classes = tf.cast(
            tf.argmax(fg_probs, axis=-1), dtype=tf.float32)      # [8732]

        selected = tf.image.non_max_suppression(
            boxes, best_scores,
            max_output_size=self.max_det,
            iou_threshold=self.nms_thresh,
            score_threshold=self.conf_thresh,
        )

        out_boxes   = tf.gather(boxes,        selected)
        out_scores  = tf.gather(best_scores,  selected)
        out_classes = tf.gather(best_classes, selected)
        num_det     = tf.cast(tf.shape(selected)[0], tf.float32)

        pad = self.max_det - tf.shape(selected)[0]
        out_boxes   = tf.pad(out_boxes,   [[0, pad], [0, 0]])
        out_scores  = tf.pad(out_scores,  [[0, pad]])
        out_classes = tf.pad(out_classes, [[0, pad]])

        return (
            tf.reshape(out_boxes,   [1, self.max_det, 4]),   # boxes
            tf.reshape(out_classes, [1, self.max_det]),       # classes
            tf.reshape(out_scores,  [1, self.max_det]),       # scores
            tf.reshape([num_det],   [1]),                     # num_detections
        )


# ─────────────────────────── Calibration (int8 only) ─────────────────────────

# def _calibration_gen(data_dir: str = DATA_DIR, num_images: int = 200):
#     try:
#         import json
#         ann_path = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
#         img_dir  = os.path.join(data_dir, 'val2017')
#         with open(ann_path) as f:
#             fnames = [i['file_name'] for i in json.load(f)['images'][:num_images]]
#         for fname in fnames:
#             raw = tf.io.read_file(os.path.join(img_dir, fname))
#             img = tf.image.decode_jpeg(raw, channels=3)
#             img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
#             img = tf.cast(img, tf.float32) / 127.5 - 1.0
#             yield [img[tf.newaxis].numpy()]
#     except Exception as exc:
#         print(f"[Convert] Calibration dataset unavailable ({exc}). Using random data.")
#         for _ in range(num_images):
#             yield [np.random.uniform(-1, 1,
#                    (1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)]


def _calibration_gen(data_dir: str = DATA_DIR, num_images: int = 200):
    img_dir = os.path.join(data_dir, 'VOC2007/JPEGImages')

    if not os.path.exists(img_dir):
        raise ValueError(f"[Convert] JPEGImages folder not found: {img_dir}")

    fnames = [f for f in os.listdir(img_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(fnames) == 0:
        raise ValueError("[Convert] No images found for calibration.")

    fnames = fnames[:num_images]

    for fname in fnames:
        raw = tf.io.read_file(os.path.join(img_dir, fname))
        img = tf.image.decode_image(raw, channels=3)
        img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        yield [img[tf.newaxis].numpy()]
        

# ─────────────────────────── Conversion ──────────────────────────────────────

def convert(quantize: bool = False,
            checkpoint: str = None,
            output_path: str = None) -> str:

    fileName = checkpoint.rsplit('.', 2)[0].split('/', 1)[1]
    
    if checkpoint is None:
        checkpoint = os.path.join(CHECKPOINT_DIR, 'best_model.weights.h5')
    if output_path is None:
        output_path = (f'outputs/{fileName}_quant.tflite'
                       if quantize else f'outputs/{fileName}.tflite')

    # ── 1. Load base model ────────────────────────────────────────────────────
    print(f"[Convert] Loading checkpoint: {checkpoint}")
    base = build_mobilenet_ssd(num_classes=NUM_CLASSES_WITH_BG)
    base.load_weights(checkpoint)

    # ── 2. Build post-processing wrapper ─────────────────────────────────────
    wrapper = SSDPostProcess(base, _ANCHORS_NP, max_det=10,
                             conf_thresh=0.3, nms_thresh=0.45)

    # ── 3. Get concrete function & FREEZE all variables ───────────────────────
    #
    # convert_variables_to_constants_v2 converts every tf.Variable (including
    # BatchNorm moving_mean / moving_variance) into a tf.constant embedded
    # directly in the graph.  This is what kills the READ_VARIABLE error.
    #
    print("[Convert] Tracing and freezing variables ...")
    concrete_fn = wrapper.__call__.get_concrete_function(
        tf.TensorSpec([1, INPUT_SIZE, INPUT_SIZE, 3], dtype=tf.float32)
    )
    frozen_fn = convert_variables_to_constants_v2(concrete_fn)

    # ── 4. Configure TFLite converter from the frozen function ────────────────
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [frozen_fn]
    )

    if quantize:
        print("[Convert] Applying full-integer (uint8) quantization ...")
        converter.optimizations              = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback for NMS op
        ]
        converter.representative_dataset = _calibration_gen
    else:
        print("[Convert] Applying dynamic-range quantization ...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # ── 5. Convert & save ────────────────────────────────────────────────────
    tflite_bytes = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_bytes)

    size_mb = len(tflite_bytes) / (1024 * 1024)
    print(f"[Convert] Saved: {output_path}  ({size_mb:.2f} MB)")
    return output_path


# ─────────────────────────── Verification ────────────────────────────────────

def verify(tflite_path: str):
    """Run a dummy inference and print output shapes with pass/fail markers."""
    print(f"\n[Verify] {tflite_path}")
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_d   = interp.get_input_details()[0]
    out_ds = interp.get_output_details()

    print(f"  Input : shape={list(in_d['shape'])}  dtype={in_d['dtype'].__name__}")
    print(f"  Outputs ({len(out_ds)} tensors):")
    for od in out_ds:
        print(f"    [{od['index']}] {od['name']:<50s} shape={list(od['shape'])}  "
              f"dtype={od['dtype'].__name__}")

    # ── Dummy inference ───────────────────────────────────────────────────────
    dummy = np.random.uniform(-1, 1, in_d['shape']).astype(np.float32)
    if in_d['dtype'] == np.uint8:
        dummy = ((dummy + 1) * 127.5).astype(np.uint8)
    interp.set_tensor(in_d['index'], dummy)
    interp.invoke()

    # ── Find each output by shape, not by assumed index order ────────────────
    # (TFLite may reorder outputs; match by shape instead)
    results = {tuple(od['shape']): interp.get_tensor(od['index'])
               for od in out_ds}

    boxes   = results.get((1, 10, 4))
    classes = results.get((1, 10))
    scores  = results.get((1, 10))   # same shape as classes — one of the two
    num     = results.get((1,))

    ok = all(x is not None for x in [boxes, classes, num])
    print(f"\n  boxes={boxes.shape if boxes is not None else 'MISSING'}  "
          f"num_det={num}  "
          f"{'✓ All outputs present' if ok else '✗ Some outputs missing'}")

    # ── Map Flutter output indices correctly ─────────────────────────────────
    print("\n  Flutter output index mapping (use these in detection_service.dart):")
    for od in out_ds:
        shape = tuple(od['shape'])
        name  = ('detection_boxes'     if shape == (1, 10, 4) else
                 'num_detections'      if shape == (1,)        else
                 'detection_classes_or_scores')
        print(f"    index {od['index']} → {name}  shape={list(shape)}")

    return out_ds   # return so caller can inspect index ordering


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize',   action='store_true',
                        help='Full-integer (uint8) quantization')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .weights.h5 checkpoint')
    parser.add_argument('--output',     type=str, default=None,
                        help='Output .tflite path (default: outputs/detect.tflite)')
    args = parser.parse_args()

    out_path = convert(quantize=args.quantize,
                       checkpoint=args.checkpoint,
                       output_path=args.output)
    out_ds = verify(out_path)

    # Print final reminder about output index order
    print("\n[Done] Copy outputs/detect.tflite → your Flutter assets/models/detect.tflite")
    print("       The verify output above shows the exact index for each tensor.")
    print("       Update detection_service.dart if index order differs from 0=boxes,1=classes,2=scores,3=num")