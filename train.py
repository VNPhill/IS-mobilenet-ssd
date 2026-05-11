"""
train.py — Training loop for MobileNetV1-SSD on Pascal VOC.

Run:
    python train.py

Features:
  • SGD + momentum optimizer with step-decay learning rate schedule
  • L2 weight decay applied to all Conv2D / DepthwiseConv2D kernels
  • SSD Multibox Loss (Smooth L1 + Hard Negative Mining cross-entropy)
  • Best-model checkpoint (by validation loss) + periodic epoch saves
  • TensorBoard scalar logging
  • Resume from last checkpoint if present
"""

import os
import time
import tensorflow as tf

from utils   import *
from model   import build_mobilenet_ssd
from loss    import SSDLoss
from dataset import build_dataset
from config  import (
    NUM_CLASSES_WITH_BG, NUM_EPOCHS, BATCH_SIZE,
    LR_INIT, LR_STEPS, MOMENTUM, WEIGHT_DECAY,
    CHECKPOINT_DIR, LOG_DIR,
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)


# ──────────────────────────── LR Schedule ────────────────────────────────────

def _get_lr(epoch: int) -> float:
    """Step-decay schedule: divide by 10 at each milestone epoch."""
    lr = LR_INIT
    for milestone in LR_STEPS:
        if epoch >= milestone:
            lr *= 0.1
    return lr


# ────────────────────────────── Main ─────────────────────────────────────────

def train():
    # ── Build model ──────────────────────────────────────────────────────────
    model = build_mobilenet_ssd(num_classes=NUM_CLASSES_WITH_BG, width=1.0)
    model.summary(line_length=100)
    print(f"\n  Total parameters: {model.count_params():,}")
    print(f"  Total anchors:    {model.output[0].shape[1]}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = build_dataset(split='train', batch_size=BATCH_SIZE)
    val_ds   = build_dataset(split='val',   batch_size=BATCH_SIZE)

    # ── Loss & optimizer ─────────────────────────────────────────────────────
    criterion = SSDLoss(loc_weight=1.0)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=LR_INIT,
        momentum=MOMENTUM,
        nesterov=False,
    )

    # Collect all weight tensors (kernels only) for L2 regularization
    l2_kernels = [v for v in model.trainable_variables
                  if 'kernel' in v.name]

    # ── TensorBoard writers ──────────────────────────────────────────────────
    train_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, 'train'))
    val_writer   = tf.summary.create_file_writer(os.path.join(LOG_DIR, 'val'))

    # ── Optional: resume from last periodic checkpoint ───────────────────────
    start_epoch = 0
    resume_ckpt = _find_latest_checkpoint(CHECKPOINT_DIR)
    if resume_ckpt:
        print(f"[Train] Resuming from {resume_ckpt}")
        model.load_weights(resume_ckpt)
        # Infer epoch number from filename if possible
        try:
            base = os.path.basename(resume_ckpt)          # e.g. epoch_080.weights.h5
            start_epoch = int(base.split('_')[1].split('.')[0])
            print(f"[Train] Starting at epoch {start_epoch + 1}")
        except Exception:
            pass

    # ── Training step (compiled as tf.function for speed) ────────────────────

    @tf.function
    def train_step(images, loc_t, cls_t):
        with tf.GradientTape() as tape:
            cls_pred, loc_pred = model(images, training=True)

            task_loss, cls_l, loc_l = criterion(
                cls_pred, loc_pred, cls_t, loc_t)

            # L2 regularization: WEIGHT_DECAY * sum(w^2) / 2
            l2_loss = WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(k) for k in l2_kernels])

            total_loss = task_loss + l2_loss

        grads = tape.gradient(total_loss, model.trainable_variables)

        # Gradient clipping to stabilize early training
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=10.0)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, cls_l, loc_l

    @tf.function
    def val_step(images, loc_t, cls_t):
        cls_pred, loc_pred = model(images, training=False)
        total_loss, cls_l, loc_l = criterion(
            cls_pred, loc_pred, cls_t, loc_t)
        return total_loss, cls_l, loc_l

    # ── Epoch loop ───────────────────────────────────────────────────────────

    best_val_loss = float('inf')

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_display = epoch + 1

        # Update learning rate
        new_lr = _get_lr(epoch)
        optimizer.learning_rate.assign(new_lr)

        # ── Training ─────────────────────────────────────────────────────────
        train_loss_sum = 0.0
        train_cls_sum  = 0.0
        train_loc_sum  = 0.0
        num_train_steps = 0
        t_epoch_start   = time.time()

        for step, (images, loc_t, cls_t) in enumerate(train_ds):
            total_l, cls_l, loc_l = train_step(images, loc_t, cls_t)

            train_loss_sum += float(total_l)
            train_cls_sum  += float(cls_l)
            train_loc_sum  += float(loc_l)
            num_train_steps += 1

            if step % 50 == 0:
                print(
                    f"  Ep {epoch_display:3d} step {step:4d} | "
                    f"total={total_l:.4f}  cls={cls_l:.4f}  loc={loc_l:.4f} "
                    f"lr={new_lr:.2e}"
                )

        mean_train_loss = train_loss_sum / max(1, num_train_steps)
        mean_train_cls  = train_cls_sum  / max(1, num_train_steps)
        mean_train_loc  = train_loc_sum  / max(1, num_train_steps)
        elapsed = time.time() - t_epoch_start

        # ── Validation ───────────────────────────────────────────────────────
        val_loss_sum = 0.0
        num_val_steps = 0

        for images, loc_t, cls_t in val_ds:
            total_l, _, _ = val_step(images, loc_t, cls_t)
            val_loss_sum += float(total_l)
            num_val_steps += 1

        mean_val_loss = val_loss_sum / max(1, num_val_steps)

        # ── Logging ──────────────────────────────────────────────────────────
        print(
            f"Epoch {epoch_display:3d}/{NUM_EPOCHS} | "
            f"train={mean_train_loss:.4f} (cls={mean_train_cls:.4f} "
            f"loc={mean_train_loc:.4f}) | "
            f"val={mean_val_loss:.4f} | "
            f"time={elapsed:.0f}s"
        )

        with train_writer.as_default():
            tf.summary.scalar('loss/total', mean_train_loss, step=epoch)
            tf.summary.scalar('loss/cls',   mean_train_cls,  step=epoch)
            tf.summary.scalar('loss/loc',   mean_train_loc,  step=epoch)
            tf.summary.scalar('lr',          new_lr,          step=epoch)

        with val_writer.as_default():
            tf.summary.scalar('loss/total', mean_val_loss, step=epoch)

        # ── Checkpointing ────────────────────────────────────────────────────
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, 'best_val.weights.h5')
            model.save_weights(best_path)
            print(f"  ✓ Best model saved  (val_loss={best_val_loss:.4f})")

        if epoch_display % 20 == 0:
            periodic_path = os.path.join(
                CHECKPOINT_DIR, f'epoch_{epoch_display:03d}.weights.h5')
            model.save_weights(periodic_path)

    print("\n[Train] Training complete.")
    print(f"[Train] Best validation loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────── Utilities ───────────────────────────────────────

def _find_latest_checkpoint(ckpt_dir: str):
    """Return the path to the most recent epoch_NNN checkpoint, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted([
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith('epoch_') and f.endswith('.weights.h5')
    ])
    return candidates[-1] if candidates else None


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == '__main__':
    # ==================== SETUP DUAL LOGGING ====================
    log_filename = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)

    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee          # Also capture errors and warnings

    print("=" * 80)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_path}")
    print("=" * 80)
    # ===========================================================

    # Use all available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[Train] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[Train] No GPU found — training on CPU (will be slow).")

    try:
        train()
    finally:
        # Ensure everything is written to disk
        tee.flush()
        tee.close()
        print(f"\n[Train] Log file saved: {log_path}")
        
        
        
        