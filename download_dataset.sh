#!/usr/bin/env bash
# download_dataset.sh — Download Pascal VOC 2007 & 2012 datasets
#
# Total download size: ~6.7 GB
#   VOC2007 trainval : 460 MB
#   VOC2007 test     : 451 MB
#   VOC2012 trainval : 1.9 GB
#
# Final structure after extraction:
#   data/VOCdevkit/
#     VOC2007/
#       Annotations/   (XML bounding-box labels)
#       ImageSets/     (train / val / test splits)
#       JPEGImages/    (images)
#     VOC2012/
#       Annotations/
#       ImageSets/
#       JPEGImages/

set -euo pipefail

DEST="data"
mkdir -p "$DEST"

echo "========================================================"
echo " Downloading Pascal VOC 2007 + 2012 datasets"
echo " Destination: $DEST/VOCdevkit/"
echo "========================================================"

# ── VOC 2007 trainval ──────────────────────────────────────────────────────
echo ""
echo "[1/3] VOC2007 trainval (~460 MB) ..."
wget -c --show-progress \
  "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" \
  -O "$DEST/VOCtrainval_06-Nov-2007.tar"

# ── VOC 2007 test ──────────────────────────────────────────────────────────
echo ""
echo "[2/3] VOC2007 test (~451 MB) ..."
wget -c --show-progress \
  "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" \
  -O "$DEST/VOCtest_06-Nov-2007.tar"

# ── VOC 2012 trainval ─────────────────────────────────────────────────────
echo ""
echo "[3/3] VOC2012 trainval (~1.9 GB) ..."
wget -c --show-progress \
  "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" \
  -O "$DEST/VOCtrainval_11-May-2012.tar"

# ── Extract ────────────────────────────────────────────────────────────────
echo ""
echo "Extracting archives ..."
tar -xf "$DEST/VOCtrainval_06-Nov-2007.tar" -C "$DEST/"
tar -xf "$DEST/VOCtest_06-Nov-2007.tar"     -C "$DEST/"
tar -xf "$DEST/VOCtrainval_11-May-2012.tar" -C "$DEST/"

echo ""
echo "Done!  Dataset is ready at $DEST/VOCdevkit/"
echo ""
echo "Verify with:"
echo "  ls data/VOCdevkit/VOC2007/JPEGImages/ | wc -l   # expect 9963"
echo "  ls data/VOCdevkit/VOC2012/JPEGImages/ | wc -l   # expect 17125"
