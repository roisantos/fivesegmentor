#!/bin/bash
MODEL="RoiNet2bottleneck"
DATASET="/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES_joined_ordered"
CONFIG="code/config/config.json"
EPOCHS=2
BATCH_SIZE=1
NUM_WORKERS=4
LR=1e-4
WEIGHT_DECAY=0.001
FOLDS=5

python3 code/training/VesselView_10fold_cv.py \
  --dataset "${DATASET}" \
  --epochs "${EPOCHS}" \
  --bs "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --folds "${FOLDS}"
