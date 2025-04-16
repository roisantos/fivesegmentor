#!/bin/bash

# ==== Userdefined variables ====
OUTPUT_PREFIX="DirectionalSanLoss_Test"
MODEL="SantosNet_PCh"  # Using the learnable channel fusion model
DATASET="FIVES"
CONFIG="code/config/config.json"
EPOCHS=300
EARLY_STOP=100
BATCH_SIZE=1
NUM_WORKERS=32
LR=1e-4
WEIGHT_DECAY=0.001
LOGGING="True"
THRESH_VALUE=100

# DirectionalSanLoss parameters
LOSS="DirectionalSanLoss"
ALPHA=0.2
BETA=0.8
GAMMA=1.5
DIRECTION_WEIGHT=0.7
KERNEL_SIZE=5

# Augmentation options
AUGMENT_OTROSFIVES="True"
AUGMENT_GEOMETRIC="False"
AUGMENT_ELASTIC="False"
AUGMENT_INTENSITY="False"
AUGMENT_GAMMA="False"
AUGMENT_NOISE="False"

RESTORMER="False"

# ==== Prep output dir ====
mkdir -p runs/${OUTPUT_PREFIX}

# ==== Run the training script ====
python3 code/training/run_benchmark.py \
  -model "${MODEL}" \
  -dataset "${DATASET}" \
  --config "${CONFIG}" \
  --epochs "${EPOCHS}" \
  --early_stopping "${EARLY_STOP}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --logging "${LOGGING}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  --thresh_value "${THRESH_VALUE}" \
  --augment_geometric "${AUGMENT_GEOMETRIC}" \
  --augment_elastic "${AUGMENT_ELASTIC}" \
  --augment_intensity "${AUGMENT_INTENSITY}" \
  --augment_gamma "${AUGMENT_GAMMA}" \
  --augment_noise "${AUGMENT_NOISE}" \
  --augment_otrosfives "${AUGMENT_OTROSFIVES}" \
  --restormer "${RESTORMER}" \
  --loss "${LOSS}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --direction_weight "${DIRECTION_WEIGHT}" \
  --kernel_size "${KERNEL_SIZE}" 