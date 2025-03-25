#!/bin/bash
# Example training launch script

# User-defined configuration variables
MODEL="RoiNetMultiSkip"
DATASET="FIVES"
CONFIG="code/config/config.json"
EPOCHS=300
EARLY_STOP=100
BATCH_SIZE=1
NUM_WORKERS=32
LR=1e-4
WEIGHT_DECAY=0.001
LOGGING="True"
OUTPUT_PREFIX="Test_Class_RoiNetMultiSkip"
THRESH_VALUE=100

# Loss function variables
LOSS="Dice"
#Parameters for FocalTversky
ALPHA=0.2
BETA=0.8
GAMMA=0.5

# Augmentation variables
AUGMENT_OTROSFIVES="True" 
AUGMENT_GEOMETRIC="False" 
AUGMENT_ELASTIC="False"
AUGMENT_INTENSITY="False"
AUGMENT_GAMMA="False"
AUGMENT_NOISE="False"

# Restormer
RESTORMER="False"



# Launch the training script with the specified parameters
python3 code/training/run_benchmark.py \
  -model "$MODEL" \
  -dataset "$DATASET" \
  --config "$CONFIG" \
  --epochs "$EPOCHS" \
  --early_stopping "$EARLY_STOP" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --logging "$LOGGING" \
  --output_prefix "$OUTPUT_PREFIX" \
  --thresh_value "$THRESH_VALUE" \
  --augment_geometric "$AUGMENT_GEOMETRIC" \
  --augment_elastic "$AUGMENT_ELASTIC" \
  --augment_intensity "$AUGMENT_INTENSITY" \
  --augment_gamma "$AUGMENT_GAMMA" \
  --augment_noise "$AUGMENT_NOISE" \
  --augment_otrosfives "$AUGMENT_OTROSFIVES" \
  --restormer "$RESTORMER"\
  --loss "$LOSS" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  --gamma "$GAMMA" \
