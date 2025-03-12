#!/bin/bash
# Script para lanzar la inferencia con RoiNet9 sobre el dataset FIVES test

# Variables modificables
MODEL_PATH="/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/run_benchmark_runs/RoiNet9_FIVES_Dice_sameAsOtrosfives_slurm_result_2025-03-05_19-14-41/RoiNet9_Dice/model_best.pth"         # Ruta al archivo .pth del modelo entrenado
DATA_PATH="/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES"                # Ruta raíz del dataset FIVES

# Ruta del script de inferencia (asegúrate de que esté en el directorio correcto)
SCRIPT="code/scripts/roinetInfer.py"

# Mostrar las variables que se usarán
echo "Usando modelo: $MODEL_PATH"
echo "Dataset FIVES en: $DATA_PATH"
echo "Ejecutando: python $SCRIPT -model $MODEL_PATH --data_path $DATA_PATH"

# Lanzar la inferencia
python3 $SCRIPT -model "$MODEL_PATH" --data_path "$DATA_PATH"
