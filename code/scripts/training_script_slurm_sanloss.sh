#!/bin/bash

# ==== Userdefined variables ====
OUTPUT_PREFIX="SLPChg1.5"
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

# SanLoss parameters
LOSS="SanLoss"
ALPHA=0.2
BETA=0.8
GAMMA=1.5
ENTROPY_WEIGHT=0.7

# Augmentation options
AUGMENT_OTROSFIVES="True"
AUGMENT_GEOMETRIC="False"
AUGMENT_ELASTIC="False"
AUGMENT_INTENSITY="False"
AUGMENT_GAMMA="False"
AUGMENT_NOISE="False"

RESTORMER="False"

# ==== Prep output dir + SBATCH file path ====
mkdir -p runs/${OUTPUT_PREFIX}
SBATCH_FILE="runs/${OUTPUT_PREFIX}/submit_${OUTPUT_PREFIX}.sbatch"

# ==== Build SBATCH script dynamically ====
cat <<EOF > "${SBATCH_FILE}"
#!/bin/bash
#SBATCH -J ${OUTPUT_PREFIX}
#SBATCH -o runs/${OUTPUT_PREFIX}/job_output_%j.log
#SBATCH -e runs/${OUTPUT_PREFIX}/job_error_%j.log
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p short
#SBATCH -t 1-00:00:00

module load cesga/2020
module load python/3.9.9

cd /mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor
source venv/bin/activate

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
  --entropy_weight "${ENTROPY_WEIGHT}"
EOF

# ==== Make script executable ====
chmod +x "${SBATCH_FILE}"

# ==== Submit to SLURM ====
sbatch "${SBATCH_FILE}" 