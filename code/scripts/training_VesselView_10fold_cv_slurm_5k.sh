#!/bin/bash
MODEL="RoiNet2bottleneck"
DATASET="/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES_joined_ordered"
CONFIG="code/config/config.json"
EPOCHS=35
BATCH_SIZE=1
NUM_WORKERS=4
LR=1e-4
WEIGHT_DECAY=0.001
FOLDS=5





OUTPUT_PREFIX="VesselView_5fold_35epochs_5k"

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
#SBATCH -t 50:00:00

module load cesga/2020
module load python/3.9.9

cd /mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor
source venv/bin/activate

python3 code/training/VesselView_10fold_cv_5k.py \
  --dataset "${DATASET}" \
  --epochs "${EPOCHS}" \
  --bs "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --folds "${FOLDS}"

EOF

# ==== Submit to SLURM ====
sbatch "${SBATCH_FILE}"
