#!/bin/bash
MODEL="RoiNet2bottleneck"
DATASET="FIVES_joined_ordered"
CONFIG="code/config/config.json"
EPOCHS=2
BATCH_SIZE=1
NUM_WORKERS=8
LR=1e-4
WEIGHT_DECAY=0.001
FOLDS=2




OUTPUT_PREFIX="VesselView_2fold_2epochs"

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
#SBATCH -t 2:00:00

module load cesga/2020
module load python/3.9.9

cd /mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor
source venv/bin/activate

python3 code/training/VesselView_10fold_cv.py \
  -dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --folds "$FOLDS"

EOF

# ==== Submit to SLURM ====
sbatch "${SBATCH_FILE}"
