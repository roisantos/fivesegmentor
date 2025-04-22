#!/bin/bash
# ====================================================
# BLOQUE DE CONFIGURACIÓN CENTRAL
# ====================================================
# Parámetros generales
OUTPUT_PREFIX="composite_loss_VesselHaloLoss70_Dice30_long"
MODEL="SantosNet_GCh"
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

# Parámetros de augmentación
AUGMENT_OTROSFIVES="True"
AUGMENT_GEOMETRIC="False"
AUGMENT_ELASTIC="False"
AUGMENT_INTENSITY="False"
AUGMENT_GAMMA="False"
AUGMENT_NOISE="False"

RESTORMER="False"

# ====================================================
# Configuración de Composite Loss: todos los parámetros están aquí
# Cada elemento se define con el formato: 
#     LossName|peso|parametro1=value1,parametro2=value2,...
# Ejemplos:
#   Dice: sin parámetros, peso 0.5
#   FocalTversky: con parámetros alpha, beta, gamma, peso 0.3
#   Conex: sin parámetros, peso 0.2
# Puedes agregar o quitar líneas según la cantidad de funciones que quieras usar.
#     "Dice|0.8|"

COMPOSITE_LOSS_COMPONENTS=(
    "Dice|0.3|"
    "VesselHaloLoss|0.70|band_width=5,alpha=1.5"
)

# ====================================================
# Fin del bloque de configuración central.
# ====================================================

# Función que procesa el array de componentes y crea la cadena final.
function build_composite_loss_string() {
    local components=("$@")
    local total_weight=0
    local weight val

    # Calcular suma total de pesos
    for comp in "${components[@]}"; do
        # Usamos la barra vertical como delimitador
        IFS='|' read -r loss_name weight _ <<< "$comp"
        total_weight=$(echo "$total_weight + $weight" | bc -l)
    done

    if (( $(echo "$total_weight == 0" | bc -l) )); then
        echo "Error: La suma total de los pesos es 0." && exit 1
    fi

    local composite_string=""
    local norm_weight comp_str
    for comp in "${components[@]}"; do
        IFS='|' read -r loss_name weight params <<< "$comp"
        norm_weight=$(echo "scale=4; $weight / $total_weight" | bc -l)
        if [ -z "$params" ]; then
            comp_str="${loss_name}:weight=${norm_weight}"
        else
            comp_str="${loss_name}:${params},weight=${norm_weight}"
        fi

        if [ -z "$composite_string" ]; then
            composite_string="$comp_str"
        else
            composite_string="${composite_string};${comp_str}"
        fi
    done
    echo "$composite_string"
}

# Construir la cadena para composite loss
COMPOSITE_LOSS_STRING=$(build_composite_loss_string "${COMPOSITE_LOSS_COMPONENTS[@]}")
echo "Composite loss configuration string:"
echo "$COMPOSITE_LOSS_STRING"
echo ""

# ====================================================
# Preparación del directorio de salida y generación del script SBATCH
# ====================================================
mkdir -p runs/${OUTPUT_PREFIX}
SBATCH_FILE="runs/${OUTPUT_PREFIX}/submit_${OUTPUT_PREFIX}.sbatch"

cat <<EOF > "${SBATCH_FILE}"
#!/bin/bash
#SBATCH -J ${OUTPUT_PREFIX}
#SBATCH -o runs/${OUTPUT_PREFIX}/job_output_%j.log
#SBATCH -e runs/${OUTPUT_PREFIX}/job_error_%j.log
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p short
#SBATCH -t 20:00:00

module load cesga/2020
module load python/3.9.9

cd /mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor
source venv/bin/activate

python3 code/training/train.py \\
  -model "${MODEL}" \\
  -dataset "${DATASET}" \\
  --config "${CONFIG}" \\
  --epochs "${EPOCHS}" \\
  --early_stopping "${EARLY_STOP}" \\
  --batch_size "${BATCH_SIZE}" \\
  --num_workers "${NUM_WORKERS}" \\
  --lr "${LR}" \\
  --weight_decay "${WEIGHT_DECAY}" \\
  --logging "${LOGGING}" \\
  --output_prefix "${OUTPUT_PREFIX}" \\
  --thresh_value "${THRESH_VALUE}" \\
  --augment_geometric "${AUGMENT_GEOMETRIC}" \\
  --augment_elastic "${AUGMENT_ELASTIC}" \\
  --augment_intensity "${AUGMENT_INTENSITY}" \\
  --augment_gamma "${AUGMENT_GAMMA}" \\
  --augment_noise "${AUGMENT_NOISE}" \\
  --augment_otrosfives "${AUGMENT_OTROSFIVES}" \\
  --restormer "${RESTORMER}" \\
  --composite_loss_components "${COMPOSITE_LOSS_STRING}"
EOF

# ==== Submit to SLURM ====
sbatch "${SBATCH_FILE}"
