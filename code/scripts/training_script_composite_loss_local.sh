#!/bin/bash
# ====================================================
# BLOQUE DE CONFIGURACIÓN CENTRAL
# ====================================================
# Parámetros generales
OUTPUT_PREFIX="testLosses_HaloClDiceLoss_100"
MODEL="SantosNet_GCh_lite"
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


#       "FocalTversky|0.0|alpha=0.2,beta=0.8,gamma=0.5"
#       "SoftCLDiceLossStrict|0.5|penalty_power=5.0,smooth=1e-6"
#       "Dice|0.15|"
#       "DistanceWeightedBCE|0.20|sigma=4.0"
#       "VesselHaloLoss|0.50|band_width=5,alpha=1.5"
#       "HaloCLDiceLoss|0.30|band_width=5,alpha=0.7,beta=0.3,iter=5"


COMPOSITE_LOSS_COMPONENTS=(
    "HaloCLDiceLoss|0.30|band_width=5,alpha=0.7,beta=0.3,iter=25"
)

# ====================================================
# Función que procesa el array de componentes y crea la cadena final.
function build_composite_loss_string() {
    local components=("$@")
    local total_weight=0
    local weight

    # Calcular la suma total de pesos
    for comp in "${components[@]}"; do
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
# Ejecutar localmente el script de entrenamiento
# ====================================================
python3 code/training/train.py \
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
  --restormer "$RESTORMER" \
  --composite_loss_components "$COMPOSITE_LOSS_STRING"
