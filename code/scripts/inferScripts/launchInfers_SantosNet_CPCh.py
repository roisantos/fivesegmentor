import os
from inferCore import run_inference_on_directory

# Define las rutas base (actualiza según tu entorno)
image_dir = r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES/test/image"
label_dir = r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES/test/label"
base_output_dir = os.path.join('inference_results')

# Diccionario con las 6 configuraciones y la ruta a su archivo .pth correspondiente.
# Asegúrate de actualizar las rutas de los pesos (.pth)
models_config = {
    "SantosNet_GCh": {
        "config": {
            "type": "SantosNet_GCh",
            "ch_in": 3,
            "ch_out": 1,
            "cls_init_block": "ResidualBlock",
            "cls_conv_block": "ResidualBlock",
            "custom_weights": [0.1,0.8,0.1]
        },
        "model_path": r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/runs/SantosNet_GCh_/SantosNet_GCh_Dice/model_best.pth"
    }
}

for model_name, model_info in models_config.items():
    print(f"\nEjecutando inferencia para {model_name}...")
    # Se define un directorio de salida específico para cada modelo
    output_dir = os.path.join(base_output_dir, model_name)
    run_inference_on_directory(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        model_config=model_info["config"],
        model_path=model_info["model_path"]
    )
