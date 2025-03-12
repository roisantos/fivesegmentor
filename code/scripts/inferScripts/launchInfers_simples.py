import os
from inferCore import run_inference_on_directory

# Define las rutas base (actualiza según tu entorno)
image_dir = r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES/test/image"
label_dir = r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/dataset/FIVES/test/label"
base_output_dir = os.path.join('inference_results')

# Diccionario con las 6 configuraciones y la ruta a su archivo .pth correspondiente.
# Asegúrate de actualizar las rutas de los pesos (.pth)
models_config = {
    "RoiNetTest1bottleneck_simple": {
        "config": {
            "type": "RoiNetTest1bottleneck",
            "ch_in": 3,
            "ch_out": 1,
            "cls_init_block": "SimpleResBlock",
            "cls_conv_block": "SimpleResBlock"
        },
        "model_path": r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/code/scripts/inferScripts/inputs/RoiNet1bottleneck_Fives_simple_result_2025-03-10_19-53-19/RoiNetTest1bottleneck_simple_Dice/model_best.pth"
    },
    "RoiNetTest3bottleneck_simple": {
        "config": {
            "type": "RoiNetTest3bottleneck",
            "ch_in": 3,
            "ch_out": 1,
            "cls_init_block": "SimpleResBlock",
            "cls_conv_block": "SimpleResBlock"
        },
        "model_path": r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/code/scripts/inferScripts/inputs/RoiNet3bottleneck_Fives_simple_result_2025-03-10_18-48-13/RoiNetTest3bottleneck_simple_Dice/model_best.pth"
    },
    "RoiNetTest2bottleneck_simple": {
        "config": {
            "type": "RoiNetTest2bottleneck",
            "ch_in": 3,
            "ch_out": 1,
            "cls_init_block": "SimpleResBlock",
            "cls_conv_block": "SimpleResBlock"
        },
        "model_path": r"/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/code/scripts/inferScripts/inputs/RoiNet2bottleneck_Fives_simple_result_2025-03-10_18-54-52/RoiNetTest2bottleneck_simple_Dice/model_best.pth"
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
