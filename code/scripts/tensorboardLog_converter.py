import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def extract_all_activations_to_single_csv(logdir, output_csv="activaciones_completas.csv"):
    ea = EventAccumulator(logdir)
    ea.Reload()

    scalars = ea.Tags().get("scalars", [])
    histograms = ea.Tags().get("histograms", [])

    data = []

    # Scalars (media, stddev, etc.)
    for tag in scalars:
        if "activation" in tag or "activations" in tag:
            for event in ea.Scalars(tag):
                data.append({
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "tag": tag,
                    "type": "scalar",
                    "value": event.value
                })

    # Histogramas (si los hay)
    for tag in histograms:
        if "activation" in tag or "activations" in tag:
            for event in ea.Histograms(tag):
                hist = event.histogram_value
                row = {
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "tag": tag,
                    "type": "histogram",
                    "min": hist.min,
                    "max": hist.max,
                    "num": hist.num,
                    "sum": hist.sum,
                    "sum_squares": hist.sum_squares
                    # Opcional: podrías expandir los buckets si quisieras
                }
                data.append(row)

    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Guardar a un solo CSV
    df.to_csv(output_csv, index=False)
    print(f" Activaciones exportadas a '{output_csv}'")


# Cambia esto a la ruta de tu archivo o carpeta de logs
log_path = "/mnt/netapp2/Store_uni/home/usc/ci/avs/tfg/tfg/fork-roi/fivesegmentor/runs/composite_loss_DistanceWeightedBCE70_Dice30_long/"
extract_all_activations_to_single_csv(log_path)