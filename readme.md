# VesselView: A CNN for Segmentation of Vessels in High-Resolution Retinal Fundus Images

This repository contains the official implementation of VesselView, a U-Net–inspired convolutional neural network designed for the segmentation of blood vessels in high‑resolution retinal fundus images. The method is described in the following peer‑reviewed chapter:

Santos‑Mateos, R., Velev‑Santos, A., Pardo, X. M. (2026). VesselView: A CNN for Segmentation of Vessels in High‑Resolution Retinal Fundus Images. In: Pattern Recognition and Image Analysis (IbPRIA 2025). Lecture Notes in Computer Science (LNCS), vol 15938, pp. 258–270. Springer, Cham. [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-99568-2_21), [DOI](https://doi.org/10.1007/978-3-031-99568-2_21).

## Authors

- Roi Santos‑Mateos — Escola Técnica Superior de Enxeñaría (ETSE) and Universidade de Santiago de Compostela; CiTIUS
- Alexander Velev‑Santos — Escola Técnica Superior de Enxeñaría (ETSE) and Universidade de Santiago de Compostela; CiTIUS
- Xosé M. Pardo — CiTIUS, Universidade de Santiago de Compostela

## Abstract

Retinal fundus imaging provides a noninvasive view of the eye’s microvasculature, enabling early detection of ocular and systemic diseases. VesselView is a U‑Net–inspired CNN that uses double‑convolution residual blocks with large kernels, a deepened bottleneck, and carefully designed skip connections to segment vessels in full‑resolution fundus images. Evaluated on the FIVES dataset at 2048×2048 resolution, VesselView achieves strong overall performance—measured by ROC AUC—with particularly competitive results on glaucomatous and normal images. Qualitative comparisons indicate a balance of fewer missed vessels at the cost of more false positives relative to competing models, and an ablation study highlights the importance of the chosen skip connections for high‑resolution segmentation.

## Overview

VesselView is an advanced deep learning model for accurate segmentation of blood vessels in retinal fundus images. The neural network architecture employs a U-Net-like structure with custom residual blocks and specialized bottleneck layers to effectively capture vessel structures at different scales. The model has been tested on high-resolution images and achieves state-of-the-art performance on standard retinal vessel segmentation benchmarks.

## Repository Structure

The repository is organized as follows:

- `code/`
  - `models/`: Neural network model definitions
    - `roinet.py`: The main VesselView model implementation
    - `frnet.py`: An alternative model architecture (FRNet)
    - `common.py`: Common building blocks for neural networks
  - `training/`: Training-related code
    - `run_benchmark.py`: Main training and evaluation script
    - `loss.py`: Implementation of loss functions (Dice, CL-Dice, etc.)
    - `soft_skeleton.py`: Implementation of soft skeletonization for loss functions
  - `ds/`: Dataset handling
    - `dataset.py`: Dataset classes and data augmentation
    - `resizeDatasetTo512.py`: Utility to resize datasets to 512x512
  - `utils/`: Utility functions
    - `utils.py`: Helper functions for model training and visualization
  - `inference/`: Code for model inference
    - `inferroi.py`: Inference script for VesselView
    - `infer.py` and `infer_linux.py`: Platform-specific inference scripts
  - `evaluation/`: Model evaluation code
    - `evaluation.py`: Implementation of evaluation metrics
    - `plot_activations.ipynb`: Notebook for visualizing model activations
  - `config/`: Configuration files
    - `config.json`: Model and training configuration
  - `scripts/`: Shell scripts for training and deployment
    - Various `.sh` files for running training on slurm clusters

## Model Architecture

VesselView is a CNN-based architecture designed for precise blood vessel segmentation:

- **Encoder**: Series of convolutional blocks with residual connections that gradually reduce spatial dimensions while increasing feature channels
- **Bottleneck**: Multiple residual blocks for processing high-level features
- **Decoder**: Upsampling blocks with skip connections from encoder to recover spatial details
- **Custom Blocks**: Residual blocks with large kernel sizes (9x9 or 11x11) for capturing vessel-like structures

The model supports different configurations with varying depths and channel widths to balance between accuracy and computational efficiency.

## Training

The training procedure uses:

- **Loss Functions**: Primarily Dice loss, with support for other advanced losses such as CL-Dice
- **Data Augmentation**: Geometric transformations, elastic deformation, and intensity adjustments
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: Regular evaluation on validation set using Dice coefficient and other metrics

## Inference

For inference on new images, use the `inferroi.py` script:

```
python code/inference/inferroi.py --model_path /path/to/model --input_dir /path/to/images --output_dir /path/to/results
```

## Requirements

Required packages are listed in the `requeriments` directory. The core dependencies include:

- PyTorch 1.9+
- OpenCV
- NumPy
- scikit-image
- TensorBoard

## Citation

If you use this code in your research, please cite the VesselView chapter:

Santos‑Mateos, R., Velev‑Santos, A., Pardo, X. M. (2026). VesselView: A CNN for Segmentation of Vessels in High‑Resolution Retinal Fundus Images. In N. Gonçalves, H. P. Oliveira, J. A. Sánchez (Eds.), Pattern Recognition and Image Analysis (IbPRIA 2025). Lecture Notes in Computer Science, vol 15938 (pp. 258–270). Springer, Cham. [DOI](https://doi.org/10.1007/978-3-031-99568-2_21).

### BibTeX

```bibtex
@incollection{Santos-Mateos_2026_VesselView,
  author    = {Santos{-}Mateos, Roi and Velev{-}Santos, Alexander and Pardo, Xos{\'e} M.},
  title     = {VesselView: A CNN for Segmentation of Vessels in High-Resolution Retinal Fundus Images},
  booktitle = {Pattern Recognition and Image Analysis (IbPRIA 2025)},
  series    = {Lecture Notes in Computer Science},
  volume    = {15938},
  pages     = {258--270},
  publisher = {Springer, Cham},
  year      = {2026},
  doi       = {10.1007/978-3-031-99568-2_21},
  url       = {https://link.springer.com/chapter/10.1007/978-3-031-99568-2_21}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was developed at the University of Santiago de Compostela.
