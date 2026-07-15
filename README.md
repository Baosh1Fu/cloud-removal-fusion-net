# UNCRTAINTS-Mamba3D: Remote Sensing Cloud Removal Model Based on NIG Representation and Mamba Encoding

This project is an extension of the original [UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS) model. The goal is to improve the effectiveness of cloud removal in remote sensing images, especially the ability to model uncertainty. We introduced a stronger spatio-temporal modeling structure, improved the output method, and designed a network structure that fuses results from multiple models.

---

## Project Background and Improvements

Based on the original UnCRtainTS model, this project makes the following extensions:

- **Four-parameter NIG Output**: Modified the output from the original `mean + variance` to a more complete **Normal-Inverse-Gamma (NIG) distribution** with four parameters (`delta`, `gamma`, `alpha`, `beta`), supporting Evidential Learning.
- **mamba Architecture**: Designed a new network structure `mamba_fusion` that combines the spatial modeling capability of Conv3D with the temporal modeling capability of Mamba to better handle the time-series characteristics of remote sensing images.
- **fusion_net Model**: Built a fusion model `fusion_net` that performs uncertainty-weighted fusion of `uncrtaints` and `conv3d_mamba` outputs (based on our custom `monig_fusion` method).
- **Backward Compatibility**: All models retain both original versions and NIG four-parameter versions, maintaining compatibility with the original project and allowing them to run directly on the original training pipeline.

---

## Dataset Information

- **Dataset**: **SEN12MS-CR-TS**
- **Contents**: Multi-temporal Sentinel-2 optical images (potentially cloud-covered) and corresponding Sentinel-1 SAR images (cloud-free).
- **Task**: Remove clouds from optical images and reconstruct cloud-free images (Image Regression Task).

---

## Model Architecture Overview

- `uncrtaints`: The original UnCRtainTS model structure, suitable for processing multi-temporal images, often achieving better MSE.
- `conv3d_mamba`: A new module combining 3D Convolution and Mamba encoder for better spatio-temporal information processing, often achieving higher SSIM.
- `fusion_net`: A fusion structure that weights the outputs of the two models based on uncertainty, theoretically combining the advantages of both.

---

## Model Output Description

All networks utilize a four-parameter output (NIG distribution):
- `delta`: Predicted value.
- `gamma`: Precision adjustment parameter.
- `alpha`, `beta`: Determine the variance and uncertainty representation capability.

All losses are designed based on Evidential Learning theory, supporting simultaneous modeling of predicted values and their uncertainties.

---

## Training Command

The following command trains the fusion model `fusion_net`, which performs uncertainty-weighted fusion of the `uncrtaints` and `conv3d_mamba` branches:

```bash
python train_reconstruct.py \
  --experiment_name my_fusionnet_experiment \
  --model fusion_net \
  --root1 /path/to/SEN12MSCRTS_train \
  --root2 /path/to/SEN12MSCRTS_test \
  --root3 /path/to/SEN12MSCR \
  --input_t 3 \
  --epochs 15 \
  --lr 0.0005 \
  --batch_size 2 \
  --scale_by 10.0 \
  --loss fusion \
  --weight_reg 0.05 \
  --lambda1 1.0 \
  --lambda2 1.0 \
  --use_sar \
  --device cuda
```

- `--loss fusion`: the dedicated evidential fusion loss (`FusionCloudLoss`) used by `fusion_net`.
- `--lambda1` / `--lambda2`: loss weights for the Mamba branch and the UnCRtainTS branch, respectively.
- `--weight_reg`: regularization weight for the evidential (NIG) loss term.

To instead train a single branch, set `--model` to `uncrtaints` or `conv3d_mamba` and use `--loss MGNLL`.

---

## Validation / Evaluation Command

The following command evaluates a trained `fusion_net` checkpoint on the test split and logs the metrics to `test_metrics.json`. Model hyperparameters (`--model`, `--loss`, `--scale_by`, etc.) are automatically restored from the experiment's `conf.json`, so they do not need to be repeated:

```bash
python test_reconstruct.py \
  --experiment_name my_fusionnet_experiment \
  --weight_folder ./results \
  --root2 /path/to/SEN12MSCRTS_test \
  --region all \
  --input_t 3 \
  --resume_at -1 \
  --batch_size 2 \
  --device cuda
```

- `--weight_folder` / `--experiment_name`: locate the trained weights and `conf.json` (i.e. `./results/my_fusionnet_experiment`).
- `--resume_at -1`: load the checkpoint that performed best on the validation split (use a positive epoch number to load a specific epoch instead).
- `--region`: restrict evaluation to a continent, e.g. `europa`, `america`, `asiaEast`, `asiaWest`, or `all`.


---


