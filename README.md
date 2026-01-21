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

The following command can be used to train the NIG version of the models (replace `--model` with `uncrtaints`, `conv3d_mamba`, or `fusion_net`):

```bash
python train_reconstruct.py \
  --experiment_name my_first_experiment \
  --model uncrtaints \
  --root1 /path/to/SEN12MSCRTS_train \
  --root2 /path/to/SEN12MSCRTS_test \
  --root3 /path/to/SEN12MSCR \
  --input_t 3 \
  --epochs 20 \
  --lr 0.001 \
  --batch_size 4 \
  --scale_by 10.0 \
  --loss MGNLL \
  --use_sar \
  --device cuda
```


---


