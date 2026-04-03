# Real-Time Video Defogging System Based on Deep Learning Algorithm

## 1. Project Overview

This package contains the deep learning model and related files for laparoscopic image defogging / desmoking.

The current model predicts a **3-channel smoke veil map** `F` from an input foggy RGB image `I`, and then reconstructs the final enhanced image through a fixed post-processing pipeline.

This package is prepared for **hardware deployment discussion and verification**, with special focus on:

- network architecture
- layer-by-layer operations and parameters
- INT8 post-training quantization (PTQ)
- input / output data format
- reconstruction equations

---

## 2. Model Summary

- **Model name:** `LaparoUNet`
- **Framework:** PyTorch
- **Original precision:** FP32
- **Quantized precision:** INT8 PTQ (backbone verified in PyTorch)
- **Input image:** RGB laparoscopic foggy image
- **Input tensor shape:** `[B, 3, H, W]`
- **Recommended deployment resolution:** `480 × 640`
- **Output tensor shape:** `[B, 3, H, W]`
- **Output meaning:** predicted smoke veil map `F`

The final enhanced image is not directly produced by the neural network.  
Instead, the network first predicts `F`, and then the following reconstruction is applied:
$$
\alpha = \text{mean}(F)
$$

$$
L = I - \alpha F
$$

$$
J = \frac{L - L_{\min}}{L_{\max} - L_{\min} + \varepsilon}
$$

where:

- `I`: input foggy image
- `F`: predicted smoke veil
- `L`: intermediate restored image
- `J`: final enhanced image

---

## 3. Folder Structure

```text
laparo_unet_package/
├─ README.md
├─ network_architecture.md
├─ model/
│  └─ model.py
├─ utils/
│  └─ reconstruction.py
├─ weights/
│  ├─ laparo_unet_fp32.pth
│  └─ laparo_unet_ptq_state_dict.pth
├─ scripts/
│  ├─ inference.py
│  └─ ptq_batch_eval.py
└─ samples/
   ├─ input.png
   ├─ F_fp32.png
   ├─ F_int8.png
   ├─ J_fp32.png
   ├─ J_int8.png
   └─ J_diff_x10.png
```

## 4. Main Files

`model/model.py`

Defines the neural network `LaparoUNet`.

`utils/reconstruction.py`

Defines the reconstruction pipeline:

- `compute_alpha(F)`
- `compute_L(I, F, alpha)`
- `linear_stretch(L)`
- `reconstruct(I, F)`

`weights/laparo_unet_fp32.pth`

Original FP32 trained checkpoint.

`weights/laparo_unet_ptq_state_dict.pth`

INT8 PTQ model state_dict for PyTorch-side verification.

`docs/network_architecture.md`

Detailed layer-by-layer network description for hardware implementation reference.

`exported_params/`

This folder contains the exported quantized parameters for hardware-side reference.

It includes:

- layer-wise quantized integer weights
- bias terms
- weight quantization scales
- weight quantization zero points
- layer metadata
- a description document explaining the meaning and relationship of these exported files

Important note:

The `.pth` file is mainly for PyTorch-side loading and verification.
 For hardware implementation, the `exported_params/` folder is more directly useful, because it provides layer-wise parameter files and quantization metadata in an easier-to-parse form.

`exported_params/quantized_weight_description.md`

Explains the role, data type, and calculation relationship of:

- `weight_int.npy`
- `weight_scales.npy`
- `weight_zero_points.npy`
- `bias.npy`
- `weight_meta.json`
- `meta.json`

------

## 5. Input and Output Format

### Input

- RGB image
- pixel range normalized to `[0, 1]`
- tensor format: `[B, 3, H, W]`
- current recommended deployment resolution: `480 × 640`

### Output

- predicted smoke veil `F`
- tensor format: `[B, 3, H, W]`
- output range approximately in `(0, 1)` due to final `Sigmoid`

------

## 6. Preprocessing

Input image loading procedure:

1. read image using OpenCV
2. convert from BGR to RGB
3. normalize by dividing by 255
4. convert to tensor
5. permute to channel-first format `[C, H, W]`
6. add batch dimension to obtain `[1, C, H, W]`

------

## 7. Postprocessing / Reconstruction

After the network predicts `F`, reconstruction is performed as follows:

### Step 1: Global alpha

$$
\alpha = \text{mean}(F)
$$

### Step 2: Intermediate image

$$
L = I - \alpha F
$$

### Step 3: Linear stretch

$$
J = \frac{L - L_{\min}}{L_{\max} - L_{\min} + \varepsilon}
$$

Notes:

- `alpha` is currently computed as the global mean of `F`
- `L_min` and `L_max` are computed per image across all channels
- `J` is clamped to `[0,1]`

------

## 8. Quantization Status

The original model is trained in **FP32**.

INT8 **post-training quantization (PTQ)** has been tested successfully in PyTorch for the main network backbone.
 The quantized model preserves the output image quality well in current experiments.

Important notes for hardware discussion:

- convolution layers are quantization-friendly
- ReLU layers are quantization-friendly
- `Sigmoid`, bilinear `Upsample`, and `Concat` are relatively more deployment-sensitive
- reconstruction operations (`mean`, `min`, `max`, linear stretch) are currently kept as floating-point post-processing in algorithm verification
- The exported quantized layer parameters are stored in `exported_params/` for hardware-side parsing and implementation reference.

## 9. Current Conclusion

The current model has been successfully validated in both FP32 and INT8 PTQ settings in PyTorch.
 The INT8 PTQ model shows very small numerical difference from the FP32 model on tested images, and the final reconstructed image quality is visually very close to the FP32 result.

This indicates that the current model has good potential for hardware deployment, although several deployment-sensitive operations still need hardware-side confirmation.