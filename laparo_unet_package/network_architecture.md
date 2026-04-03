
# Network Architecture Description

## 1. Overview

The network is a shallow U-Net style convolutional neural network named `LaparoUNet`.

Its purpose is **not** to directly output the final defogged image.  
Instead, it predicts a **3-channel smoke veil map** `F`, and the final restored image is obtained through a reconstruction pipeline.

Overall flow:

$$
I \xrightarrow{\text{LaparoUNet}} F
$$

$$
F \xrightarrow{\text{reconstruction}} L \xrightarrow{\text{linear stretch}} J
$$

where:

- `I`: input foggy image
- `F`: predicted smoke veil
- `L`: intermediate image
- `J`: final enhanced image

## 2. Network Type

The network consists of:

- encoder
- bottleneck
- decoder
- skip connections
- final output layer

The model uses:
- `3×3` convolution
- `1×1` convolution
- `ReLU`
- `MaxPool2d`
- bilinear `Upsample`
- `Sigmoid` at output

---

## 3. Basic Building Block

Each `ConvBlock(in_channels, out_channels)` contains:

1. `Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)`
2. `ReLU`
3. `Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)`
4. `ReLU`

This means each block preserves spatial resolution.

---

## 4. Input / Output

### Input
- shape: `[B, 3, 480, 640]`
- format: RGB
- value range: `[0,1]`

### Output
- shape: `[B, 3, 480, 640]`
- meaning: smoke veil map `F`
- final activation: `Sigmoid`

---

## 5. Layer-by-Layer Architecture

Below assumes input image size is `480 × 640`.

---

### 5.1 Encoder Stage 1

`enc1 = ConvBlock(3, 32)`

| Layer       | Operation | Parameters                   | Input Shape      | Output Shape     |
| ----------- | --------- | ---------------------------- | ---------------- | ---------------- |
| enc1.conv.0 | Conv2d    | in=3, out=32, k=3, s=1, p=1  | `[B,3,480,640]`  | `[B,32,480,640]` |
| enc1.conv.1 | ReLU      | -                            | `[B,32,480,640]` | `[B,32,480,640]` |
| enc1.conv.2 | Conv2d    | in=32, out=32, k=3, s=1, p=1 | `[B,32,480,640]` | `[B,32,480,640]` |
| enc1.conv.3 | ReLU      | -                            | `[B,32,480,640]` | `[B,32,480,640]` |

`pool1 = MaxPool2d(2)`

| Layer | Operation | Parameters | Input Shape      | Output Shape     |
| ----- | --------- | ---------- | ---------------- | ---------------- |
| pool1 | MaxPool2d | k=2, s=2   | `[B,32,480,640]` | `[B,32,240,320]` |

---

### 5.2 Encoder Stage 2

`enc2 = ConvBlock(32, 64)`

| Layer       | Operation | Parameters                   | Input Shape      | Output Shape     |
| ----------- | --------- | ---------------------------- | ---------------- | ---------------- |
| enc2.conv.0 | Conv2d    | in=32, out=64, k=3, s=1, p=1 | `[B,32,240,320]` | `[B,64,240,320]` |
| enc2.conv.1 | ReLU      | -                            | `[B,64,240,320]` | `[B,64,240,320]` |
| enc2.conv.2 | Conv2d    | in=64, out=64, k=3, s=1, p=1 | `[B,64,240,320]` | `[B,64,240,320]` |
| enc2.conv.3 | ReLU      | -                            | `[B,64,240,320]` | `[B,64,240,320]` |

`pool2 = MaxPool2d(2)`

| Layer | Operation | Parameters | Input Shape      | Output Shape     |
| ----- | --------- | ---------- | ---------------- | ---------------- |
| pool2 | MaxPool2d | k=2, s=2   | `[B,64,240,320]` | `[B,64,120,160]` |

---

### 5.3 Encoder Stage 3

`enc3 = ConvBlock(64, 128)`

| Layer       | Operation | Parameters                     | Input Shape       | Output Shape      |
| ----------- | --------- | ------------------------------ | ----------------- | ----------------- |
| enc3.conv.0 | Conv2d    | in=64, out=128, k=3, s=1, p=1  | `[B,64,120,160]`  | `[B,128,120,160]` |
| enc3.conv.1 | ReLU      | -                              | `[B,128,120,160]` | `[B,128,120,160]` |
| enc3.conv.2 | Conv2d    | in=128, out=128, k=3, s=1, p=1 | `[B,128,120,160]` | `[B,128,120,160]` |
| enc3.conv.3 | ReLU      | -                              | `[B,128,120,160]` | `[B,128,120,160]` |

---

### 5.4 Bottleneck

`bottleneck = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)`

| Layer      | Operation | Parameters                     | Input Shape       | Output Shape      |
| ---------- | --------- | ------------------------------ | ----------------- | ----------------- |
| bottleneck | Conv2d    | in=128, out=128, k=3, s=1, p=1 | `[B,128,120,160]` | `[B,128,120,160]` |

Note: this layer does not include ReLU in the current model definition.

---

### 5.5 Decoder Stage 1

`up1 = Upsample(scale_factor=2, mode='bilinear')`

| Layer | Operation | Parameters        | Input Shape       | Output Shape      |
| ----- | --------- | ----------------- | ----------------- | ----------------- |
| up1   | Upsample  | bilinear, scale=2 | `[B,128,120,160]` | `[B,128,240,320]` |

`cat1 = torch.cat([up1, x2], dim=1)`

| Layer | Operation     | Parameters  | Input Shape                        | Output Shape      |
| ----- | ------------- | ----------- | ---------------------------------- | ----------------- |
| cat1  | Concatenation | channel dim | `[B,128,240,320] + [B,64,240,320]` | `[B,192,240,320]` |

`dec1 = ConvBlock(128+64, 64)`

| Layer       | Operation | Parameters                    | Input Shape       | Output Shape     |
| ----------- | --------- | ----------------------------- | ----------------- | ---------------- |
| dec1.conv.0 | Conv2d    | in=192, out=64, k=3, s=1, p=1 | `[B,192,240,320]` | `[B,64,240,320]` |
| dec1.conv.1 | ReLU      | -                             | `[B,64,240,320]`  | `[B,64,240,320]` |
| dec1.conv.2 | Conv2d    | in=64, out=64, k=3, s=1, p=1  | `[B,64,240,320]`  | `[B,64,240,320]` |
| dec1.conv.3 | ReLU      | -                             | `[B,64,240,320]`  | `[B,64,240,320]` |

---

### 5.6 Decoder Stage 2

`up2 = Upsample(scale_factor=2, mode='bilinear')`

| Layer | Operation | Parameters        | Input Shape      | Output Shape     |
| ----- | --------- | ----------------- | ---------------- | ---------------- |
| up2   | Upsample  | bilinear, scale=2 | `[B,64,240,320]` | `[B,64,480,640]` |

`cat2 = torch.cat([up2, x1], dim=1)`

| Layer | Operation     | Parameters  | Input Shape                       | Output Shape     |
| ----- | ------------- | ----------- | --------------------------------- | ---------------- |
| cat2  | Concatenation | channel dim | `[B,64,480,640] + [B,32,480,640]` | `[B,96,480,640]` |

`dec2 = ConvBlock(64+32, 32)`

| Layer       | Operation | Parameters                   | Input Shape      | Output Shape     |
| ----------- | --------- | ---------------------------- | ---------------- | ---------------- |
| dec2.conv.0 | Conv2d    | in=96, out=32, k=3, s=1, p=1 | `[B,96,480,640]` | `[B,32,480,640]` |
| dec2.conv.1 | ReLU      | -                            | `[B,32,480,640]` | `[B,32,480,640]` |
| dec2.conv.2 | Conv2d    | in=32, out=32, k=3, s=1, p=1 | `[B,32,480,640]` | `[B,32,480,640]` |
| dec2.conv.3 | ReLU      | -                            | `[B,32,480,640]` | `[B,32,480,640]` |

---

### 5.7 Output Layer

`out_conv = Conv2d(32, 3, kernel_size=1, stride=1, padding=0)`

| Layer    | Operation | Parameters                  | Input Shape      | Output Shape    |
| -------- | --------- | --------------------------- | ---------------- | --------------- |
| out_conv | Conv2d    | in=32, out=3, k=1, s=1, p=0 | `[B,32,480,640]` | `[B,3,480,640]` |

`activation = Sigmoid()`

| Layer      | Operation | Parameters | Input Shape     | Output Shape    |
| ---------- | --------- | ---------- | --------------- | --------------- |
| activation | Sigmoid   | -          | `[B,3,480,640]` | `[B,3,480,640]` |

The final `Sigmoid` constrains the output `F` to approximately `(0,1)`.

---

## 6. Reconstruction Equations

The neural network output is `F`, not the final enhanced image.

The post-processing pipeline is:

### 6.1 Global alpha
$$
\alpha = \text{mean}(F)
$$

### 6.2 Intermediate image
$$
L = I - \alpha F
$$

### 6.3 Linear stretch
$$
J = \frac{L - L_{\min}}{L_{\max} - L_{\min} + \varepsilon}
$$

where:
- `I` is the input foggy image
- `F` is the predicted smoke veil
- `L` is the intermediate image
- `J` is the final enhanced image

Notes:
- `alpha` is computed as the global mean of `F`
- `L_min` and `L_max` are computed per image over all channels
- `J` is clamped to `[0,1]`

---

## 7. Important Hardware Notes

The following parts are relatively straightforward for hardware implementation:
- convolution
- ReLU
- max pooling

The following parts require extra attention:
- bilinear upsampling
- channel concatenation (`torch.cat`)
- final `Sigmoid`
- reconstruction post-processing (`mean`, `min`, `max`, linear stretch)

---

## 8. Quantization Notes

The original model is FP32.

INT8 post-training quantization (PTQ) has been validated in PyTorch for the backbone.  
The quantized model preserves output quality well on current test images.

For hardware implementation, besides the `.pth` file, it is recommended to provide:
- layer-wise weights
- bias
- quantization parameters
- tensor shape specification
- preprocessing / postprocessing equations