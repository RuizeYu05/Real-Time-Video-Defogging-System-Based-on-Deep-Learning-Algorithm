# Quantized Weight File Description

## 1. Purpose of This Document

This document explains the meaning, role, data type, and calculation relationship of the exported weight files for each quantized convolution layer.

The exported files are intended for hardware-side understanding and implementation of the INT8 PTQ model.

For each quantized convolution layer, the parameter folder typically contains:

- `weight_int.npy`
- `weight_scales.npy`
- `weight_zero_points.npy`
- `bias.npy`
- `weight_meta.json`
- `meta.json`

These files together describe the quantized convolution parameters and how they should be interpreted.

---

## 2. Overview

For a quantized convolution layer, the actual convolution no longer directly uses FP32 weights.

Instead, the main convolution kernel is stored as **INT8 integer weights**, together with quantization parameters that describe how to map these integers back to their original floating-point meaning.

Therefore, each layer contains:

### Core convolution parameters
- `weight_int.npy`
- `bias.npy`

### Quantization interpretation parameters
- `weight_scales.npy`
- `weight_zero_points.npy`

### Structural and metadata description
- `weight_meta.json`
- `meta.json`

---

## 3. File-by-File Description

---

### 3.1 `weight_int.npy`

#### Meaning
This file stores the **quantized integer convolution weights**.

#### Current data type
- `int8`

#### Shape
Its shape follows the convolution kernel format:

$\text{out\_channels}, \text{in\_channels}, k_h, k_w$

For example:
- `(32, 3, 3, 3)`
- `(64, 32, 3, 3)`
- `(3, 32, 1, 1)`

#### Role
This is the most important parameter file of the quantized convolution layer.

It contains the actual integer kernel values used for convolution computation in the quantized model.

#### Note
This file does **not** directly represent the original floating-point weights.  
It must be interpreted together with:
- `weight_scales.npy`
- `weight_zero_points.npy`

---

### 3.2 `weight_scales.npy`

#### Meaning
This file stores the **quantization scale(s)** for the weights.

#### Current data type
- `float32`

#### Shape
In the current model, this file has shape:
$\text{out\_channels}$

That means the current weight quantization is **per-output-channel quantization**.

#### Role
Each output channel has its own scale value.

These scales are used to map the integer weights back to their floating-point meaning.

For one quantized weight value \(q\), its approximate floating-point value \(r\) is:

$$
r \approx s \cdot (q - z)
$$
where:
- \(q\): integer weight from `weight_int.npy`
- \(s\): scale from `weight_scales.npy`
- \(z\): zero point from `weight_zero_points.npy`

#### Why float32 is acceptable
Although this file is still floating-point, it contains only a very small number of values compared with the full convolution kernel.

For example:
- a convolution layer may have tens of thousands of `weight_int` values
- but only 32 / 64 / 128 scale values

Therefore, this file is not a major storage or computation burden.

---

### 3.3 `weight_zero_points.npy`

#### Meaning
This file stores the **quantization zero point(s)** for the weights.

#### Current data type
- `int32`

#### Shape
Same as `weight_scales.npy`:
$\text{out\_channels}$

#### Role
Each output channel has one zero point.

Together with `weight_scales.npy`, this file defines how to interpret the integer weight values.

The quantization relationship is:

$$
q = \text{round}(r/s) + z
$$
and the dequantization relationship is:

$$
r \approx s \cdot (q - z)
$$
where:
- \(r\): original floating-point weight
- \(q\): quantized integer weight
- \(s\): scale
- \(z\): zero point

#### Note
In many symmetric quantization settings, zero points may often be 0, but they are still exported explicitly for completeness and correctness.

---

### 3.4 `bias.npy`

#### Meaning
This file stores the **bias term** of the convolution layer.

#### Current data type
- `float32`

#### Shape
$\text{out\_channels}$

#### Role
This is the bias added after convolution accumulation.

If the convolution output before bias is denoted as \(Y_{\text{conv}}\), then the final output before activation is:

$$
Y = Y_{\text{conv}} + b
$$
where \(b\) is the bias from `bias.npy`.

#### Why bias is still float32
Bias is usually much smaller in quantity than the full convolution kernel.

For example:
- a convolution layer may contain tens of thousands of weight values
- but only one bias per output channel

Therefore, keeping bias in `float32` does not significantly affect storage or computational complexity.

In actual hardware implementation, this bias can still be further converted to a fixed-point or integer-friendly representation if needed.

---

### 3.5 `weight_meta.json`

#### Meaning
This file describes the quantization information of the exported weight tensor.

#### Typical contents
- quantization scheme (`qscheme`)
- tensor shape
- tensor dtype
- scale file name
- zero point file name
- quantization axis
- integer representation dtype

#### Role
This file explains how to interpret:
- `weight_int.npy`
- `weight_scales.npy`
- `weight_zero_points.npy`

It acts as a metadata description for the quantized weight tensor.

---

### 3.6 `meta.json`

#### Meaning
This file describes the full convolution layer structure and output quantization metadata.

#### Typical contents
- `layer_name`
- `module_type`
- `in_channels`
- `out_channels`
- `kernel_size`
- `stride`
- `padding`
- `dilation`
- `groups`
- `padding_mode`
- `output_scale`
- `output_zero_point`
- `weight_meta_file`
- `bias_file`

#### Role
This file is the layer-level description for hardware developers.

It tells:
1. how this convolution layer is structured
2. how the output tensor of this layer is quantized
3. which files contain the weight and bias data

---

## 4. Relationship Between These Files

For each quantized convolution layer:

### 4.1 Core parameter files
These are the main parameters used in convolution:
- `weight_int.npy`
- `bias.npy`

### 4.2 Quantization interpretation files
These tell how to interpret the integer weights:
- `weight_scales.npy`
- `weight_zero_points.npy`

### 4.3 Description files
These describe the tensor format and layer structure:
- `weight_meta.json`
- `meta.json`

So the relationship can be summarized as:

$$
\text{Quantized Conv Layer}
=
\text{Integer Weight}
+
\text{Bias}
+
\text{Quantization Parameters}
+
\text{Layer Metadata}
$$

---

## 5. Quantized Weight Calculation Relationship

For the exported weight files, the floating-point weight value is approximately reconstructed as:

$$
r \approx s \cdot (q - z)
$$
where:
- \(q\): value in `weight_int.npy`
- \(s\): value in `weight_scales.npy`
- \(z\): value in `weight_zero_points.npy`
- \(r\): corresponding floating-point weight value

In the current model, weight quantization is per output channel, so:
- each output channel has one scale
- each output channel has one zero point

---

## 6. Current Exported Data Types

The current exported file types are:

| File                     | Meaning                   | Current dtype |
| ------------------------ | ------------------------- | ------------- |
| `weight_int.npy`         | quantized integer weights | `int8`        |
| `weight_scales.npy`      | weight scales             | `float32`     |
| `weight_zero_points.npy` | weight zero points        | `int32`       |
| `bias.npy`               | convolution bias          | `float32`     |
| `weight_meta.json`       | weight tensor metadata    | JSON          |
| `meta.json`              | layer metadata            | JSON          |

---

## 7. Why Not All Files Are INT8

Only the main convolution kernel is stored in `int8`.

Other files such as:
- `weight_scales.npy`
- `weight_zero_points.npy`
- `bias.npy`

are auxiliary parameters needed to interpret or use the quantized weights correctly.

This does **not** mean the quantization is ineffective.

The main reason is that:
- the overwhelming majority of parameters are in `weight_int.npy`
- scales and bias are very small in number compared with the full kernel size

Therefore, the main storage and computation burden has already been reduced significantly.

---

## 8. Practical Hardware Interpretation

From the hardware point of view:

### Main computation
- integer convolution uses `weight_int.npy`

### Auxiliary parameters
- `weight_scales.npy` and `weight_zero_points.npy` define how the quantized weights correspond to original values
- `bias.npy` provides the additive bias for each output channel

### Metadata support
- `weight_meta.json` and `meta.json` describe how to parse and use all files

If needed, hardware implementation can further convert:
- `weight_scales.npy`
- `bias.npy`

into a more hardware-friendly fixed-point representation.
