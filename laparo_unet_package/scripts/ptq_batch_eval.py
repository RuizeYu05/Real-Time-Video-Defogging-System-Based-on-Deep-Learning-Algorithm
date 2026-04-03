import copy
import os
import cv2
import csv
import torch
import torch.nn as nn
import torch.ao.quantization as tq

from model.model import LaparoUNet, ConvBlock
from utils.reconstruction import reconstruct


# -----------------------------
# image io
# -----------------------------
def load_image(path):
    img = cv2.imread(path)  # BGR
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return tensor


def save_tensor_image(tensor, path):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# -----------------------------
# quant wrapper
# -----------------------------
class QuantizedWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# -----------------------------
# fuse conv + relu
# -----------------------------
def fuse_laparo_unet(model):
    for _, module in model.named_modules():
        if isinstance(module, ConvBlock):
            tq.fuse_modules(module.conv, [['0', '1'], ['2', '3']], inplace=True)
    return model


# -----------------------------
# calibration
# -----------------------------
def get_image_list(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(exts):
            files.append(os.path.join(folder, name))
    files.sort()
    return files


def get_calib_tensors(image_paths):
    tensors = []
    for p in image_paths:
        tensors.append(load_image(p))
    return tensors


# -----------------------------
# metric
# -----------------------------
def mae(a, b):
    return torch.mean(torch.abs(a - b)).item()


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


def summarize(values, name):
    if len(values) == 0:
        print(f"{name}: empty")
        return
    vals = torch.tensor(values, dtype=torch.float32)
    print(f"{name}: mean={vals.mean().item():.8f}, "
          f"min={vals.min().item():.8f}, "
          f"max={vals.max().item():.8f}")


# -----------------------------
# main
# -----------------------------
def main():
    # ===== path config =====
    ckpt_path = r"checkpoints/laparo_unet_epoch_50.pth"

    # 需要批量评估的图片文件夹
    eval_folder = r"F:/laparoscopy_dehazing/data/ptq_calib"

    # calibration 图片数量
    calib_num = 20

    # 是否保存前几张图的可视化结果
    save_vis_num = 5

    result_dir = r"results/ptq_batch"
    os.makedirs(result_dir, exist_ok=True)

    # ===== cpu only =====
    device = torch.device("cpu")

    print("Torch version:", torch.__version__)
    print("Supported quantized engines:", torch.backends.quantized.supported_engines)

    if "onednn" in torch.backends.quantized.supported_engines:
        backend = "onednn"
    elif "x86" in torch.backends.quantized.supported_engines:
        backend = "x86"
    elif "fbgemm" in torch.backends.quantized.supported_engines:
        backend = "fbgemm"
    elif "qnnpack" in torch.backends.quantized.supported_engines:
        backend = "qnnpack"
    else:
        raise RuntimeError("No supported quantization backend found.")

    torch.backends.quantized.engine = backend
    print("Using quantization backend:", backend)

    # ===== collect image list =====
    image_paths = get_image_list(eval_folder)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {eval_folder}")

    print(f"Found {len(image_paths)} images in eval folder.")

    calib_paths = image_paths[:min(calib_num, len(image_paths))]
    print(f"Using {len(calib_paths)} images for calibration.")

    # ===== load FP32 model =====
    base_model = LaparoUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt["model_state"])
    base_model.eval()

    # ===== build quant model =====
    quant_model = copy.deepcopy(base_model)
    quant_model.eval()
    quant_model = fuse_laparo_unet(quant_model)
    quant_model = QuantizedWrapper(quant_model)
    quant_model.eval()

    quant_model.qconfig = tq.get_default_qconfig(backend)
    print("QConfig:", quant_model.qconfig)

    tq.prepare(quant_model, inplace=True)

    # ===== calibration =====
    calib_tensors = get_calib_tensors(calib_paths)
    with torch.no_grad():
        for x in calib_tensors:
            _ = quant_model(x.to(device))

    tq.convert(quant_model, inplace=True)

    print("\nQuantized model built successfully.")

    # ===== batch evaluation =====
    rows = []

    f_mae_list, f_mse_list = [], []
    l_mae_list, l_mse_list = [], []
    j_mae_list, j_mse_list = [], []

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            x = load_image(img_path).to(device)

            f_fp32 = base_model(x)
            f_int8 = quant_model(x)

            l_fp32, j_fp32 = reconstruct(x, f_fp32)
            l_int8, j_int8 = reconstruct(x, f_int8)

            f_mae_val = mae(f_fp32, f_int8)
            f_mse_val = mse(f_fp32, f_int8)

            l_mae_val = mae(l_fp32, l_int8)
            l_mse_val = mse(l_fp32, l_int8)

            j_mae_val = mae(j_fp32, j_int8)
            j_mse_val = mse(j_fp32, j_int8)

            f_mae_list.append(f_mae_val)
            f_mse_list.append(f_mse_val)
            l_mae_list.append(l_mae_val)
            l_mse_list.append(l_mse_val)
            j_mae_list.append(j_mae_val)
            j_mse_list.append(j_mse_val)

            rows.append({
                "image": os.path.basename(img_path),
                "f_mae": f_mae_val,
                "f_mse": f_mse_val,
                "l_mae": l_mae_val,
                "l_mse": l_mse_val,
                "j_mae": j_mae_val,
                "j_mse": j_mse_val,
            })

            print(f"[{idx+1:04d}/{len(image_paths):04d}] "
                  f"{os.path.basename(img_path)} | "
                  f"J_MAE={j_mae_val:.8f}, J_MSE={j_mse_val:.8f}")

            # 保存前几张图的可视化
            if idx < save_vis_num:
                vis_dir = os.path.join(result_dir, f"sample_{idx+1:03d}")
                os.makedirs(vis_dir, exist_ok=True)

                save_tensor_image(x, os.path.join(vis_dir, "input.png"))
                save_tensor_image(f_fp32, os.path.join(vis_dir, "F_fp32.png"))
                save_tensor_image(f_int8, os.path.join(vis_dir, "F_int8.png"))
                save_tensor_image(l_fp32, os.path.join(vis_dir, "L_fp32.png"))
                save_tensor_image(l_int8, os.path.join(vis_dir, "L_int8.png"))
                save_tensor_image(j_fp32, os.path.join(vis_dir, "J_fp32.png"))
                save_tensor_image(j_int8, os.path.join(vis_dir, "J_int8.png"))

                f_diff_vis = torch.clamp(torch.abs(f_fp32 - f_int8) * 10.0, 0.0, 1.0)
                l_diff_vis = torch.clamp(torch.abs(l_fp32 - l_int8) * 10.0, 0.0, 1.0)
                j_diff_vis = torch.clamp(torch.abs(j_fp32 - j_int8) * 10.0, 0.0, 1.0)

                save_tensor_image(f_diff_vis, os.path.join(vis_dir, "F_diff_x10.png"))
                save_tensor_image(l_diff_vis, os.path.join(vis_dir, "L_diff_x10.png"))
                save_tensor_image(j_diff_vis, os.path.join(vis_dir, "J_diff_x10.png"))

    # ===== save csv =====
    csv_path = os.path.join(result_dir, "ptq_batch_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "f_mae", "f_mse", "l_mae", "l_mse", "j_mae", "j_mse"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Summary =====")
    summarize(f_mae_list, "F_MAE")
    summarize(f_mse_list, "F_MSE")
    summarize(l_mae_list, "L_MAE")
    summarize(l_mse_list, "L_MSE")
    summarize(j_mae_list, "J_MAE")
    summarize(j_mse_list, "J_MSE")

    print(f"\nSaved csv to: {csv_path}")

    # 也保存量化后的 state_dict
    quant_ckpt_path = os.path.join(result_dir, "laparo_unet_ptq_state_dict.pth")
    torch.save(quant_model.state_dict(), quant_ckpt_path)
    print(f"Saved quantized state_dict to: {quant_ckpt_path}")


if __name__ == "__main__":
    main()