import torch
import torchvision.transforms as T
import cv2
import os

from model.model import LaparoUNet
from utils.reconstruction import reconstruct


def load_image(path):
    img = cv2.imread(path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return tensor


def save_tensor_image(tensor, path):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def main():
    input_path = "F:/laparoscopy_dehazing/data/train/000572.png"   # ✅ 修改为你的测试图路径
    ckpt_path = "checkpoints/laparo_unet_epoch_50.pth"  # ✅ 修改为你的模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    model = LaparoUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Load image ----
    I = load_image(input_path).to(device)

    # ---- Predict F ----
    with torch.no_grad():
        F = model(I)

    # ---- Reconstruct L and J ----
    from utils.reconstruction import reconstruct
    L, J = reconstruct(I, F)

    # ---- Save outputs ----
    os.makedirs("results", exist_ok=True)

    save_tensor_image(I, "results/572/input.png")
    save_tensor_image(F, "results/572/F.png")
    save_tensor_image(L, "results/572/L.png")
    save_tensor_image(J, "results/572/J.png")

    print("✅ Saved results to /results:")
    print(" - input.png")
    print(" - F.png")
    print(" - L.png")
    print(" - J.png")


if __name__ == "__main__":
    main()
