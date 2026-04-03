import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# Basic Conv + ReLU Block
# ============================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ============================
# UNet-based Model for F prediction
# ============================
class LaparoUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder -----
        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)

        # ----- Bottleneck (optional SE can be added later) -----
        self.bottleneck = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # ----- Decoder -----
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(128 + 64, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(64 + 32, 32)

        # ----- Output Layer -----
        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.activation = nn.Sigmoid()  # ensure F in [0,1]

    def forward(self, x):
        # ----- Encoder path -----
        x1 = self.enc1(x)   # [B,32,H,W]
        p1 = self.pool1(x1) # [B,32,H/2,W/2]

        x2 = self.enc2(p1)  # [B,64,H/2,W/2]
        p2 = self.pool2(x2) # [B,64,H/4,W/4]

        x3 = self.enc3(p2)  # [B,128,H/4,W/4]

        # ----- Bottleneck -----
        b = self.bottleneck(x3)

        # ----- Decoder path -----
        up1 = self.up1(b)   # [B,128,H/2,W/2]
        cat1 = torch.cat([up1, x2], dim=1)  # skip connection
        d1 = self.dec1(cat1)

        up2 = self.up2(d1)  # [B,64,H,W]
        cat2 = torch.cat([up2, x1], dim=1)
        d2 = self.dec2(cat2)

        # ----- Output F -----
        F_pred = self.out_conv(d2)
        F_pred = self.activation(F_pred)

        return F_pred


if __name__ == "__main__":
    # Sanity check
    model = LaparoUNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Input:", x.shape)
    print("Output F:", y.shape)
