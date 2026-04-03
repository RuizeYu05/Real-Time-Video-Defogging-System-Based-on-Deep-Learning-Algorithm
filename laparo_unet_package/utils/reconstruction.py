import torch


def compute_alpha(F):
    """
    Computes the global alpha value used in the paper.
    Alpha = mean over all pixels and all channels of F.

    Args:
        F: predicted smoke veil tensor [B, 3, H, W]
    Returns:
        alpha: scalar tensor, shape []
    """
    return torch.mean(F)


def compute_L(I, F, alpha):
    """
    Computes L = I - alpha * F
    As defined in the paper.

    Args:
        I: input foggy image tensor [B, 3, H, W]
        F: predicted smoke veil tensor [B, 3, H, W]
        alpha: scalar tensor
    Returns:
        L: intermediate luminance-corrected image [B, 3, H, W]
    """
    return I - alpha * F


def linear_stretch(L, eps=1e-8):
    """
    Performs the linear range normalization described in the paper:

        J = (L - L_min) / (L_max - L_min)

    Applied per-batch, per-image, across all channels.

    Args:
        L: tensor [B, 3, H, W]
    Returns:
        J: tensor [B, 3, H, W], in [0,1]
    """
    B = L.shape[0]
    L_flat = L.reshape(B, -1)  # flatten each sample for min/max

    L_min = torch.min(L_flat, dim=1)[0].view(B, 1, 1, 1)
    L_max = torch.max(L_flat, dim=1)[0].view(B, 1, 1, 1)

    J = (L - L_min) / (L_max - L_min + eps)
    return torch.clamp(J, 0.0, 1.0)


def reconstruct(I, F):
    """
    Full reconstruction pipeline:
        1) compute alpha
        2) compute L = I - alpha * F
        3) apply linear stretch to get J

    Args:
        I: input foggy image tensor [B, 3, H, W]
        F: predicted smoke veil [B, 3, H, W]

    Returns:
        L: intermediate image
        J: final dehazed image in [0,1]
    """
    alpha = compute_alpha(F)
    L = compute_L(I, F, alpha)
    J = linear_stretch(L)
    return L, J


if __name__ == "__main__":
    # quick sanity check
    I = torch.rand(1, 3, 192, 384)
    F = torch.rand(1, 3, 192, 384)

    L, J = reconstruct(I, F)
    print("L range:", float(L.min()), float(L.max()))
    print("J range:", float(J.min()), float(J.max()))