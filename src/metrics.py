import torch
import torch.nn.functional as F
import math

def calc_psnr(sr, hr, scale):
    """
    using two images, one super resolved and one high res (ground truth), we
    calculate the PSNR (Peak Signal to Noise Ratio) between them.
    High PSNR => better quality
    0 dB = worst quality, infinite dB = perfect quality
    If we call this function on image that are identical, it will return infinite (since they are the same)
    """

    # assert shapes are compatible
    assert sr.shape == hr.shape, "SR and HR shapes are different"

    # mandatory crop of borders to ignre border effects
    if scale > 0:
        sr = sr[:,:, scale:-scale, scale:-scale] # ignore H and W borders
        hr = hr[:,:, scale:-scale, scale:-scale] # and we remove the border pixels too

    mse = torch.mean((sr - hr) ** 2) 
    if mse == 0:
        return float("inf")

    psnr = 10 * torch.log10(1.0 / mse) # psnr calculation 
    return psnr.item()

def calc_ssim(sr, hr, scale, data_range=1.0, window_size=11, sigma=1.5):
    """
    Compute SSIM (Structural Similarity Index) between SR and HR images.

    Args:
        sr (Tensor): super-resolved image, shape (N, C, H, W)
        hr (Tensor): high-resolution ground truth, same shape
        scale (int): border crop size (same as PSNR)
        data_range (float): max pixel value (1.0 if normalized, 255 if uint8)
        window_size (int): Gaussian window size
        sigma (float): Gaussian std

    Returns:
        float: mean SSIM over batch
    """

    assert sr.shape == hr.shape, "SR and HR shapes are different"
    assert sr.dim() == 4, "Expected shape (N, C, H, W)"

    if scale > 0:
        sr = sr[:, :, scale:-scale, scale:-scale]
        hr = hr[:, :, scale:-scale, scale:-scale]

    device = sr.device
    channel = sr.size(1)

    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(0)
    window_2d = window_1d.T @ window_1d
    window = window_2d.expand(channel, 1, window_size, window_size)

    mu_sr = F.conv2d(sr, window, padding=window_size // 2, groups=channel)
    mu_hr = F.conv2d(hr, window, padding=window_size // 2, groups=channel)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr
    sigma_sr_sq = F.conv2d(sr * sr, window, padding=window_size // 2, groups=channel) - mu_sr_sq
    sigma_hr_sq = F.conv2d(hr * hr, window, padding=window_size // 2, groups=channel) - mu_hr_sq
    sigma_sr_hr = F.conv2d(sr * hr, window, padding=window_size // 2, groups=channel) - mu_sr_hr

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)) / \
               ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))

    return ssim_map.mean().item()


if __name__ == "__main__":
    # example with identical images, which should give infinite PSNR
    x = torch.rand(1, 256, 256)
    psnr = calc_psnr(sr=x, hr=x, scale=4)
    print(psnr)  # should be inf
    ssim = calc_ssim(sr=x, hr=x, scale=4)
    print(ssim)  # should be 1

