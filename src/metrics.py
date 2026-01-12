import torch
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

if __name__ == "__main__":
    # example with identical images, which should give infinite PSNR
    x = torch.rand(1, 256, 256)
    psnr = calc_psnr(sr=x, hr=x, scale=4)
    print(psnr)  # should be inf
