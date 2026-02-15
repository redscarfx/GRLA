import torch
import torch.nn as nn
import yaml
from metrics import calc_psnr, calc_ssim
from torch.utils.data import DataLoader, Subset
from data_loader import DIV2KDataset
from bicubic import BicubicBaseline

@torch.no_grad()
def validate(model, dataloader, scale, device):
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for batch in dataloader:
        # get the high res and low res images
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        sr = model(lr) # get the super resolved image from the model
        sr = nn.functional.interpolate(sr, size=(hr.shape[2], hr.shape[3]), mode='bilinear', align_corners=False)
        psnr = calc_psnr(sr, hr, scale)
        ssim = calc_ssim(sr, hr, scale)

        # track batch values
        total_psnr += psnr
        total_ssim +=ssim
        count += 1

    return total_psnr / count, total_ssim/count # return our score 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_names = ["Set5", "Set14", "BSD100"]
    def run_eval(dataset_name):
        with open("src/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        val_dataset = DIV2KDataset(
            root_dir=cfg["dataset"]["root_dir"],
            split=dataset_name,
            scale=cfg["dataset"]["scale"],
            patch_size=None,
            patches_per_image=1,
            augment=False, # always false for validation
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
        )
        
        ckpt_path = cfg["training"]["resume_checkpoint"]
        print(f"Loading model from {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        from model import GRLASR
        model = GRLASR(
            scale=cfg["model"]["scale"],
            dim=cfg["model"]["dim"],
            num_blocks=cfg["model"]["num_blocks"],  
            window_size=cfg["model"]["window_size"],
            num_heads=cfg["model"]["num_heads"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state"])


        # non parameterized baseline model
        print("Validating Bicubic Baseline model...")
        baseline_model = BicubicBaseline(scale=4).to(device)
        psnr, ssim = validate(baseline_model, val_loader, scale=4, device=device)
        print(f"BASELINE Bicubic PSNR: {psnr:.2f} dB")
        print(f"BASELINE Bicubic SSIM: {ssim:.4f}\n")

        print("Validating GRLASR model...")
        psnr, ssim = validate(model, val_loader, scale=4, device=device)
        print(f"GRLASR PSNR: {psnr:.2f} dB")
        print(f"GRLASR SSIM: {ssim:.4f}\n")
    
    for dataset_name in dataset_names:
        print(f"\nEVALUATING ON {dataset_name}...")
        run_eval(dataset_name)