import torch
import torch.nn as nn
from metrics import calc_psnr
from torch.utils.data import DataLoader, Subset
from data_loader import DIV2KDataset
from bicubic import BicubicBaseline


@torch.no_grad()
def validate(model, dataloader, scale, device):
    model.eval()

    total_psnr = 0.0
    count = 0

    for batch in dataloader:
        # get the high res and low res images
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        sr = model(lr) # get the super resolved image from the model
        psnr = calc_psnr(sr, hr, scale)

        # track batch values
        total_psnr += psnr
        count += 1

    return total_psnr / count # return our score 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    val_dataset = DIV2KDataset(
        root_dir="src/data/DIV2K",
        split="train",
        scale=4,
        patch_size=None,
        augment=False,
    )

    train_dataset = DIV2KDataset(
        root_dir="src/data/DIV2K",
        split="train",
        scale=4,
        patch_size=64,  
        augment=True,  
    )

    # Take first 100 images since I did not download the validation set yet
    val_subset = Subset(val_dataset, range(100))

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # non parameterized baseline model
    print("Validating Bicubic Baseline model...")
    model = BicubicBaseline(scale=4).to(device)
    psnr = validate(model, val_loader, scale=4, device=device)
    print(f"Bicubic PSNR: {psnr:.2f} dB\n")