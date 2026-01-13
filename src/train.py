import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from edsr import train_one_epoch
from grla import GRLABlock
from validate import validate
from data_loader import DIV2KDataset
from model import GRLASR
import yaml

if __name__ == "__main__":
    '''
    Main training loop for the GRLA Super Resolution model
    '''
    with open("src/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
        if cfg["device"] == "auto"
        else cfg["device"]
    )

    model = GRLASR(
        scale=cfg["model"]["scale"],
        dim=cfg["model"]["dim"],
        num_blocks=cfg["model"]["num_blocks"],  
    ).to(device)

    print(model)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    val_dataset = DIV2KDataset(
        root_dir=cfg["dataset"]["root_dir"],
        split="train",
        scale=cfg["dataset"]["scale"],
        patch_size=cfg["dataset"]["val_patch_size"],
        augment=False, # always false for validation
    )

    train_dataset = DIV2KDataset(
        root_dir=cfg["dataset"]["root_dir"],
        split="train",
        scale=cfg["dataset"]["scale"],
        patch_size=cfg["dataset"]["train_patch_size"],  
        augment=cfg["dataset"]["augment"],  
    )

    # Take first 100 images since I did not download the validation set yet
    val_subset = Subset(val_dataset, range(cfg["dataset"]["val_subset_size"]))
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    
    print("Training GRLA model...")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_psnr = validate(
            model, val_loader, scale=4, device=device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB"
        )