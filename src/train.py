import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from validate import validate
from data_loader import DIV2KDataset
from model import GRLASR
import yaml
from time import time
from tqdm import tqdm

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

    print(f"Using device: {device}")

    model = GRLASR(
        scale=cfg["model"]["scale"],
        dim=cfg["model"]["dim"],
        num_blocks=cfg["model"]["num_blocks"],  
        window_size=cfg["model"]["window_size"],
        num_heads=cfg["model"]["num_heads"],
    ).to(device)

    print(model)

    # print total amount of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg["training"]["learning_rate"], 
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]), 
        eps=cfg["training"]["epsilon"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["lr_decay_step"],  
        gamma=cfg["training"]["lr_decay_gamma"],      
    )

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
        batch_size=1, # wanted to increase this for speed but cannot because images vary in size?? SUper annoying
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    
    scaler = torch.amp.GradScaler()

    print("Training GRLA model...")
    model.train()
    start_time = time()
    # use pbar for logging
    val_psnr = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        total_loss = 0.0

        # Batch-level progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}", leave=False)
        batch_start = time()
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device):
                sr = model(lr)
                loss = criterion(sr, hr)

            # Add these 3 lines back:
            scaler.scale(loss).backward()  
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Update tqdm with current loss + batch time
            batch_elapsed = time() - batch_start
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "batch_time": f"{batch_elapsed:.2f}s"
            })
            batch_start = time()  # reset for next batch

        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Validation
        if epoch % cfg["training"]["val_interval"] == 0 or epoch == 1:
            val_psnr = validate(
                model, val_loader, scale=cfg["dataset"]["scale"], device=device
            )

        # Scheduler step
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_time = time() - start_time

        # Epoch-level print
        print(
            f"Epoch {epoch:03d} | "
            f"LR: {current_lr:.2e} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"Elapsed Time: {elapsed_time:.2f} sec"
        )