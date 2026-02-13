import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from validate import validate
from data_loader import DIV2KDataset
from model import GRLASR
import yaml
from time import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
        split="val",
        scale=cfg["dataset"]["scale"],
        patch_size=None,
        patches_per_image=1,
        augment=False, # always false for validation
    )

    print("Preparing training data... Might take a minute (putting data into RAM)...")
    train_dataset = DIV2KDataset(
        root_dir=cfg["dataset"]["root_dir"],
        split="train",
        scale=cfg["dataset"]["scale"],
        patch_size=cfg["dataset"]["train_patch_size"],  
        patches_per_image=cfg["dataset"]["patches_per_image"],
        augment=cfg["dataset"]["augment"],  
    )


    val_loader = DataLoader(
        val_dataset,
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
    
    scaler = torch.amp.GradScaler()

    writer = SummaryWriter(log_dir=cfg["training"]["log_dir"])

    start_epoch = 1
    # optionally resume from a checkpoint
    if cfg["training"]["resume_checkpoint"] is not None:
        ckpt_path = cfg["training"]["resume_checkpoint"]
        print(f"Loading model and resuming from {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        if scheduler and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        if scaler and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Continued training at epoch {start_epoch}")

    print(model)

    # print total amount of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # config to not lose track of hyperparameters for each run
    writer.add_text('Config', str(cfg), 0)

    # log total trainable params
    writer.add_scalar('Params/total_trainable', total_params, 0)

    #log architecture graph
    # writer.add_graph(model, torch.randn(1, 1, 64, 64).to(device))

    # see dataset size
    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")

    print("Training GRLA model...")
    model.train()
    start_time = time()

    val_psnr = 0.0
    val_ssim = 0.0
    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        total_loss = 0.0

        # Batch-level progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}", leave=False)
        batch_start = time()
        batch_times = [] # track batch times for logging
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device):
                sr = model(lr) # should we clip this? => torch.clamp(model(lr), 0.0, 1.0) We can test later
                loss = criterion(sr, hr)

            # speed up with scaler
            scaler.scale(loss).backward()  
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            """ total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            writer.add_scalar("Gradients/grad_norm", total_norm, epoch) """
            # idk which is better, the above or the below? 
            writer.add_scalar('Gradients/grad_norm', scaler.get_scale(), epoch)

            # Update tqdm with current loss + batch time
            batch_elapsed = time() - batch_start
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "batch_time": f"{batch_elapsed:.2f}s"
            })
            batch_times.append(batch_elapsed)
            batch_start = time()  # reset for next batch

        # log average batch time per epoch
        writer.add_scalar('Time/avg_batch_time', sum(batch_times) / len(batch_times), epoch)

        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch) # log it

        # Validation
        if epoch % cfg["training"]["val_interval"] == 0 or epoch == 1:
            val_psnr = validate(
                model, val_loader, scale=cfg["dataset"]["scale"], device=device
            )
            writer.add_scalar('PSNR/val', val_psnr, epoch)

        if epoch % cfg["training"]["log_image_interval"] == 0:
            # log 4 images, we take the first 4 from the last batch (every 10 epochs)
            lr_grid = torchvision.utils.make_grid(lr[:4])
            sr_grid = torchvision.utils.make_grid(sr[:4])
            hr_grid = torchvision.utils.make_grid(hr[:4])
            writer.add_image('Images/LR', lr_grid, epoch)
            writer.add_image('Images/SR', sr_grid, epoch)
            writer.add_image('Images/HR', hr_grid, epoch)

        # save model every n epochs
        if epoch % cfg["training"]["save_interval"] == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "scaler_state": scaler.state_dict() if scaler else None,
            }

            torch.save(checkpoint, "models/checkpoint.pth")

        # Scheduler step
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar('Learning_Rate', current_lr, epoch) # track lr to see how it evolves
        elapsed_time = time() - start_time

        # GPU memory usage logging
        writer.add_scalar('Memory/gpu_mem', torch.cuda.max_memory_allocated() / 1e9, epoch)

        # Epoch-level print
        print(
            f"Epoch {epoch:03d} | "
            f"LR: {current_lr:.2e} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"Elapsed Time: {elapsed_time:.2f} sec"
        )