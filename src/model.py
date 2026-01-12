import torch
import torch.nn as nn
from edsr import train_one_epoch
from grla import GRLABlock
from validate import validate
from data_loader import DIV2KDataset
from torch.utils.data import DataLoader, Subset

class GRLASR(nn.Module):
    def __init__(
        self,
        scale=4,
        dim=64,
        num_blocks=6,
    ):
        super().__init__()

        self.head = nn.Conv2d(1, dim, 3, padding=1)

        self.body = nn.Sequential(
            *[GRLABlock(dim) for _ in range(num_blocks)]
        )

        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, 1, 3, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRLASR(
        scale=4,
        dim=64,
        num_blocks=2,  
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
    
    print("Training GRLA model...")

    for epoch in range(1, 30):
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