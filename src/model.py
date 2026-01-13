import torch
import torch.nn as nn
from edsr import train_one_epoch
from grla import GRLABlock
from validate import validate
from data_loader import DIV2KDataset
from torch.utils.data import DataLoader, Subset

class GRLASR(nn.Module):
    '''
    This is our main Super Resolution model using GRLA blocks
    The head is fixed to a basic conv 3x3 layer,
    followed by the body of the model which is a sequence of GRLA blocks (defined in grla.py)
    the tail of the model upsamples and reconstructs the high resolution image
    '''
    def __init__(
        self,
        scale=4,
        dim=64,
        num_blocks=6,
        window_size=8,
        num_heads=4,
        include_layer_norm=False,
    ):
        super().__init__()

        self.head = nn.Conv2d(1, dim, 3, padding=1) # this wont change, basic conv layer

        self.body = nn.Sequential(
            *[GRLABlock(dim, window_size=window_size, num_heads=num_heads, include_layer_norm=include_layer_norm) for _ in range(num_blocks)] # stacked GRLA blocks
        )

        # the tail also wont change, basic upsampling and reconstruction
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

    train_dataset = DIV2KDataset(
        root_dir="src/data/DIV2K",
        split="train",
        scale=4,
        patch_size=64,  
        augment=True,  
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    print("Sanity check for GRLA model...")

    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    print("Train step completed. Model forward and backward pass working.")
    print(f"Loss: {train_loss:.4f} ")