import torch.nn as nn
import torch
from data_loader import DIV2KDataset
from torch.utils.data import DataLoader, Subset
from validate import validate

'''
Before implementing the more complex GLBF model, we will implement a fully convolutional model
to see what results we can get with this to begin with
This will serve as an improvement (hopefully) to the bicubic baseline (bicubic.py)
'''
class ResidualBlock(nn.Module):
    '''
    Standard residual block with two conv layers and a skip connection
    '''
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res

class EDSR(nn.Module):
    ''' The default values are picked to be lighter than the original EDSR for faster training '''
    def __init__(self, scale=4, num_blocks=8, channels=64):
        super().__init__()

        self.head = nn.Conv2d(1, channels, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # now lets test the EDSR Lite model which is fully convoltional and should do better than bicubic
    # original EDSR uses 32 blocks and 256 channels, but this is too big for quick experiments
    # we halved the model capacity for faster training
    # just a test to see if we can reach close to 30 dB PSNR on DIV2K validation set
    model = EDSR(scale=4, num_blocks=16, channels=128).to(device)
    print(model)

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
    
    print("Training EDSR Lite model...")

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

'''
I managed to get a dB score on validation of about 27.5 dB after 30 epochs of training with this lighter EDSR model
In the original paper they report about 32.6 dB with the full EDSR model (32 blocks, 256 channels)
This improves only by a little bit from the baseline, but if i were to beef up the model im sure that it would reach the 30db score mark
Also in the original training they do about 200 (if not more) epochs of training so its definetely possible to reach higher scores with more training
Anyways this works for now!
'''