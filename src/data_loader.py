import os
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class DIV2KDataset(Dataset):
    """
    DIV2K dataset compatible with pytorch's data loader,
    Uses high-resolution images to generate low-resolution counterparts on the go
    the y channel is for super res in color space YCbCr
    """

    def __init__(
        self,
        root_dir,
        split="train",
        scale=4,
        patch_size=64,
        patches_per_image=16,
        use_y_channel=True,
        augment=True,
    ):
        assert split in ["train", "val"] # atm we only have train but we will have val at some point
        assert scale in [2, 3, 4] # DIV2K standard scales

        self.scale = scale
        self.patch_size = patch_size
        self.use_y = use_y_channel
        self.augment = augment if split == "train" else False
        self.patches_per_image = patches_per_image if split == "train" else 1

        if split == "train":
            hr_dir = os.path.join(root_dir, "DIV2K/DIV2K_train_HR")
        else: 
            hr_dir = os.path.join(root_dir, "Set5/original")

        self.hr_paths = sorted(glob(os.path.join(hr_dir, "*.png"))) # all the png images in the dir
        assert len(self.hr_paths) > 0, "No HR images found." # sanity check

        self.hr_images = [
            TF.to_tensor(Image.open(p).convert("RGB"))
            for p in self.hr_paths
        ] # preload all images in memory for speed

    def __len__(self):
        return len(self.hr_paths) * self.patches_per_image

    def _random_crop(self, img): # standard SR random crop
        hr_crop = self.patch_size * self.scale
        w, h = img.size
        x = random.randint(0, w - hr_crop)
        y = random.randint(0, h - hr_crop)
        return img.crop((x, y, x + hr_crop, y + hr_crop))

    def _augment(self, hr, lr): # data augmentation, on both hr and lr images to increase diversity
        if random.random() < 0.5:
            hr, lr = TF.hflip(hr), TF.hflip(lr)
        if random.random() < 0.5:
            hr, lr = TF.vflip(hr), TF.vflip(lr)
        if random.random() < 0.5:
            hr = hr.transpose(Image.ROTATE_90)
            lr = lr.transpose(Image.ROTATE_90)
        return hr, lr

    def _to_y(self, img): # convert to Y channel, this is used in some evaluations, like for PSNR
        y, _, _ = img.convert("YCbCr").split()
        return TF.to_tensor(y)

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image

        # HR tensor: (3, H, W), range [0, 1]
        hr = self.hr_images[img_idx].clone()

        _, H, W = hr.shape

        # --- Random crop (TRAIN ONLY) ---
        if self.patch_size is not None:
            hr_crop = self.patch_size * self.scale
            x = random.randint(0, W - hr_crop)
            y = random.randint(0, H - hr_crop)
            hr = hr[:, y:y+hr_crop, x:x+hr_crop]

        # --- Generate LR via bicubic downsampling ---
        lr = torch.nn.functional.interpolate(
            hr.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        # --- Data augmentation (TRAIN ONLY) ---
        if self.augment:
            if random.random() < 0.5:
                hr = torch.flip(hr, dims=[2])
                lr = torch.flip(lr, dims=[2])
            if random.random() < 0.5:
                hr = torch.flip(hr, dims=[1])
                lr = torch.flip(lr, dims=[1])
            if random.random() < 0.5:
                hr = hr.transpose(1, 2)
                lr = lr.transpose(1, 2)

        # --- Convert to Y channel (ONCE) ---
        if self.use_y:
            hr = 0.257 * hr[0:1] + 0.504 * hr[1:2] + 0.098 * hr[2:3] + 16/255
            lr = 0.257 * lr[0:1] + 0.504 * lr[1:2] + 0.098 * lr[2:3] + 16/255

        return {
            "lr": lr,
            "hr": hr,
        }

if __name__ == "__main__":
    dataset = DIV2KDataset(
        root_dir="src/data/DIV2K",
        split="train",
        scale=4,
        patch_size=64,
        patches_per_image=16,
    )

    sample = dataset[random.randint(0, len(dataset) - 1)] # get a random sample (this contains both lr and hr images)
    print(len(dataset))  # Number of images in the dataset
    # should be an image 4 times smaller in h x w
    print(sample["lr"].shape)  # (1, 64, 64)
    print(sample["hr"].shape)  # (1, 256, 256)
    # visualize the images
    lr_img = TF.to_pil_image(sample["lr"])
    hr_img = TF.to_pil_image(sample["hr"])
    plt.subplot(1, 2, 1)
    plt.title("Low-Resolution Image")
    plt.imshow(lr_img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("High-Resolution Image")
    plt.imshow(hr_img, cmap='gray')
    plt.axis('off')
    plt.show()