import os
import random
from glob import glob
from PIL import Image
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
        use_y_channel=True,
        augment=True,
    ):
        assert split in ["train", "val"] # atm we only have train but we will have val at some point
        assert scale in [2, 3, 4] # DIV2K standard scales

        self.scale = scale
        self.patch_size = patch_size
        self.use_y = use_y_channel
        self.augment = augment if split == "train" else False

        if split == "train":
            hr_dir = os.path.join(root_dir, "DIV2K_train_HR")
        else: 
            hr_dir = os.path.join(root_dir, "DIV2K_valid_HR")

        self.hr_paths = sorted(glob(os.path.join(hr_dir, "*.png"))) # all the png images in the dir
        assert len(self.hr_paths) > 0, "No HR images found." # sanity check

    def __len__(self):
        return len(self.hr_paths)

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
            hr, lr = hr.rotate(90, expand=True), lr.rotate(90, expand=True)
        return hr, lr

    def _to_y(self, img): # convert to Y channel, this is used in some evaluations, like for PSNR
        y, _, _ = img.convert("YCbCr").split()
        return TF.to_tensor(y)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_paths[idx]).convert("RGB")

        if self.patch_size is not None:
            hr = self._random_crop(hr)

        lr = hr.resize(
            (hr.width // self.scale, hr.height // self.scale),
            resample=Image.Resampling.BICUBIC,
        ) # generate LR image once fetched as a hr image

        if self.augment:
            hr, lr = self._augment(hr, lr) # only if specified we want augmentation

        if self.use_y:
            hr = self._to_y(hr)
            lr = self._to_y(lr)
        else:
            hr = TF.to_tensor(hr)
            lr = TF.to_tensor(lr)

        return { # return a dict with both images
            "lr": lr,
            "hr": hr,
        }

if __name__ == "__main__":
    dataset = DIV2KDataset(
        root_dir="src/data/DIV2K",
        split="train",
        scale=4,
        patch_size=64,
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