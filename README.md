# GRLA: BRIDGING SOFTMAX AND LINEAR ATTENTION VIA GAUSSIAN RBF KERNEL FOR LIGHTWEIGHT IMAGE SUPER-RESOLUTION

## Project Objective

The goal of this project is to understand the article, implement the proposed architecture and experiment with it. Many changes will need to be made, most likely on the dataset used but also on the experiment in the paper. The paper was found on the list of submissions of ICLR 2025. This means that the paper was not peer review yet and this project is no guarantee that the original paper is perfect.

## Article

The paper is available [here](https://openreview.net/pdf?id=qS3LTUrncS).

## Model Architecture

The model architecture is the following:

![Schematic illustration of the proposed Gaussian Radial Basis Function (GRBF)-based
Linear Attention (GRLA)](docs/arch.png)

In our implementation, we built this to be fully modular. Each file contains different blocks, or parts of the different blocks. For example, the ```GRLA``` block is defined in the ```grla.py``` and it is built using the ```TWSABlock```, the ```TSABlock``` and the ```ConvFFN``` classes (each in their respective files). 

### Architecture Blocks Overview

- **ConvResidualBlock**  
  EDSR-style convolutional residual block that refines local features using two 3×3 convolutions with a skip connection.

- **TWSABlock (Transformer Window Self-Attention)**  
  Local self-attention block that models short-range dependencies by applying multi-head self-attention within spatial windows.

- **TLA / GRBFLA (Transformer Linear Attention)**  
  Global attention block that captures long-range dependencies using a GRBF linear attention mechanism with linear complexity.

- **GRLABlock**  
  Core building block of the model that sequentially combines convolutional refinement, local window attention (TWSA), and global linear attention (TLA).

- **GRLASR Model**  
  End-to-end super-resolution network composed of a shallow convolutional head, stacked GRLA blocks for feature extraction, and a pixel-shuffle-based upsampling tail.


## Installation

Create a ```venv``` using the command ```python -m venv venv``` and install the requirements using the command ```pip install -r requirements.txt```.

### Data

To train and evaluation the data, we use the *DIV2K* dataset for training and the *Set5*, *Set14* and *BSD100* for evaluation. All 4 datasets can be downloaded from kaggle at the following links:

- [DIV2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images)
- [Set5](https://www.kaggle.com/datasets/bijaygurung/set5-superresolution)
- [Set14](https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset)
- [BSD100](https://www.kaggle.com/datasets/asilva1691/bsd100)

All four of these folders need to be unzipped and placed in a folder called ```data``` inside of the ```src/``` folder. The structure of the ```data``` folder should be the following:

```plaintext
GRLA\SRC\DATA
├───bsd100
│   ├───bicubic_2x
│   │   ├───train
│   │   │   ├───HR
│   │   │   └───LR
│   │   └───val
│   │       ├───HR
│   │       └───LR
│   └───bicubic_4x
│       ├───train
│       │   ├───HR
│       │   └───LR
│       └───val
│           ├───HR
│           └───LR
├───DIV2K
│   └───DIV2K_train_HR
├───Set14
└───Set5
    ├───GTmod12
    ├───LRbicx2
    ├───LRbicx3
    ├───LRbicx4
    └───original
```
> Note: It is important for the data folder to have this structure for the code to find the data and run properly!

### Running the code

#### Training

To train a new model, modify the ```config.yaml``` and remove any paths from the ```resume_checkpoint``` variable (which causes the training to resume from a pre-existing model).
The training hyperparameters can all be changed from the config and the training script should work out of the box.

To run a training just run ```python src/train.py```, the training can be monitored directly in the terminal but also through the tensorboard. During training, the model is evaluated
every n epochs only on the *Set5* dataset

#### Validation

Models can be evaluated using the ```validation.py``` script. Just create/change the variable ```ckpt_path``` to evaluate the proper model (used in ```checkpoint = torch.load(ckpt_path, map_location=device)``` to load the model).

The model is then evaluated on *Set5*, *Set14* and *BSD100*. It is evaluated on two metrics (PSNR and SSIM) and is compared to a bicubic basline to compare performance.

### Progress List

- [x] Implement DataLoader
- [x] Implement Bicubic Baseline
- [x] Implement PSNR score calculator
- [X] Implement Convolution only SR model -> UPDATE: with a lite model (much smaller and less blocks than state-of-the-art) we get slightly better results and the bicubic baseline 
- [X] Implement TWSA - > UPDATE: implemented this and seems to work fine (combined with EDSR layer) it gets results identical to baseline, and even a bit higher with more training time (not bad considering TLA is not added yet)
- [x] add conv layers before MHSA in TWSA
- [x] in TWSA use Batch Norm instead of Layer Norm?
- [X] Keep layer norm in TWSA/TLA or remove? (experiment with it) -> made it optional!
- [X] Implement Transformer Linear Attention with GRL kernel
- [X] Implement Code skeleton for main GRLA model (missing modules replaced with ```nn.Identity()```)
- [X] Log model info (trainable params, architecture per training, hyperparameters etc...)
- [X] Add proper logging of training
- [X] Download the Validation sets
- [x] Train for > 100 epochs (change lr if resuming training)
- [x] Added SSIM score

After this is done, test the model with different depths and hyperparameters (and might have to change the quantity of data if speed takes to long)

Model does seem to steadily improve over time! We hit a score of > 29 dB at about 80 epochs. No doubt it will keep increasing if we look at the score and loss curves.
Very expensive in terms of training time but baseline has been beat at the moment. 

### Logging

Everything is being tracked using tensorboard. All the runs are saved in the ```logs``` directory. Here is what we are currently being logged (will probably be updated in the future):

- **Images**
- **Loss**
- **PSNR**
- **Learning rate**
- **Batch time**
- **Gradient norm**
- **GPU memory**
- **Config**

### Future Improvements

- Study impact of hyperparameters
- Study more the behavior on x2 and x3 resolution (this project focused mostly of x4)
- Try and get exact match on parameters count (would need more specifications on the paper's implementation)
- Add the other validation sets in the training script (currently only includes the Set5 validation)

### Notes

~~I think there might be an issue with the model because of the param count. When we initialize everything the way they mention in the paper, with 6 MPB blocks, each block has the proper architecture and hyperparams we get about 1.5M parameters. But in the paper they say getting about 800k~900k. Not sure where this increase of params comes from but it could be the reason why training won't be optimal. Also trianing is taking place on a RTX 3070 at the moment.~~ Nvm i fixed it was just a problem in the architecture, i fixed it now i think (get 10k more params but that's like a 1% difference so who cares).

Also they say they trained for 1000 epochs, if we replicate that it will take forever because with a model of this size one epoch takes like 3min (6 MPB blocks). Need to find a workaround for that. Either reduce model complexity or lower training time, but we need to find a work around.
Also possible vanishing gradient problem with too many blocks. Looks like sometimes the loss because NaN. It's surprised considering the amount of residual connections in the network. Need to look into that.