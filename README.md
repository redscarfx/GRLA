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

First the data needs to be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip).
Unzip it and put in the the data folder in the ```/src``` folder (the data folder will get ignored by the .gitignore).
Same goes for the validation data which can be found [here](https://www.kaggle.com/datasets/bijaygurung/set5-superresolution?resource=download). We use the 'original' folder for validation but download the entire folder, and put the Set5 folder in data (remove from the archive folder it is stored in).

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
- [X] Download the Validation Set
- [ ] Train for > 100 epochs (change lr if resuming training)

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

**NOTES:**

~~I think there might be an issue with the model because of the param count. When we initialize everything the way they mention in the paper, with 6 MPB blocks, each block has the proper architecture and hyperparams we get about 1.5M parameters. But in the paper they say getting about 800k~900k. Not sure where this increase of params comes from but it could be the reason why training won't be optimal. Also trianing is taking place on a RTX 3070 at the moment.~~ Nvm i fixed it was just a problem in the architecture, i fixed it now i think (get 10k more params but that's like a 1% difference so who cares).

Also they say they trained for 1000 epochs, if we replicate that it will take forever because with a model of this size one epoch takes like 3min (6 MPB blocks). Need to find a workaround for that. Either reduce model complexity or lower training time, but we need to find a work around.
Also possible vanishing gradient problem with too many blocks. Looks like sometimes the loss because NaN. It's surprised considering the amount of residual connections in the network. Need to look into that.