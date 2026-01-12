# GRLA: BRIDGING SOFTMAX AND LINEAR ATTENTION VIA GAUSSIAN RBF KERNEL FOR LIGHTWEIGHT IMAGE SUPER-RESOLUTION

## Project Objective

The goal of this project is to understand the article, implement the proposed architecture and experiment with it. Many changes will need to be made, most likely on the dataset used but also on the experiment in the paper. The paper was found on the list of submissions of ICLR 2025. This means that the paper was not peer review yet and this project is no guarantee that the original paper is perfect.

## Article

The paper is available [here](https://openreview.net/pdf?id=qS3LTUrncS).

## Installation

Create a ```venv``` using the command ```python -m venv venv``` and install the requirements using the command ```pip install -r requirements.txt```.

### Data

First the data needs to be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip).
Unzip it and put in the the data folder in the ```/src``` folder (the data folder will get ignored by the .gitignore).

### Progress List

- [x] Implement DataLoader
- [x] Implement Bicubic Baseline
- [x] Implement PSNR score calculator
- [X] Implement Convolution only SR model -> UPDATE: with a lite model (much smaller and less blocks than state-of-the-art) we get slightly better results and the bicubic baseline 
- [X] Implement TWSA - > UPDATE: implemented this and seems to work fine (combined with EDSR layer) it gets results identical to baseline, and even a bit higher with more training time (not bad considering TLA is not added yet)
- [ ] Implement Transformer Linear Attention with GRL kernel
- [X] Implement Code skeleton for main GRLA model (missing modules replaced with ```nn.Identity()```)
- [ ] Download the Validation Set

After this is done, test the model with different depths and hyperparameters (and might have to change the quantity of data if speed takes to long)
