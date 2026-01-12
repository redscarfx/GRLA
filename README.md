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
- [ ] Implement Convolution only SR model 
- [ ] Implement Linear Attention
- [ ] Implement GRBF Kernel
