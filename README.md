# Deep Learning Project – Autoencoder (NumPy Implementation)

## Overview
This project implements a fully connected Autoencoder from scratch using NumPy.  
The model is trained on the MNIST dataset to reconstruct handwritten digit images and perform basic outlier detection using reconstruction error.



## Project Files

DeepLearning_Project/
- autoencoder.py
- train_autoencoder.py
- mnist_train.csv
- README.md



## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install required libraries:

pip install numpy matplotlib



## How to Run

1. Place `mnist_train.csv` inside the project folder.
2. Open terminal inside the project directory.
3. Run the training script:

python train_autoencoder.py



## What the Program Does

- Loads MNIST dataset
- Normalizes pixel values to range [0,1]
- Splits data into training and validation sets
- Trains an Autoencoder model
- Plots training loss curve
- Displays original vs reconstructed images
- Performs outlier detection using reconstruction error
- Plots reconstruction error distribution



## Model Architecture

- Input Layer: 784 neurons
- Hidden Layer: 128 neurons
- Latent Layer: 32 neurons
- Activation Functions:
  - ReLU (hidden layers)
  - Sigmoid (output layer)
- Loss Function:
  - Mean Squared Error (MSE)
  - L1 sparsity penalty on latent representation



## Output

- Training loss graph
- Image reconstruction comparison
- Reconstruction error histogram
- Number of detected outliers


---
Author: Manya
---
Course: Deep Learning
---
