import numpy as np
import matplotlib.pyplot as plt
from rbm import RBM
import pandas as pd

# Load MNIST dataset
data = pd.read_csv("mnist_train.csv")

# Remove label column and normalize pixel values
X = data.iloc[:, 1:].values
X = X / 255.0

# Define RBM parameters
n_visible = X.shape[1]
n_hidden = 64

# Initialize RBM
rbm = RBM(n_visible, n_hidden, learning_rate=0.01)

epochs = 30
errors = []

# Train RBM
for epoch in range(epochs):
    error = rbm.train(X)
    errors.append(error)
    print(f"Epoch {epoch+1}/{epochs}, Error: {error:.6f}")

# Plot reconstruction error
plt.figure()
plt.plot(errors)
plt.title("RBM Training Error")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Error")
plt.show()

# Visualize first 16 learned filters
plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(rbm.W[:, i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.suptitle("RBM Learned Filters")
plt.show()