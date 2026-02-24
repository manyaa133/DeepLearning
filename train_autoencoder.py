import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder

# Load and normalize MNIST
data = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
X = data[:, 1:] / 255.0

# Train / Validation split
split = int(0.8 * X.shape[0])
X_train = X[:split]
X_val = X[split:]

# Initialize model
model = Autoencoder(input_size=784, hidden1=128, latent_size=32, lr=0.01)

epochs = 20
losses = []

# Training
for epoch in range(epochs):
    output = model.forward(X_train)
    loss = model.compute_loss(X_train, output, lambda_l1=0.001)
    model.backward(X_train)

    losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

# Plot loss
plt.figure()
plt.plot(losses)
plt.title("Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Reconstruction
reconstructed = model.forward(X_val[:5])

for i in range(5):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(X_val[i].reshape(28, 28), cmap='gray')
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")

    plt.show()

# Outlier detection
recon_all = model.forward(X_val)
recon_error = np.mean((X_val - recon_all) ** 2, axis=1)

threshold = np.mean(recon_error) + 2 * np.std(recon_error)

print("Outlier Threshold:", threshold)
print("Number of Outliers Detected:", np.sum(recon_error > threshold))

plt.figure()
plt.hist(recon_error, bins=30)
plt.axvline(threshold, linestyle='--')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()