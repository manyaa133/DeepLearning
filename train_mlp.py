import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

# Load MNIST dataset (CSV)
data = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)

# Separate features and labels
X = data[:, 1:]
y = data[:, 0]

# Normalize pixel values
X = X / 255.0

# One-hot encoding function
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
    return one_hot

y_onehot = one_hot_encode(y)

# Train-validation split (80-20)
split = int(0.8 * X.shape[0])
X_train, X_val = X[:split], X[split:]
y_train, y_val = y_onehot[:split], y_onehot[split:]

# Optional: visualize one sample
plt.imshow(X_train[0].reshape(28, 28), cmap="gray")
plt.title("Label: " + str(np.argmax(y_train[0])))
plt.show()

# MLP configuration
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.01
epochs = 30

model = MLP(input_size, hidden_size, output_size, learning_rate)

losses = []
train_accs = []
val_accs = []

# Training loop
for epoch in range(epochs):
    Y_pred = model.forward(X_train)
    loss = model.compute_loss(Y_pred, y_train)
    model.backward(X_train, y_train)

    train_acc = model.accuracy(Y_pred, y_train)
    val_pred = model.forward(X_val)
    val_acc = model.accuracy(val_pred, y_val)

    losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f}")

# Plot training loss
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot training and validation accuracy
plt.figure()
plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()