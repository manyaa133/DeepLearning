import numpy as np

class Autoencoder:
    def __init__(self, input_size=784, hidden1=128, latent_size=32, lr=0.01):
        self.lr = lr
        
        # Encoder
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden1))
        
        self.W2 = np.random.randn(hidden1, latent_size) * np.sqrt(2. / hidden1)
        self.b2 = np.zeros((1, latent_size))
        
        # Decoder
        self.W3 = np.random.randn(latent_size, hidden1) * np.sqrt(2. / latent_size)
        self.b3 = np.zeros((1, hidden1))
        
        self.W4 = np.random.randn(hidden1, input_size) * np.sqrt(2. / hidden1)
        self.b4 = np.zeros((1, input_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.latent = self.relu(self.Z2)

        self.Z3 = np.dot(self.latent, self.W3) + self.b3
        self.A3 = self.relu(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.output = self.sigmoid(self.Z4)

        return self.output

    def compute_loss(self, X, output, lambda_l1=0.001):
        mse = np.mean((X - output) ** 2)
        l1_penalty = lambda_l1 * np.mean(np.abs(self.latent))
        return mse + l1_penalty

    def backward(self, X):
        m = X.shape[0]
        
        dZ4 = (self.output - X) * self.sigmoid_derivative(self.Z4)
        dW4 = np.dot(self.A3.T, dZ4) / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m
        
        dA3 = np.dot(dZ4, self.W4.T)
        dZ3 = dA3 * self.relu_derivative(self.Z3)
        dW3 = np.dot(self.latent.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dLatent = np.dot(dZ3, self.W3.T)
        dZ2 = dLatent * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1