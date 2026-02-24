import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        # Initialize dimensions and learning rate
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        
        # Initialize weights and biases
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.b_visible = np.zeros(n_visible)
        self.b_hidden = np.zeros(n_hidden)

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sample_prob(self, probs):
        # Sample binary states from probabilities
        return (probs > np.random.rand(*probs.shape)).astype(float)

    def forward(self, v):
        # Compute hidden probabilities and sample hidden states
        h_prob = self.sigmoid(np.dot(v, self.W) + self.b_hidden)
        h_sample = self.sample_prob(h_prob)
        return h_prob, h_sample

    def backward(self, h):
        # Reconstruct visible probabilities and sample
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.b_visible)
        v_sample = self.sample_prob(v_prob)
        return v_prob, v_sample

    def train(self, X):
        # Positive phase
        h_prob, h_sample = self.forward(X)

        # Negative phase (reconstruction)
        v_recon_prob, v_recon_sample = self.backward(h_sample)
        h_recon_prob, _ = self.forward(v_recon_sample)

        # Update weights and biases (Contrastive Divergence)
        self.W += self.lr * (np.dot(X.T, h_prob) - np.dot(v_recon_sample.T, h_recon_prob)) / X.shape[0]
        self.b_visible += self.lr * np.mean(X - v_recon_sample, axis=0)
        self.b_hidden += self.lr * np.mean(h_prob - h_recon_prob, axis=0)

        # Compute reconstruction error
        error = np.mean((X - v_recon_prob) ** 2)
        return error