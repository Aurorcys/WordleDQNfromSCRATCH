import numpy as np
import pandas as pd

class NN:
    def __init__(self, all_words):
        n_words = len(all_words)

        self.W1 = np.random.randn(130, 128) * np.sqrt(2.0 / 128)
        self.b1 = np.zeros(128)

        self.W2 = np.random.randn(128, 256) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros(256)

        self.W3 = np.random.randn(256, n_words) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(n_words)

        
        self.word_to_idx = {word: i for i, word in enumerate(all_words)}
        self.n_words = n_words

    def forward(self, state, possible):
        self.state = state

        self.z1 = state @ self.W1 + self.b1
        self.r1 = np.maximum(self.z1, 0)
        self.z2 = self.r1 @ self.W2 + self.b2
        self.r2 = np.maximum(self.z2, 0)
        q_values = self.r2 @ self.W3 + self.b3

        valid_mask = np.ones(self.n_words) * -np.inf
        for word in possible:
            if word in self.word_to_idx:
                valid_mask[self.word_to_idx[word]] = 0
        
        masked_q = q_values + valid_mask  # Invalid words → -inf
        return masked_q
    
    def backward(self, target, chosen_idx):

        q_chosen = self.r2 @ self.W3[:, chosen_idx] + self.b3[chosen_idx]

        d_loss = -2 * (target - q_chosen)

        d_q = np.zeros(self.n_words)
        d_q[chosen_idx] = d_loss

        #layer three
        d_W3 = np.outer(self.r2, d_q)
        d_b3 = d_q
        d_r2 = d_q @ self.W3.T

        #layer two
        d_z2 = d_r2 * (self.z2 > 0)
        d_W2 = np.outer(self.r1, d_z2)
        d_b2 = d_z2
        d_r1 = d_z2 @ self.W2.T 

        #layer one
        d_z1 = d_r1 * (self.z1 > 0)
        d_W1 = np.outer(self.state, d_z1)
        d_b1 = d_z1 

        lr = 0.001
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3