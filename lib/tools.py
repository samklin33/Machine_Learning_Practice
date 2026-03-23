import numpy as np

def sign(z):
    return 1 if z >= 0 else -1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))