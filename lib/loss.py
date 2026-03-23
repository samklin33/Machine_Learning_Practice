import numpy as np

def bce(y_true, y_pred):
    """Binary Cross-Entropy Loss"""
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse(y_true, y_pred):
    """Mean Squared Error Loss"""
    return np.mean((y_true - y_pred) ** 2)