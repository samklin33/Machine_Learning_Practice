import numpy as np


def gradient_descent(x, y, theta, learning_rate=0.01, epoch=1000):
    x_flat = x.flatten()
    loss_history = []
    theta_history = []
    for _ in range(epoch):
        predictions = theta[0] + theta[1] * x_flat
        error = predictions - y
        loss_history.append(np.mean(error ** 2))
        theta_history.append(theta.copy())
        theta[0] -= learning_rate * np.mean(error)
        theta[1] -= learning_rate * np.mean(error * x_flat)
    return theta, loss_history, theta_history
