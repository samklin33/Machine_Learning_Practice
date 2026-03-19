import numpy as np


def sign(z):
    return 1 if z >= 0 else -1

def gradient_descent(grad_fn, theta, learning_rate=0.01, epoch=1000):
    """
    General-purpose gradient descent.

    Args:
        grad_fn: callable(theta) -> (loss, grads)
            Returns the scalar loss and gradient array for the current theta.
        theta:   initial parameter array (modified in-place).
    """
    loss_history = []
    theta_history = []
    for _ in range(epoch):
        loss, grads = grad_fn(theta)
        loss_history.append(loss)
        theta_history.append(theta.copy())
        theta -= learning_rate * grads
    return theta, loss_history, theta_history
