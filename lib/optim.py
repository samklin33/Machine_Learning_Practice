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

def perceptron(x_train, y_train, x_test, y_test, learning_rate=0.01, epoch=1000):
    """
    Perceptron algorithm for binary classification.

    Args:
        x_train: Training feature matrix (n_samples, n_features).
        y_train: Training labels (n_samples,), with values in {1, -1}.
        x_test:  Testing feature matrix (n_samples, n_features).
        y_test:  Testing labels (n_samples,), with values in {1, -1}.
    """
    n_features = x_train.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    acc_history = []
    theta_history = []

    for _ in range(epoch):
        indices = np.random.permutation(len(x_train))
        for i in indices:
            xi, yi = x_train[i], y_train[i]
            if sign(np.dot(w, xi) + b) != yi:
                w += learning_rate * yi * xi
                b += learning_rate * yi
            
        preds = np.array([sign(np.dot(w, xi) + b) for xi in x_test])
        acc_history.append(np.mean(preds == y_test))
        theta_history.append((w.copy(), b))

    return w, b, acc_history, theta_history