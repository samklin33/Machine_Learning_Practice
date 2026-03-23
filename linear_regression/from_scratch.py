import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from config import LEARNING_RATE, EPOCH, TEST_SIZE

from lib.data import split_data
from lib.optim import gradient_descent
from lib.plot import plot_regression_fit, plot_loss_curve, plot_parameter_convergence

path = os.path.dirname(os.path.abspath(__file__))
print(f"learning_rate: {LEARNING_RATE}, epoch: {EPOCH}, test_size: {TEST_SIZE}")

if __name__ == '__main__':
    data = datasets.load_diabetes()

    X = data.data[:, np.newaxis, 2]
    X = (X - X.mean()) / X.std()  # Standardize feature
    x_train, x_test, y_train, y_test = split_data(X, data.target, test_size=TEST_SIZE)

    # gradient descent
    x_flat = x_train.flatten()
    def linear_grad_fn(theta):
        predictions = theta[0] + theta[1] * x_flat
        error = predictions - y_train
        loss = np.mean(error ** 2)
        grads = np.array([np.mean(error), np.mean(error * x_flat)])
        return loss, grads

    start = time.time()
    theta = np.zeros(2)
    theta, loss_history, theta_history = gradient_descent(linear_grad_fn, theta, learning_rate=LEARNING_RATE, epoch=EPOCH)
    print(f"Gradient Descent Coefficients: Intercept = {theta[0]:.4f}, Slope = {theta[1]:.4f}")
    print(f"Gradient Descent Final Loss: {loss_history[-1]:.4f}")
    print(f"Gradient Descent Time: {time.time() - start:.4f} seconds")

    # closed-form solution
    m = np.cov(x_train.flatten(), y_train, bias=True)[0, 1] / np.var(x_train)
    b = np.mean(y_train) - m * np.mean(x_train)
    print(f"Closed-form Coefficients: Intercept = {b:.4f}, Slope = {m:.4f}")

    x_line = np.linspace(X.min(), X.max(), 100)
    y_gd = theta[0] + theta[1] * x_line
    y_cf = m * x_line + b

    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    plot_regression_fit(ax1, x_train, y_train, x_test, y_test, x_line,
                        [(y_gd, 'Gradient Descent', 'blue'), (y_cf, 'Closed-form', 'red')])
    plot_loss_curve(ax2, loss_history)
    plot_parameter_convergence(ax3, theta_history)
    plt.tight_layout()
    plt.savefig(f'{path}/figure/epoch={EPOCH}_learning_rate={LEARNING_RATE}_normalized.png')
    plt.show()