import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import datasets
from config import LEARNING_RATE, EPOCH, TEST_SIZE

path = os.path.dirname(os.path.abspath(__file__))
print(f"learning_rate: {LEARNING_RATE}, epoch: {EPOCH}, test_size: {TEST_SIZE}")

def split_data(x, y, test_size=0.1):
    n = len(x)
    indices = np.arange(n)
    np.random.shuffle(indices)

    test_size = int(n * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test

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

if __name__ == '__main__':
    data = datasets.load_diabetes()

    X = data.data[:, np.newaxis, 2]
    X = (X - X.mean()) / X.std()  # Standardize feature
    x_train, x_test, y_train, y_test = split_data(X, data.target, test_size=TEST_SIZE)

    # gradient descent
    theta = np.zeros(2)
    theta, loss_history, theta_history = gradient_descent(x_train, y_train, theta, learning_rate=LEARNING_RATE, epoch=EPOCH)
    print(f"Gradient Descent Coefficients: Intercept = {theta[0]:.4f}, Slope = {theta[1]:.4f}")
    print(f"Gradient Descent Final Loss: {loss_history[-1]:.4f}")

    # closed-form solution
    m = np.cov(x_train.flatten(), y_train, bias=True)[0, 1] / np.var(x_train)
    b = np.mean(y_train) - m * np.mean(x_train)
    print(f"Closed-form Coefficients: Intercept = {b:.4f}, Slope = {m:.4f}")

    x_line = np.linspace(X.min(), X.max(), 100)
    y_gd = theta[0] + theta[1] * x_line
    y_cf = m * x_line + b

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: fitted lines
    ax1.scatter(x_train, y_train, color='gray', label='Train', s=5)
    ax1.scatter(x_test, y_test, color='green', label='Test', s=5)
    ax1.plot(x_line, y_gd, color='blue', label='Gradient Descent', linewidth=2)
    ax1.plot(x_line, y_cf, color='red', label='Closed-form', linewidth=2)
    ax1.set_title('Linear Regression Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()

    # Plot 2: loss over epochs
    ax2.plot(loss_history, color='purple')
    ax2.set_title('Gradient Descent Learning Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_yscale('log')

    # Plot 3: parameter convergence
    theta_history = np.array(theta_history)
    ax3.plot(theta_history[:, 0], label='Intercept (theta[0])', color='orange')
    ax3.plot(theta_history[:, 1], label='Slope (theta[1])', color='cyan')
    ax3.set_title('Parameter Convergence')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Parameter Value')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'{path}/figure/epoch={EPOCH}_learning_rate={LEARNING_RATE}_normalized.png')
    plt.show()