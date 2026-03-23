import numpy as np
import matplotlib.pyplot as plt


def plot_regression_fit(ax, x_train, y_train, x_test, y_test, x_line, y_lines):
    """Scatter train/test data and overlay one or more fitted lines.

    y_lines: list of (y_values, label, color) tuples
    """
    ax.scatter(x_train, y_train, color='gray', label='Train', s=5)
    ax.scatter(x_test, y_test, color='green', label='Test', s=5)
    for y_vals, label, color in y_lines:
        ax.plot(x_line, y_vals, color=color, label=label, linewidth=2)
    ax.set_title('Linear Regression Fit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()


def plot_loss_curve(ax, loss_history):
    """Plot MSE loss over epochs on a log scale."""
    ax.plot(loss_history, color='purple')
    ax.set_title('Gradient Descent Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_yscale('log')


def plot_parameter_convergence(ax, theta_history):
    """Plot how each parameter evolves over epochs."""
    theta_history = np.array(theta_history)
    ax.plot(theta_history[:, 0], label='Intercept (theta[0])', color='orange')
    ax.plot(theta_history[:, 1], label='Slope (theta[1])', color='cyan')
    ax.set_title('Parameter Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Parameter Value')
    ax.legend()
