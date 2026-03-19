import numpy as np
import matplotlib.pyplot as plt


def plot_regression_fit(ax, x_train, y_train, x_test, y_test, x_line, y_lines):
    """y_lines: list of (y_values, label, color) tuples"""
    ax.scatter(x_train, y_train, color='gray', label='Train', s=5)
    ax.scatter(x_test, y_test, color='green', label='Test', s=5)
    for y_vals, label, color in y_lines:
        ax.plot(x_line, y_vals, color=color, label=label, linewidth=2)
    ax.set_title('Linear Regression Fit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()


def plot_loss_curve(ax, loss_history):
    ax.plot(loss_history, color='purple')
    ax.set_title('Gradient Descent Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_yscale('log')


def plot_accuracy_curve(ax, acc_history):
    ax.plot(acc_history, color='purple')
    ax.set_title('Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1.05)


def plot_parameter_convergence(ax, theta_history, labels=None):
    """theta_history: 2D array-like where each row is one epoch's parameters.
    labels: optional list of names for each column.
    """
    theta_history = np.array(theta_history)
    for i in range(theta_history.shape[1]):
        label = labels[i] if labels else f'theta[{i}]'
        ax.plot(theta_history[:, i], label=label)
    ax.set_title('Parameter Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Parameter Value')
    ax.legend()


def plot_decision_boundary(ax, x_train, y_train, x_test, y_test, w, b, label_map, color_map):
    """label_map: {label_value: display_name}, color_map: {label_value: color}"""
    for label, name in label_map.items():
        mask = y_train == label
        ax.scatter(x_train[mask, 0], x_train[mask, 1],
                   c=color_map[label], label=f'{name} (train)', alpha=0.6)
        mask_t = y_test == label
        ax.scatter(x_test[mask_t, 0], x_test[mask_t, 1],
                   c=color_map[label], marker='*', s=150, label=f'{name} (test)')
    x0_range = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    x1_boundary = -(w[0] * x0_range + b) / w[1]
    ax.plot(x0_range, x1_boundary, 'k-', linewidth=2, label='Decision boundary')
    ax.set_title('Decision Boundary')
    ax.legend()
