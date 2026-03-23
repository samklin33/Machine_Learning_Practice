# lib/

Shared utilities for this ML workspace. Installed as a package via `pip install -e .` from the project root, so any script in the project can import from it directly:

```python
from lib.data import split_data
from lib.optim import gradient_descent, perceptron, sign
from lib.plot import plot_loss_curve, plot_regression_fit, ...
```

---

## Modules

### `data.py` — Data utilities

| Function | Signature | Description |
|----------|-----------|-------------|
| `split_data` | `(x, y, test_size=0.1)` | Randomly shuffle and split arrays into train/test sets. Returns `x_train, x_test, y_train, y_test`. |

**Adding a new function:** Put general-purpose data helpers here (normalization, one-hot encoding, etc.). Keep functions stateless and NumPy-based.

---

### `optim.py` — Optimization algorithms

| Function | Signature | Description |
|----------|-----------|-------------|
| `sign` | `(z)` | Returns `1` if `z >= 0`, else `-1`. |
| `gradient_descent` | `(grad_fn, theta, learning_rate, epoch)` | General-purpose gradient descent. `grad_fn(theta)` must return `(loss, grads)`. Returns `theta, loss_history, theta_history`. |
| `perceptron` | `(x_train, y_train, x_test, y_test, learning_rate, epoch)` | Perceptron binary classifier. Labels must be `{1, -1}`. Returns `w, b, acc_history, theta_history`. |

**Adding a new optimizer:** Follow the same return convention — return the final parameters plus any history lists you want to plot. This keeps the calling code and plotting code consistent.

**`theta_history` convention:**
- For `gradient_descent`: a list of flat arrays (one per epoch), shape `(epoch, n_params)`.
- For `perceptron`: a list of `(w_array, b_scalar)` tuples. Flatten to a 2D array before passing to `plot_parameter_convergence`:
  ```python
  theta_arr = [[*t[0], t[1]] for t in theta_history]
  plot_parameter_convergence(ax, theta_arr, labels=['w[0]', 'w[1]', 'b'])
  ```

---

### `plot.py` — Visualization helpers

All functions take a Matplotlib `Axes` object as the first argument so they work with any subplot layout.

| Function | Signature | Description |
|----------|-----------|-------------|
| `plot_regression_fit` | `(ax, x_train, y_train, x_test, y_test, x_line, y_lines)` | Scatter train/test points and overlay fitted lines. `y_lines` is a list of `(y_values, label, color)` tuples. `x_line` should span the full x range (use `np.linspace`). |
| `plot_loss_curve` | `(ax, loss_history)` | MSE loss over epochs on a log scale. |
| `plot_accuracy_curve` | `(ax, acc_history)` | Classification accuracy over epochs (y-axis capped at 1.05). |
| `plot_parameter_convergence` | `(ax, theta_history, labels=None)` | Plots each column of a 2D `theta_history` array as a separate line. Pass `labels` to name each parameter. |
| `plot_decision_boundary` | `(ax, x_train, y_train, x_test, y_test, w, b, label_map, color_map)` | Scatter 2D points and draw the linear decision boundary `w·x + b = 0`. `label_map = {value: name}`, `color_map = {value: color}`. |

**Adding a new plot function:**
1. Accept `ax` as the first argument.
2. Set title, axis labels, and call `ax.legend()` if lines are labeled.
3. Do not call `plt.show()` — leave that to the calling script.

---

## Setup

Run once from the project root with the venv active:

```bash
pip install -e .
```

After that, `from lib.xxx import ...` works in every script regardless of how deep it is nested. No `sys.path` manipulation needed.

To verify the installation:

```bash
python -c "from lib.data import split_data; print('OK')"
```
