# Machine Learning

A personal learning workspace for implementing ML algorithms from scratch and comparing them with library implementations.

## Structure

```
Machine_Learning/
├── lib/                  # Shared utilities (data splitting, optimization, plotting)
├── linear_regression/    # Supervised: Linear Regression
└── ...
```

## Topics

### [Linear Regression](linear_regression/)

**Dataset:** sklearn Diabetes dataset (single feature: BMI)

**Approach:**
1. Implemented gradient descent from scratch to learn intercept and slope iteratively
2. Derived the closed-form (analytical) solution using covariance/variance formulas as a reference
3. Compared both against sklearn's `LinearRegression` baseline

**Key observations:**
- Gradient descent converges to the same coefficients as the closed-form solution when the learning rate and epochs are tuned properly
- Feature standardization is critical — without it, a high learning rate causes the loss to diverge
- Visualized regression fit, MSE loss curve (log scale), and parameter convergence across epochs

**Config:** [`linear_regression/config.py`](linear_regression/config.py) — adjust `LEARNING_RATE`, `EPOCH`, `TEST_SIZE`

---

## Shared Library (`lib/`)

| File | Purpose |
|---|---|
| `data.py` | Manual train/test split |
| `optim.py` | Gradient descent with loss and parameter history |
| `plot.py` | Regression fit, loss curve, parameter convergence plots |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
