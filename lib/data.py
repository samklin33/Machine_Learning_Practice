import numpy as np


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
