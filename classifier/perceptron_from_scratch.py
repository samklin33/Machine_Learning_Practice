import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets
from config import TARGET_NAME, TARGET_CLASS, LEARNING_RATE, EPOCH, TEST_SIZE

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, '..'))
from lib.data import split_data
from lib.optim import sign, perceptron
from lib.plot import plot_decision_boundary, plot_accuracy_curve, plot_parameter_convergence

if __name__ == '__main__':
    # Load the Iris dataset
    iris = datasets.load_iris()
    x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y = pd.DataFrame(iris['target'], columns=['target'])

    # Data preprocessing
    iris_data = pd.concat([x, y], axis=1)
    iris_data['target_name'] = iris_data['target'].map(TARGET_NAME)
    iris_data = iris_data[iris_data['target_name'].isin(['setosa', 'versicolor'])]
    iris_data['target_class'] = iris_data['target_name'].map(TARGET_CLASS)
    iris_data = iris_data[['sepal length (cm)', 'sepal width (cm)', 'target_class']]

    # Data splitting
    x_train, x_test, y_train, y_test = split_data(iris_data.iloc[:, :2].values, iris_data['target_class'].values, test_size=TEST_SIZE)
    print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

    # Perceptron training
    start = time.time()
    w, b, acc_history, theta_history = perceptron(x_train, y_train, x_test, y_test, learning_rate=LEARNING_RATE, epoch=EPOCH)
    final_acc = acc_history[-1]
    print(f"Test accuracy: {final_acc:.4f}")
    print(f"Training time: {time.time() - start:.4f} seconds")

    # Visualization
    colors = {1: 'blue', -1: 'red'}
    labels = {1: 'setosa', -1: 'versicolor'}
    theta_arr = [[*t[0], t[1]] for t in theta_history]
    param_labels = ['w[0] (sepal length)', 'w[1] (sepal width)', 'b (bias)']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    plot_decision_boundary(ax1, x_train, y_train, x_test, y_test, w, b, labels, colors)
    ax1.set_xlabel('Sepal length (cm)')
    ax1.set_ylabel('Sepal width (cm)')
    plot_accuracy_curve(ax2, acc_history)
    plot_parameter_convergence(ax3, theta_arr, labels=param_labels)

    plt.tight_layout()
    plt.savefig(f"{path}/figure/perceptron_epoch={EPOCH}_learning_rate={LEARNING_RATE}.png")
    plt.show()
