import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, '..'))
from lib.plot import plot_decision_boundary, plot_accuracy_curve, plot_parameter_convergence
from config import TARGET_NAME, TARGET_CLASS, LEARNING_RATE, EPOCH, TEST_SIZE

if __name__ == "__main__":
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
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.iloc[:, :2].values, iris_data['target_class'].values, test_size=TEST_SIZE, random_state=42)
    print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

    # Perceptron training
    clf = Perceptron(max_iter=EPOCH, eta0=LEARNING_RATE, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")

    # Visualization
    colors = {1: 'blue', -1: 'red'}
    labels = {1: 'setosa', -1: 'versicolor'}
    param_labels = ['w[0] (sepal length)', 'w[1] (sepal width)', 'b (bias)']

    w = clf.coef_[0]
    b = clf.intercept_[0]

    fig, (ax) = plt.subplots(1, 1)
    plot_decision_boundary(ax, x_train, y_train, x_test, y_test, w, b, labels, colors)
    ax.set_xlabel('Sepal length (cm)')
    ax.set_ylabel('Sepal width (cm)')

    plt.tight_layout()
    plt.show()