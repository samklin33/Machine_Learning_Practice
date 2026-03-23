import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import TARGET_NAME, TARGET_CLASS, LEARNING_RATE, EPOCH, TEST_SIZE

from lib.plot import plot_decision_boundary

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
        iris_data[['sepal length (cm)', 'sepal width (cm)']].values,
        iris_data['target_class'].values,
        test_size=0.3,
        random_state=0
    )

    # Feature scaling
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # Train logistic regression model
    lr = LogisticRegression()
    lr.fit(x_train_std, y_train)
    lr.predict_proba(x_test_std)
    print(f"Training accuracy: {lr.score(x_train_std, y_train):.4f}")
    print(f"Testing accuracy: {lr.score(x_test_std, y_test):.4f}")

    # Visualization
    colors = {1: 'blue', -1: 'red'}
    labels = {1: 'setosa', -1: 'versicolor'}
    x_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_boundary(plt.gca(), x_combined_std, y_combined, x_test_std, y_test, lr.coef_[0], lr.intercept_[0], labels, colors)
    plt.xlabel('sepal length (standardized)')
    plt.ylabel('sepal width (standardized)')
    plt.legend(loc='upper left')
    plt.title('Logistic Regression Decision Regions')
    plt.show()