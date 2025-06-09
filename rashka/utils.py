import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ['o', 's', '^', 'v', '<']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl, 0], X[y==cl, 1], marker=markers[idx], c=colors[idx], label=f'Class = {cl}', edgecolors='black')
    plt.legend()
    plt.show()

def plot_linear_equation(X: np.ndarray, y: np.ndarray, w_list: list[list[float]]):
    """
    Plots a linear equation y = mx + c.

    Parameters:
    m (float): The slope of the line.
    c (float): The y-intercept of the line.
    x_range (tuple): The range of x values to plot (default: -10 to 10).
    """
    plt.scatter(
        X[y == 0, 0], X[y == 0, 1], color="red", marker="o", label="Iris-setosa"
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1], color="blue", marker="x", label="Iris-versicolor"
    )
    plt.legend(loc="upper left")
    x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)  #
    for w_ in w_list:
        m = -round(float(w_[0] / w_[1]), 2)
        c = -round(float(w_[2] / w_[1]), 2)

        y_predict = m * x + c

        plt.plot(x, y_predict, label=f"y = {m}x + {c}")
        plt.xlabel("x")
        plt.ylabel("y")
    plt.title(f"Graph of y = {m}x + {c}")
    plt.grid(True)
    plt.legend()
    plt.show()