from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


iris = datasets.load_iris()
X = iris["data"][:100, [0, 2]]  # type: ignore # Sepal length and Petal length
y = iris["target"][:100]  # type: ignore

arr = np.array([1, 2, 2, 3, 4, 4, 14])
counts = np.bincount(arr)


print(counts)