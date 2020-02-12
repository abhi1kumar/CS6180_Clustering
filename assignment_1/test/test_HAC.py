import numpy as np

X = np.array([[1,2],
    [2,2],
    [3,4],
    [4,4],
    [5,6],
    [6,6],
    [7,8],
    [8,8],
    [9,10],
    [10,10],])

import matplotlib.pyplot as plt

labels = range(1, 11)
"""
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
"""

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(X, 'single')
labelList = range(0, 10)

print(X)
print(linked)
print(labelList)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
