#%% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

#%% LOAD AND SPLIT DATA
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X, X_test, y, y_test = train_test_split(
    mnist.data, mnist.target, test_size=50000, random_state=42)

#%%
# from sklearn.manifold import MDS

# mds = MDS(n_components=2, random_state=42)
# X_reduced_mds = mds.fit_transform(X)

# from sklearn.manifold import Isomap

# isomap = Isomap(n_components=2)
# X_reduced_isomap = isomap.fit_transform(X)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, verbose=3)
t1 = time.time()
X_reduced_tsne = tsne.fit_transform(X)
t2 = time.time()

print('Time to train: '+ str(t2-t1))
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# lda = LinearDiscriminantAnalysis(n_components=2)
# X_mnist = mnist["data"]
# y_mnist = mnist["target"]
# lda.fit(X_mnist, y_mnist)
# X_reduced_lda = lda.transform(X_mnist)

#%% 
X_reduced = X_reduced_tsne
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()

plt.figure(figsize=(9,9))
cmap = mpl.cm.get_cmap("jet")
for digit in (2, 3, 5):
    plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=[cmap(digit / 9)])
plt.axis('off')
plt.show()

#%% RUNING FOR 3 and 5

idx = (y == 2) | (y == 3) | (y == 5) 
X_subset = X[idx]
y_subset = y[idx]

tsne_subset = TSNE(n_components=2, random_state=42, verbose=3)
X_subset_reduced = tsne_subset.fit_transform(X_subset)
plt.figure(figsize=(9,9))
for digit in (2, 3, 5):
    plt.scatter(X_subset_reduced[y_subset == digit, 0], X_subset_reduced[y_subset == digit, 1], c=[cmap(digit / 9)])
plt.axis('off')
plt.show()