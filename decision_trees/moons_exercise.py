#%% IMPORTS
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mlfunctions import grid_search, random_search, rmse_from_predictions
from sklearn.model_selection import train_test_split

#%% FUNCTIONS
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

#%% MAKE MOONS DATASET
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% LINEAR SVR
param_grid = [
    {'min_samples_split': [2, 3, 4, 5], 'max_leaf_nodes': list(range(2, 30))},
  ]

deep_tree_clf = DecisionTreeClassifier(random_state=42)
dr_gs, dr_gs_res = grid_search(deep_tree_clf,param_grid, X_train, y_train, cv=5, verbose=0)
print('\n BEST PARAMETERS: '+str(dr_gs.best_params_)+'\n' + str(dr_gs_res))
best_dr_model = dr_gs.best_estimator_

# BaseLine
deep_tree_clf_bl = DecisionTreeClassifier(random_state=42)
deep_tree_clf_bl.fit(X_train, y_train)

#%% SCORE
rmse_0,   y_0 = rmse_from_predictions(best_dr_model,X_train,y_train, "gs")
rmse_1,   y_1  = rmse_from_predictions(deep_tree_clf_bl, X_train, y_train, "bl")
        
#%% TEST
rmse_0,   y_0 = rmse_from_predictions(best_dr_model,X_test,y_test, "gs")
rmse_1,   y_1  = rmse_from_predictions(deep_tree_clf_bl, X_test, y_test, "bl")
        
#%% PLOT
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(best_dr_model, X_train, y_train, axes=[-1.5, 2.4, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)
plt.sca(axes[1])
plot_decision_boundary(deep_tree_clf2, X_train, y_train, axes=[-1.5, 2.4, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.ylabel("")

plt.show()

#%% RANDOM FOREST
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
    
#%% TRAIN WITH THE SETS
from sklearn.base import clone
from sklearn.metrics import accuracy_score

forest = [clone(best_dr_model) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(np.mean(accuracy_scores))

#%% FOREST EVALUATION
from scipy.stats import mode

Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))