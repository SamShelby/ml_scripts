#%% IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib as mpl
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib

#%% LOAD AND SPLIT DATA
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

#%% TRAIN MODELS
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42, verbose=2)
svm_clf = LinearSVC(random_state=42, verbose=2)
mlp_clf = MLPClassifier(random_state=42, verbose=2)

#%%
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

for estimator in estimators:
    print(estimator.score(X_val, y_val))
# print([estimator.score(X_val, y_val) for estimator in estimators])


#%% VOTING CLASSIFIER
voting_clf = VotingClassifier(
    estimators=[('erf', extra_trees_clf), ('rf', random_forest_clf),
                ('mlp', mlp_clf),('svc', svm_clf)])
voting_clf.fit(X_train, y_train)


voting_score = voting_clf.score(X_val, y_val)
print(voting_score)

#%% CLEAN CLASSIFIER
# Remove worste model
voting_clf.set_params(svc=None)
del voting_clf.estimators_[3]
print(voting_clf.score(X_val, y_val))

voting_clf.voting = "soft"
print(voting_clf.score(X_val, y_val))

voting_clf.voting = "hard"

#%% TEST
print(voting_clf.score(X_test, y_test))

print([estimator.score(X_test, y_test) for estimator in voting_clf.estimators_])

#%% SAVE MODELS
data_all = [X_train, X_val, X_test, y_train, y_val, y_test]
joblib.dump(estimators, 'estimators.pkl')
joblib.dump(data_all, 'data_all.pkl')