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

#%% LOAD DATA & MODELS
data_all = joblib.load('data_all.pkl')
estimators_tuple = joblib.load('estimators_tuple.pkl')

X_train, X_val, X_test, y_train, y_val, y_test = data_all
estimators = list(estimators_tuple)
random_forest_clf, extra_trees_clf, svm_clf, mlp_clf = estimators_tuple

#%% RESULTING PREDICTIONS

X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)
    
#%% BLENDER
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)
print(rnd_forest_blender.oob_score_)


#%% TEST STACKING ENSEMBLE
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
    
y_pred = rnd_forest_blender.predict(X_test_predictions)
print(accuracy_score(y_test, y_pred))