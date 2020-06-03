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
import time
from sklearn.decomposition import PCA

#%% LOAD AND SPLIT DATA
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)

#%% TRAIN MODEL

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

t1 = time.time()
rfc.fit(X_train, y_train)
t2 = time.time()

print('Time to train: '+ str(t2-t1))

#%% REDUCE DIMENSION
pca = PCA(n_components=0.95)
t1 = time.time()
X_reduced = pca.fit_transform(X_train)
t2 = time.time()
print('Time to PCA: '+ str(t2-t1))
print('new dimension: ' + str(X_reduced.shape[1]) )

#%% NEW TRAINING
rfc2 = RandomForestClassifier(n_estimators=100, random_state=42)

t1 = time.time()
rfc2.fit(X_reduced, y_train)
t2 = time.time()

print('Time to train: '+ str(t2-t1))

#%% SCORES WITH TRAIN DATA
from sklearn.metrics import accuracy_score

y_pred1 = rfc.predict(X_train)
y_pred2 = rfc2.predict(X_reduced)
print('Train score 1: '+str(accuracy_score(y_pred1,y_train)))
print('Train score 0: '+str(accuracy_score(y_pred2,y_train)))

#%% SCORES WITH TRAIN DATA
from sklearn.metrics import accuracy_score
X_test_reduced = pca.transform(X_test)

y_pred1 = rfc.predict(X_test)
y_pred2 = rfc2.predict(X_test_reduced)
print('Train score 1: '+str(accuracy_score(y_pred1,y_test)))
print('Train score 1: '+str(accuracy_score(y_pred2,y_test)))


#%% LOGIT
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf.fit(X_train, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))

#%%  
y_pred = log_clf.predict(X_test)
print("Score {:.3f}".format(accuracy_score(y_test, y_pred)))

#%%
log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
t0 = time.time()
log_clf2.fit(X_reduced, y_train)
t1 = time.time()
print("Training took {:.2f}s".format(t1 - t0))

#%%  
y_pred = log_clf2.predict(X_test_reduced)
print("Score {:.3f}".format(accuracy_score(y_test, y_pred)))