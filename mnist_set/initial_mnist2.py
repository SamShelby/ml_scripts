from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


MNIST_PATH = os.path.join("datasets")
MNIST_FILE = os.path.join(MNIST_PATH,"minst.pkl")

if not os.path.exists(MNIST_FILE):
    os.makedirs(MNIST_PATH, exist_ok=True)
    mnist = fetch_openml('mnist_784', version=1)
    joblib.dump(mnist,"datasets/minst.pkl")
else:
    mnist = joblib.load("datasets/minst.pkl")

X, y = mnist["data"], mnist["target"]

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
some_index = 5

y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Otherwise too slow
X_train, y_train = X_test, y_test

# OvO strategy (one vs one)
svm_clf = SVC()
svm_clf.fit(X_train, y_train) # y_train, not y_train_5
print(svm_clf.predict([some_digit]))

some_digit_scores = svm_clf.decision_function([some_digit])
print(np.argmax(some_digit_scores))
print(svm_clf.classes_)

ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
print(ovr_clf.predict([some_digit]))
len(ovr_clf.estimators_)

# sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))
# print(sgd_clf.decision_function([some_digit]))
# sgd_scores = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# print(sgd_scores)

# sgd_clf = SGDClassifier(random_state=42)
# skfolds = StratifiedKFold(n_splits=3, random_state=42)
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]

#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))  # prints 0.9502, 0.96565, and 0.96495
    
# scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(scores)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)


# y_train_large = (y_train >= 7)
# y_train_odd = (y_train % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_multilabel)
# print(knn_clf.predict([some_digit]))

# noise = np.random.randint(0, 100, (len(X_train), 784))
# X_train_mod = X_train + noise
# noise = np.random.randint(0, 100, (len(X_test), 784))
# X_test_mod = X_test + noise
# y_train_mod = X_train
# y_test_mod = X_test

# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[some_index]])
# plot_digit(clean_digit)