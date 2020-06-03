#%%
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure, axis
from math import ceil, floor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time

#%% 
data = fetch_olivetti_faces()
X = data['data']
y = data['target']

#%%
from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(data.data, data.target))
X_train_valid = data.data[train_valid_idx]
y_train_valid = data.target[train_valid_idx]
X_test = data.data[test_idx]
y_test = data.target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]
    
#%% REDUCE PCA
pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

print(pca.n_components_)
        
#%% ELBOW and SILHOUETTE
k_range =  range(50, 150, 5)
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_train_pca) for k in k_range]
plot_inertia_silhouette(kmeans_per_k, X_train_pca)

silhouette_scores = [silhouette_score(X_train_pca, model.labels_) for model in kmeans_per_k]
k_best=k_range[np.argmax(silhouette_scores)] 
print(k_best)

#%%
kmeans = KMeans(n_clusters=k_best, random_state=42).fit(X_train_pca)
y_pred = kmeans.predict(X_train_pca)

#Plot first 3 clusters
def plot_faces(faces, labels, n_cols=5):
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face.reshape(64, 64), cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()

for cluster_id in np.unique(kmeans.labels_):
    print("Cluster", cluster_id)
    in_cluster = kmeans.labels_==cluster_id
    faces = X_train[in_cluster].reshape(-1, 64, 64)
    labels = y_train[in_cluster]
    plot_faces(faces, labels)
    
    
#%% CLASSIFIER
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
t0 = time.time()
log_reg.fit(X_train, y_train)
t1 = time.time()
print("{}: {:.1f} seconds".format('log reg', t1 - t0))
print(log_reg.score(X_test, y_test))

#%% CLASSIFIER WITH KMEANS AND PCA
pipeline = Pipeline([
    ("PCA", PCA(0.99)),
    ("kmeans", KMeans(n_clusters=k_best, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", 
                                   max_iter=5000, random_state=42)),
])
t0 = time.time()
pipeline.fit(X_train, y_train)
t1 = time.time()
print("{}: {:.1f} seconds".format('full pipeline', t1 - t0))
#%%
print(pipeline.score(X_test, y_test))

#%% OPTIMIZE K
param_grid = dict(kmeans__n_clusters=range(50, 150,2))
grid_clf = GridSearchCV(pipeline, param_grid, cv=2, verbose=2)
t0 = time.time()
grid_clf.fit(X_train, y_train)
t1 = time.time()
print("{}: {:.1f} seconds".format('full pipeline', t1 - t0))
#%%
print(grid_clf.best_estimator_.score(X_test, y_test))

#%% APPEND DATA
X_train_reduced = kmeans.transform(X_train_pca)
X_valid_reduced = kmeans.transform(X_valid_pca)
X_test_reduced = kmeans.transform(X_test_pca)

X_train_extended = np.c_[X_train_pca, X_train_reduced]
X_valid_extended = np.c_[X_valid_pca, X_valid_reduced]
X_test_extended = np.c_[X_test_pca, X_test_reduced]

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_reduced, y_train)
    
print(clf.score(X_valid_reduced, y_valid))

from sklearn.pipeline import Pipeline

for n_clusters in k_range:
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=n_clusters)),
        ("forest_clf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])
    pipeline.fit(X_train_pca, y_train)
    print(n_clusters, pipeline.score(X_valid_pca, y_valid))


#%% GAUSSIAN MODEL
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=40, random_state=42)
y_pred = gm.fit_predict(X_train_pca)

#%% GENERATE SAMPLES
n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gm.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)

plot_faces(gen_faces, y_gen_faces)

#%% MODIFY AND PLOT
n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
darkened = darkened.reshape(-1, 64*64)
y_darkened = y_train[:n_darkened]

X_bad_faces = np.r_[rotated, flipped, darkened]
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])

plot_faces(X_bad_faces, y_bad)

#%%
X_bad_faces_pca = pca.transform(X_bad_faces)
print(gm.score_samples(X_bad_faces_pca))
print(gm.score_samples(X_train_pca[:10]))


#%% RECONSTRUCT
def reconstruction_errors(pca, X):
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.square(X_reconstructed - X).mean(axis=-1)
    return mse

print(reconstruction_errors(pca, X_train).mean())
print(reconstruction_errors(pca, X_bad_faces).mean())

X_bad_faces_reconstructed = pca.inverse_transform(X_bad_faces_pca)
plot_faces(X_bad_faces_reconstructed, y_gen_faces)
