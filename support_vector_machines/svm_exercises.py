##% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from mlfunctions import grid_search, random_search, rmse_from_predictions
from sklearn.decomposition import PCA
from util import plot_pca_with_hue
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from scipy.stats import reciprocal, uniform



#%% LOAD DATA AND SPLIT
housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% LINEAR SVR
lin_svr = LinearSVR(random_state=42, max_iter=2000)
lin_svr.fit(X_train_scaled, y_train)



#%% RANDOMIZED SEARCH
param_distributions = {'kernel':['linear', 'rbf'],"gamma": reciprocal(0.1, 0.4), "C": uniform(1, 50)}
svr_rs, svr_rs_res = random_search(SVR(), param_distributions, X_train_scaled, y_train, n_iter=15, cv=2)
print('\n BEST PARAMETERS: '+str(svr_rs.best_params_)+'\n' + str(svr_rs_res))
best_rand_model = svr_rs.best_estimator_

#%% LINEAR SVR
param_grid = [
    {'kernel':['linear'], 'C': [1, 10, 50]},
  ]

svr_lin_gs, svr_lin_gs_res = grid_search(SVR(),param_grid, X_train_scaled, y_train, cv=2)
print('\n BEST PARAMETERS: '+str(svr_lin_gs.best_params_)+'\n' + str(svr_lin_gs_res))
best_lin_model = svr_lin_gs.best_estimator_

#%% POLY SVR
param_grid = [
    {'kernel':['poly'], 'C': [1, 5], 'gamma': [.01,.1], 'degree': [2]},
  ]

svr_poly_gs, svr_poly_gs_res = grid_search(SVR(),param_grid, X_train_scaled, y_train, cv=2)
print('\n BEST PARAMETERS: '+str(svr_poly_gs.best_params_)+'\n' + str(svr_poly_gs_res))
best_poly_model = svr_poly_gs.best_estimator_

#%% RBF SVR
param_grid = [
    {'kernel':['rbf'], 'C': [1, 5], 'gamma': [.01,.1]},
  ]

svr_rbf_gs, svr_rbf_gs_res = grid_search(SVR(),param_grid, X_train_scaled, y_train, cv=2)
print('\n BEST PARAMETERS: '+str(svr_rbf_gs.best_params_)+'\n' + str(svr_rbf_gs_res))
best_rbf_model = svr_rbf_gs.best_estimator_

#%% GRID SEARCH SVR
param_grid = [
    {'kernel':['linear'], 'C': [10, 50, 100]},
    {'kernel':['poly'], 'C': [100, 200], 'degree': [2,3]},
  ]

#%% TEST MODELS
rmse_lin1,   y_lin1 = rmse_from_predictions(lin_svr,X_test_scaled,y_test, "linSVR model")
rmse_lin,   y_lin  = rmse_from_predictions(best_lin_model, X_test_scaled, y_test, "lin model")
rmse_poly,  y_poly = rmse_from_predictions(best_poly_model, X_test_scaled, y_test, "poly model")
rmse_rbf,   y_rbf  = rmse_from_predictions(best_rbf_model, X_test_scaled, y_test, "rbf model")
rmse_rand,  y_rand  = rmse_from_predictions(best_rand_model, X_test_scaled, y_test, "rand model")

ax = plt.subplots()
plt.plot(((y_lin-y_test)**2)**0.5,'.', alpha=0.2,label='lin')
plt.plot(((y_poly-y_test)**2)**0.5,'.', alpha=0.2,label='poly')
plt.plot(((y_rbf-y_test)**2)**0.5,'.', alpha=0.2,label='rbf')
plt.plot(((y_rbf-y_test)**2)**0.5,'.', alpha=0.2,label='rand')
plt.legend()
plt.title('Error in USD')



