import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from mlfunctions import grid_search_svr, rmse_from_predictions
from sklearn.decomposition import PCA
from util import plot_pca_with_hue

# Load models and data
housing_labels      = joblib.load("datasets/housing_labels.pkl")
housing_prepared    = joblib.load("datasets/housing_prepared.pkl")
num_attribs         = joblib.load("datasets/num_attribs.pkl")
cat_one_hot_attribs = joblib.load("datasets/cat_one_hot_attribs.pkl")
X_test_prepared     = joblib.load("datasets/X_test_prepared.pkl")
y_test              = joblib.load("datasets/y_test.pkl")
targetName = "median_house_value"


#%% LINEAR SVR
param_grid = [
    {'kernel':['linear'], 'C': [200, 300]},
  ]

svr_lin_gs, svr_lin_gs_res = grid_search_svr(param_grid, housing_prepared, housing_labels, cv=3)
print('\n BEST PARAMETERS: '+str(svr_lin_gs.best_params_)+'\n' + str(svr_lin_gs_res))
best_lin_model = svr_lin_gs.best_estimator_

#%% POLY SVR
param_grid = [
    {'kernel':['poly'], 'C': [200, 300], 'gamma': [.1,.5], 'degree': [2]},
  ]

svr_poly_gs, svr_poly_gs_res = grid_search_svr(param_grid, housing_prepared, housing_labels, cv=3)
print('\n BEST PARAMETERS: '+str(svr_poly_gs.best_params_)+'\n' + str(svr_poly_gs_res))
best_poly_model = svr_poly_gs.best_estimator_

#%% RBF SVR
param_grid = [
    {'kernel':['rbf'], 'C': [200, 300], 'gamma': [.1,.5]},
  ]

svr_rbf_gs, svr_rbf_gs_res = grid_search_svr(param_grid, housing_prepared, housing_labels, cv=3)
print('\n BEST PARAMETERS: '+str(svr_rbf_gs.best_params_)+'\n' + str(svr_rbf_gs_res))
best_rbf_model = svr_rbf_gs.best_estimator_

#%% GRID SEARCH SVR
param_grid = [
    {'kernel':['linear'], 'C': [10, 50, 100]},
    {'kernel':['poly'], 'C': [100, 200], 'degree': [2,3]},
  ]

#%% TEST MODELS
rmse_lin,   y_lin  = rmse_from_predictions(best_lin_model, X_test_prepared, y_test, "lin model")
rmse_poly,  y_poly = rmse_from_predictions(best_poly_model, X_test_prepared, y_test, "poly model")
rmse_rbf,   y_rbf  = rmse_from_predictions(best_rbf_model, X_test_prepared, y_test, "rbf model")

ax = plt.subplots()
plt.plot(((y_lin-y_test.to_numpy())**2)**0.5,'.', alpha=0.2,label='lin')
plt.plot(((y_poly-y_test.to_numpy())**2)**0.5,'.', alpha=0.2,label='poly')
plt.plot(((y_rbf-y_test.to_numpy())**2)**0.5,'.', alpha=0.2,label='rbf')
plt.legend()
plt.title('Error in USD')



