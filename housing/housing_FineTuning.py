import pandas as pd
import numpy as np
import seaborn as sns
from FetchData import fetch_data, load_data
from util import print2, compareCategoryProportionsSamples
from util import getCorrVector,display_scores
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from mlFunctions import CombinedAttributesAdder, trainAndPrint, cross_val_scoresAndPrint, testAndPrint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
import joblib

# Load models and data
housing_labels      = joblib.load("housing_labels.pkl")
housing_prepared    = joblib.load("housing_prepared.pkl")
num_attribs         = joblib.load("num_attribs.pkl")
cat_one_hot_attribs = joblib.load("cat_one_hot_attribs.pkl")
full_pipeline       = joblib.load("full_pipeline.pkl")
strat_test_set      = joblib.load("strat_test_set.pkl")
targetName = "median_house_value"

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     print(np.sqrt(-mean_score), params)

forest_reg_calibrated = RandomForestRegressor(**grid_search.best_params_)
forest_reg_calibrated = trainAndPrint(forest_reg_calibrated, housing_prepared, housing_labels, "ForestReg")
forest_reg_calibrated = cross_val_scoresAndPrint(forest_reg_calibrated, housing_prepared, housing_labels)

feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
feature_importances = sorted(zip(feature_importances, attributes), reverse=True)
print(feature_importances)

 # Test Model
X_test = strat_test_set.drop(targetName, axis=1)
y_test = strat_test_set[targetName].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_model = grid_search.best_estimator_
final_predictions = testAndPrint(final_model, X_test_prepared, y_test, "final model")

#Confidence Interval
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors)))

print(confidence_interval)