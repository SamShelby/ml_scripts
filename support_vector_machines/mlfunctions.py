import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def grid_search_svr(param_grid, data, labels, cv=3, verbose=2):
    svr = SVR() 
    svr_grid_search = GridSearchCV(svr, param_grid=param_grid, cv=cv,
                               scoring='neg_mean_squared_error',
                               return_train_score=True, verbose=verbose)
    svr_grid_search.fit(data, labels)
    svr_gs_res = compile_results_gs(svr_grid_search)
    
    return svr_grid_search, svr_gs_res

def compile_results_gs(grid_search):
    cvres = grid_search.cv_results_
    gs_res = pd.DataFrame(zip(-cvres["mean_test_score"], cvres["params"]),
                 columns=['mean_test_score','params'])
    gs_res.sort_values('mean_test_score', inplace=True, ascending=True)
    return gs_res

def test_model(model, X, y):
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    return y_predictions, rmse


def rmse_from_predictions(model, X, y, name=""):
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    print("")
    print(name,"RMSE: ","{:.2f}".format(rmse))
    print(name, " Model: ", "{:.2f}".format(model.score(X,y)))
    return rmse, y_predictions