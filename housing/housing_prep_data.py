import os
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
from mlFunctions import CombinedAttributesAdder, EncoderAndDeleteCol, trainAndPrint, cross_val_scoresAndPrint, DeleteAttributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

plt.style.use('seaborn')

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_TGZ_NAME = "housing.tgz"
DISPLAY_WIDTH = 400
targetName = "median_house_value"

HOUSING_FILE = os.path.join(HOUSING_PATH,HOUSING_TGZ_NAME.replace("tgz","csv"))

if not os.path.exists(HOUSING_FILE):
    fetch_data(url=HOUSING_URL, path=HOUSING_PATH,tgz_name=HOUSING_TGZ_NAME)
housing = load_data(path=HOUSING_PATH)

print2(housing.head(),9)
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print2(housing.describe())

# Stratified Shuffle for test and train data according to median_income
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])  

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

# Deal only with train data
housingOriginal = housing.copy()
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_numeric = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_numeric)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", EncoderAndDeleteCol([0,2,3,4]) , cat_attribs), #OneHotEncoder()
    ])

housing_prepared = full_pipeline.fit_transform(housing)
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

# # Train Model
# lin_reg = LinearRegression()
# lin_reg = trainAndPrint(lin_reg, housing_prepared, housing_labels, "LinReg")
# lin_reg_scores = cross_val_scoresAndPrint(lin_reg, housing_prepared, housing_labels)
# joblib.dump(lin_reg, "lin_reg.pkl")

# tree_reg = DecisionTreeRegressor()
# tree_reg = trainAndPrint(lin_reg, housing_prepared, housing_labels, "TreeReg")
# tree_reg_scores = cross_val_scoresAndPrint(tree_reg, housing_prepared, housing_labels)
# joblib.dump(tree_reg, "tree_reg.pkl")

# forest_reg = RandomForestRegressor()
# forest_reg = trainAndPrint(forest_reg, housing_prepared, housing_labels, "ForestReg")
# forest_reg_scores = cross_val_scoresAndPrint(forest_reg, housing_prepared, housing_labels)
# joblib.dump(forest_reg, "forest_reg.pkl")

# save files
joblib.dump(full_pipeline, "full_pipeline.pkl")
joblib.dump(housing_labels, "housing_labels.pkl")
joblib.dump(housing_prepared, "housing_prepared.pkl")
joblib.dump(num_attribs, "num_attribs.pkl")
joblib.dump(cat_one_hot_attribs, "cat_one_hot_attribs.pkl")
joblib.dump(X_test_prepared,"X_test_prepared.pkl")
joblib.dump(y_test,"y_test.pkl")
joblib.dump(strat_test_set,"strat_test_set.pkl")