import os
import pandas as pd
import numpy as np
import seaborn as sns
from FetchData import fetch_data, load_data
from util import print2, split_train_test, split_train_test_by_id, compareCategoryProportionsSamples
from util import getCorrVector
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from mlFunctions import CombinedAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

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
    
# Deal only with train data
housingOriginal = housing.copy()
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

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
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# corrTarget, mostCorrelatedVarNames = getCorrVector(housing_prepared,targetName)
# print(corrTarget)
# fig = sns.pairplot(housing, vars = mostCorrelatedVarNames[:5])