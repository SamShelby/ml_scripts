@@ -0,0 +1,113 @@
import os
import pandas as pd
import numpy as np
import seaborn as sns
from FetchData import fetch_data, load_data
from util import print2, split_train_test, split_train_test_by_id, compareCategoryProportionsSamples
from util import getCorrVector, display_scores
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

housing.hist(bins=50, figsize=(20,15))
plt.tight_layout()

# housing_with_id = housing.reset_index()   # adds an `index` column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
# print(len(housing_with_id["id"].unique())/len(housing_with_id))

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])  
 
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
fig, ax = plt.subplots()
ax = housing["income_cat"].hist()
plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

proportionsComparison = compareCategoryProportionsSamples(
    housing, "income_cat", train_set, strat_train_set)
print(proportionsComparison)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
# Only Deal with train data
housingOriginal = housing.copy()
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.2)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

corrTarget, mostCorrelatedVarNames = getCorrVector(housing,targetName)
print(corrTarget)
fig = sns.pairplot(housing, vars = mostCorrelatedVarNames[:5])

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corrTarget, mostCorrelatedVarNames = getCorrVector(housing,targetName)
print(corrTarget)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")

housing_numeric = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_numeric)
X = imputer.transform(housing_numeric)

housing_filled = pd.DataFrame(X, columns=housing_numeric.columns,
                          index=housing_numeric.index)

housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# housing_cat_hot = pd.get_dummies(housing_filled[['ocean_proximity'],drop_first=True)
