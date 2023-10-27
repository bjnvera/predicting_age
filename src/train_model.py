from lib.i_o import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import umap
from sklearn import preprocessing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Read data from files
df_x_train = read_csv("X_train.csv", "../data/").head(100)
df_y_train = read_csv("y_train.csv", "../data/").head(100)
print(f"#Features: {df_x_train.shape[1]}\n#observations: {df_x_train.shape[0]}")

# Scale features
df_impute_before = df_x_train.drop(columns='id')
scaler = preprocessing.RobustScaler().fit(df_impute_before)
df_impute_before = pd.DataFrame(scaler.transform(df_impute_before))

# Impute missing values
if 'id' in df_impute_before.columns:
    df_impute_before = df_impute_before.drop(columns='id')

imp = KNNImputer(missing_values=np.nan, weights="distance")
imp.fit(df_impute_before)
df_x_train_imputed = pd.DataFrame(imp.transform(df_impute_before))
print("Imputed NAs")

# Reduce dimensions
reducer = umap.UMAP(n_neighbors=10, n_components=8)
embedding = reducer.fit_transform(df_x_train_imputed)
df_embedding = pd.DataFrame(embedding)
print("Reduced dimensions")

# Detect out / inliers
from pyod.models.knn import KNN
from pyod.models.kde import KDE
from pyod.models.inne import INNE
clf = INNE(n_estimators=300)
X = np.array(embedding)
clf.fit(X)
pd.Series(clf.labels_).value_counts()

## Join out / inlier column
df_embedding["is_inlier"] = clf.labels_ == 0
df_x_train_imputed['is_inlier'] = clf.labels_ == 0
print(df_x_train_imputed['is_inlier'].value_counts())

## Join y column
df_x_train_imputed['id'] = df_x_train['id']
if 'y' not in df_x_train_imputed.columns:
    df_x_train_imputed = df_x_train_imputed.merge(df_y_train, how='left', left_on='id', right_on='id')

## Remove outliers
df_x_train_inliers = df_x_train_imputed[df_x_train_imputed['is_inlier']].drop(columns=['id', 'is_inlier'])
# df_x_train_inliers = df_x_train_imputed.drop(columns=['id', 'is_inlier'])
print("Removed outliers")

# Split into train, val, test
from sklearn.model_selection import train_test_split

# Prepare data for splitting
X = df_x_train_inliers.drop(columns='y')
y = df_x_train_inliers['y']

scaler = preprocessing.RobustScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=1)
print("Splitted data")

# Train variable selector
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# Variable selection model
lasso_model = LassoCV(cv=20, random_state=92, n_jobs=-1, selection="random", max_iter=30000).fit(X_train, y_train)
var_selector = SelectFromModel(lasso_model, prefit=True)
num_features_before = X_train.shape[1]
X_train = var_selector.transform(X_train)
print(f"#Features before {num_features_before} after {X_train.shape[0]}")
X_val = var_selector.transform(X_val)
print("Modelled variable slection")

# Train regression model
from sklearn.ensemble import RandomForestRegressor
reg_model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100, n_jobs=-1)
reg_model.fit(X_train, y_train)
print("Trained regressoin model")

# Evaluate model
train_pred = reg_model.predict(X_train)
val_pred = reg_model.predict(X_val)

validation_score = round(r2_score(y_val, val_pred), 3)
train_score = round(r2_score(y_train, train_pred), 3)
print(f"Training score {train_score}")
print(f"Validation score {validation_score}")