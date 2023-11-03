# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:54:57 2023

@author: yvesa
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile
import seaborn as sns
from xgboost import XGBRegressor
import random
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error as rmse

# Read data
X_train_df = pd.read_csv('./X_train.csv', skiprows=1, header= None)
y_train_df = pd.read_csv('./y_train.csv', skiprows=1, header=None)
X_test_df = pd.read_csv('./X_test.csv', skiprows=1, header=None)

X_train_input = X_train_df.values[:,1:]
X_test_input = X_test_df.values[:,1:]
y_train_input = y_train_df.values[:,1:]


#%% Missing data

# Use K-nearest neighbors (KNN) to find missing values

# number of neighbors to be considered
k = 5
imputer = KNNImputer(n_neighbors = k)

# Perform the KNN imputation on each column

X_train_complete = imputer.fit_transform(X_train_input)
X_test_complete = imputer.fit_transform(X_test_input)

#%% Outlier Detection

# Use Isolation Forest to identify the outliers


pca = PCA(n_components=20)
pca.fit(X_train_complete)
X_train_pca = pca.transform(X_train_complete)

plt.scatter(X_train_pca[:,0], X_train_pca[:,1])

# Create an isolation forest model and fit it
clf = IsolationForest()
clf.fit(X_train_complete)

# Obtain anomaly scores for each sample and sort it
anomaly_scores = clf.decision_function(X_train_complete)
sorted_anomaly_scores = np.sort(anomaly_scores)

# Determine a threshold (xth precentil) -> vary to find best threshold
threshold = np.percentile(sorted_anomaly_scores, 10)

# Classify samples as outleiers (1) or inliners (-1)
outlier_prediction = np.where(anomaly_scores < threshold, 1, -1)

outlier_mask = outlier_prediction == 1
X_train_no_outliers = X_train_complete[~outlier_mask]
y_train_no_outliers = y_train_input[~outlier_mask]

# Count the number of outliers
n_outliers = np.sum(outlier_prediction == 1)

print("Number of outliers", n_outliers)
print("Number of rows before:", X_train_complete.shape)
print("Number of rows after:", X_train_no_outliers.shape)


# Plot after outlier removal
pca.fit(X_train_no_outliers)
X_pca = pca.transform(X_train_no_outliers)
plt.scatter(X_pca[:,0], X_pca[:,1])


#%% Feature selection

# First attempt is a correlation analysis

feature_analysis = XGBRegressor()

feature_analysis.fit(X_train_no_outliers, y_train_no_outliers)

importance = feature_analysis.get_booster().get_score(importance_type='weight')

importance_reduced = dict((k,v) for k, v in importance.items() if v >= 10)

importance_drop = dict((k,v) for k, v in importance.items() if v < 10)

list_imp = list(importance_drop.keys())

# Create data frame from the float matrix
X_train_df_corr = pd.DataFrame(X_train_no_outliers)

for i in range(0, len(importance_drop)):
    list_imp[i] = int(list_imp[i][1:])

    
X_train_df_corr = X_train_df_corr.drop(list_imp, axis = 1)
print(X_train_df_corr.shape)


    
#%%
# Create data frame from the float matrix
# X_train_df_corr = pd.DataFrame(X_train_no_outliers)

# Calculate the correlation matrix
correlation_matrix = X_train_df_corr.corr()


X_train_new = X_train_df_corr.values[:,1:]
y_train = y_train_no_outliers
        
# X_train_new = SelectPercentile(percentile=70).fit_transform(X_train, y_train)

#%% Regression

X_train = X_train_new
y_train = y_train_no_outliers
# X_test = X_test_df_corr.values[:,1:]
print("Number of rows after:", X_train.shape)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=50)


#%% XGBoost

xgb_params = {'n_estimators': 1000,
                'learning_rate': hp.uniform('learning_rate',0.01,0.1),
                'subsample': hp.uniform('subsample', 0.05, 0.7),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.5),
                'max_depth': hp.randint('max_depth', 3),
              'reg_lambda': hp.uniform('reg_lambda', 20, 60),
              'reg_alpha': hp.uniform('reg_alpha', 20, 60)
              }

# xgb_params = {
#                 'learning_rate': np.linspace(0.01, 0.1, 10),
#                 'subsample': np.linspace(0.05, 0.45, 10),
#                 'colsample_bytree': np.linspace(0.1, 1, 10).astype(float),
#                 'max_depth': [1, 2],
#                 'reg_lambda': np.linspace(1, 50, 50),
#               'reg_alpha': np.linspace(1, 50, 500),
#               }

r2_scores = [0,0]

def objective(xgb_params):
    model_XGB = XGBRegressor(objective= 'reg:squarederror',
                         n_estimators = 100,
                         learning_rate= xgb_params['learning_rate'],
                         subsample = xgb_params['subsample'],
                         colsample_bytree = xgb_params['colsample_bytree'],
                         grow_policy = 'lossguide',
                         max_depth = xgb_params['max_depth'],
                         booster = 'gbtree',
                         reg_lambda = xgb_params['reg_lambda'],
                         reg_alpha = xgb_params['reg_alpha'],
                         random_state = 42
                         )
    model_XGB.fit(X_train, y_train)

    y_train_pred = model_XGB.predict(X_train)
    y_val_pred = model_XGB.predict(X_val)

    train_score = r2_score(y_train, y_train_pred)
    val_score = r2_score(y_val, y_val_pred)
    # rmse2 = rmse(y_val,y_val_pred)
    r2_scores.append([val_score, train_score])
    # print("Train score: ", round(train_score,3))
    # print("Validation score: ", round(val_score,3))
    return -val_score

trials = Trials()

best_hp = fmin(fn = objective,
               space = xgb_params,
               algo = tpe.suggest,
               max_evals= 100,
               trials = trials)
print("The best hyperparameters are : ","\n")
print(best_hp)

#%%
objective(best_hp)
