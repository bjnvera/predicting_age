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


# Read data
X_train_df = pd.read_csv('./X_train.csv', skiprows=1, header= None)
y_train_df = pd.read_csv('./y_train.csv', skiprows=1, header=None)
X_test_df = pd.read_csv('./X_test.csv', skiprows=1, header=None)

X_train_input = X_train_df.values[:,1:]
X_test_input = X_test_df.values[:,1:]
y_train = y_train_df.values[:,1:]


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

# Create an isolation forest model and fit it
clf = IsolationForest()
clf.fit(X_train_complete)

# Obtain anomaly scores for each sample and sort it
anomaly_scores = clf.decision_function(X_train_complete)
sorted_anomaly_scores = np.sort(anomaly_scores)

# Determine a threshold (xth precentil) -> vary to find best threshold
threshold = np.percentile(sorted_anomaly_scores, 5)

# Classify samples as outleiers (1) or inliners (-1)
outlier_prediction = np.where(anomaly_scores < threshold, 1, -1)

# outlier_mask = outlier_prediction == 1
# X_train_no_outliers = X_train_complete[~outlier_mask]
# y_train_no_outliers = y_train[~outlier_mask]
#!!! need to handle outliers

# Count the number of outliers
n_outliers = np.sum(outlier_prediction == 1)

print("Number of outliers", n_outliers)
print("Number of rows before:", X_train_complete.shape)
# print("Number of rows after:", X_train_no_outliers.shape)

#%% Feature selection

# First attempt is a correlation analysis


# Create data frame from the float matrix
X_train_df_corr = pd.DataFrame(X_train_complete)

# Calculate the correlation matrix
correlation_matrix = X_train_df_corr.corr()

# Set a threshold for high correlation
correlation_threshold = 0.7

# Create a list to store highly correlated feature pairs
correlated_pairs = []

# Loop through the upper triangular part of correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i,j]) >= correlation_threshold:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            correlation_value = correlation_matrix.iloc[i,j]
            correlated_pairs.append((feature1, feature2, correlation_value))

# Sort highly correlated pairs
correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# print("Feature Pairs with high correlation:")
# for pair in correlated_pairs:
#     print(f"{pair[0]} and {pair[1]} (Correlation: {pair[2]})")

# Remove one feature from each highly correlated pair
for pair in correlated_pairs:
    if pair[0] in X_train_df_corr.columns:
        X_train_df_corr.drop(pair[0], axis=1, inplace=True)

#%% Regression

X_train = X_train_df_corr.values[:,1:]
# y_train = y_train_no_outliers
# X_test = X_test_df_corr.values[:,1:]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#%% Lasso
# Scale
# scaler =StandardScaler()
# X_train = scaler.fit_transform(X_train)

# Use Lasso Regression with alpha as regularization strength
# alphas = [10, 20, 30]

# lasso_cv_model = LassoCV(alphas=alphas, cv=10)
# lasso_cv_model.fit(X_train, y_train)

# optimal_alpha = lasso_cv_model.alpha_


# print("Optimal Alpha: ", optimal_alpha)

# lasso_model = Lasso(alpha=optimal_alpha)
# lasso_model.fit(X_train, y_train)

# y_train_pred = lasso_model.predict(X_train)
# y_val_pred = lasso_model.predict(X_val)

# train_score = r2_score(y_train, y_train_pred)
# val_score = r2_score(y_val, y_val_pred)

# print("Train score: ", train_score)
# print("Validation score: ", val_score)

#%% Ridge

alphas = [10, 20, 30]
ridge_cv_model = RidgeCV(alphas=alphas, store_cv_values=True)

ridge_cv_model.fit(X_train, y_train)

optimal_alpha = ridge_cv_model.alpha_

print("Optimal Alpha: ", optimal_alpha)

ridge_model = Ridge(alpha = optimal_alpha)
ridge_model.fit(X_train, y_train)

y_train_pred = ridge_model.predict(X_train)
y_val_pred = ridge_model.predict(X_val)

train_score = r2_score(y_train, y_train_pred)
val_score = r2_score(y_val, y_val_pred)

print("Train score: ", train_score)
print("Validation score: ", val_score)