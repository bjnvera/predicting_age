# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:50:13 2023

@author: yvesa
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

# Read data from csv files
df_x_train = pd.read_csv('./X_train.csv').drop(columns="id")
df_y_train = pd.read_csv('./y_train.csv').drop(columns="id")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df_x_train, df_y_train, test_size=0.2, random_state=42)

# Define the model for feature selection
model = XGBRegressor(n_estimators = 1000,
                     learning_rate = 0.02,
                     random_state = 42)


# RobustScaler for outlier removing, default to Interquartile Range IQR)
# KNNImputer to insert missing data according to k-nearest neighbors
# SelectFromModel for feature selection (XGBRegressor)
# Regression model: XGBRegressor

pipeline = Pipeline([
    ('scalar', preprocessing.RobustScaler()),
    ('imputer', KNNImputer(n_neighbors = 5)),
    ('feature_selector', SelectFromModel(model)),
    ('regression_model', XGBRegressor())])

#%% Train the model

# Parameters can be adapted. Including reg_lambda or reg_alpha didn't lead to any improvements against overfitting.
# However, they were not deeply investigated
parameters = {
    'regression_model__n_estimators': [1000],
    'regression_model__learning_rate': [0.005],
    'regression_model__subsample': [0.6],
}

gridmodel = GridSearchCV(pipeline, parameters, scoring='r2',
                         n_jobs = -1, cv=3, return_train_score=True)

gridmodel.fit(X_train, y_train)

y_train_pred = gridmodel.predict(X_train)
y_val_pred = gridmodel.predict(X_val)

val_score = round(r2_score(y_val, y_val_pred), 3)
train_score = round(r2_score(y_train, y_train_pred), 3)
print("Training score:", train_score)
print("Validation score", val_score)

#%% Prediction

df_sample = pd.read_csv('./sample.csv')

df_sample.head()

df_x_test = pd.read_csv('./X_test.csv').drop(columns="id")
df_sample['y'] =gridmodel.predict(df_x_test)


df_sample.to_csv("predictions.csv", index=False)
