import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# Read data from csv files
df_x_train = pd.read_csv('./data/X_train.csv').drop(columns="id")
df_y_train = pd.read_csv('./data/y_train.csv').drop(columns="id")

# Define ML pipeline as follows
# RobustScaler for outlier removing, default to Interquartile Range IQR)
# KNNImputer to insert missing data according to k-nearest neighbors
# SelectFromModel for feature selection: XGBRegressor
# Regression model: XGBRegressor

# Define model for feature selection
feature_selector = XGBRegressor(
    n_estimators = 1000,
    learning_rate = 0.02,
    random_state = 42
)

pipe = Pipeline([
('scaler', preprocessing.RobustScaler())
, ('imputer', KNNImputer())
, ('feature_selector', SelectFromModel(feature_selector))
, ('regression_model', XGBRegressor())
])

# Set parameters of the pipeline
pipe.set_params(
    imputer__missing_values=np.nan,
    imputer__n_neighbors=5,
    regression_model__device="gpu", # enables GPU acceleration for XGBoostRegressor
    regression_model__n_estimators=1000,
    regression_model__learning_rate=0.005,
    regression_model__subsample=0.6,
)

# Train pipeline on entire training data
pipe.fit(df_x_train, np.array(df_y_train).ravel())

# Make predictions for submission
df_sample = pd.read_csv('./sample.csv')
df_sample.head()
df_x_test = pd.read_csv('./X_test.csv').drop(columns="id")
df_sample['y'] =pipe.predict(df_x_test)
df_sample.to_csv("predictions.csv", index=False)
