from lib.i_o import read_csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from time import time, sleep

# Read data from csv
df_x_train = read_csv("X_train.csv", "../data/").drop(columns="id")
df_y_train = read_csv("y_train.csv", "../data/").drop(columns="id")

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(df_x_train, df_y_train, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=1)

# Setup model as pipeline and do CV on several hyperparameters such as n_estimators
start_t = time()

selector_model = GradientBoostingRegressor(
    random_state=0,
    n_estimators=70,
    subsample=0.7,
    max_features=0.9,
    learning_rate=0.06
)

pipe = Pipeline([
('scaler', preprocessing.RobustScaler()),
('imputer', KNNImputer()),
('feature_selector', SelectFromModel(selector_model)),
('regression_model', GradientBoostingRegressor())
])

pipe.set_params(
    imputer__missing_values=np.nan,
    imputer__weights="distance",
    feature_selector__threshold="2*mean",
)

# Set grid / solution space
n_max_features_range = np.arange(0.7, 1.1, 0.1)
n_estim_range =  np.arange(30, 90, 5) # 50 - 90
n_subsample_range = np.arange(0.6, 1, 0.1) #0.6 - 0.9
learning_rate_range = np.arange(0.01, 0.2, 0.01)


parameters = {
    'regression_model__subsample': n_subsample_range,
    'regression_model__n_estimators':n_estim_range,
    'regression_model__max_features': n_max_features_range,
    'regression_model__learning_rate': learning_rate_range
}

# Train grided model
grided_model = GridSearchCV(pipe, parameters, scoring='r2', n_jobs=-1, cv=25, return_train_score=True)
grided_model.fit(X_train, np.array(y_train).ravel())

print(f"Elapsed time {round((time()-start_t)/60, 3)} min")


# Assemble evaluation data into a data frame
df_cv_results = pd.DataFrame(grided_model.cv_results_)
df_model_params = df_cv_results.apply(lambda x: pd.Series(x["params"]), axis=1)
df_cv_results[df_model_params.columns] = df_model_params
df_cv_results.sort_values(by="rank_test_score", inplace=True)