from sklearn import preprocessing
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from pyod.models.inne import INNE
import numpy as np
import umap


def get_preprocessors(df):
    scaler = preprocessing.RobustScaler().fit(df)
    imputer = KNNImputer(missing_values=np.nan, weights="distance").fit(df)
    return scaler, imputer


def reduce_dimension(df):
    reducer = umap.UMAP(n_neighbors=10, n_components=8)
    embedding = reducer.fit_transform(df)
    return pd.DataFrame(embedding)


def detect_outliers(df):
    clf = INNE(n_estimators=300)
    df_np = np.array(df)
    clf.fit(df_np)
    print(pd.Series(clf.labels_).value_counts())

    return clf.labels_ == 0
