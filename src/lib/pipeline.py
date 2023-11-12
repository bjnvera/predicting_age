import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from sklearn.neighbors import KernelDensity


class oRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, thresh):
        self.model = model
        self.thresh = thresh

    def fit(self, X, y):
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        X_rm, y_rm = self._rm_outliers(X, y)
        self.model.fit(X_rm, y_rm)

        return self

    def _rm_outliers(self, X, y):
        print("Hello")

        df = pd.DataFrame(X)
        num_obs_before = df.shape[0]
        X_cols = df.columns
        self.model.fit(X, y)
        # df['y'] = y
        df['predictions'] = self.model.predict(X)
        df['residuals'] = df.apply(lambda x: abs(x['predictions'] - x['y']), axis=1)
        df['prob'] = self._get_prob(y)
        df['anomaly_score'] = df['prob'] * df['residuals']

        print(df['anomaly_score'].describe())
        subdf = df[df['anomaly_score'] < self.thresh]
        # y_formatted = np.array(subdf['y']).ravel()
        y_formatted = subdf['y']
        print(f"#Outliers removed: {num_obs_before - subdf.shape[0]}")

        return subdf[X_cols], y_formatted

    def _get_prob(self, y):
        array_y = np.array(y).reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(array_y)
        log_density = kde.score_samples(array_y)

        return np.exp(log_density)

    def predict(self, X):
        return self.model.predict(X)