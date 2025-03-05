import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


class FeatureTester:
    def __init__(self, df, feature, target):
        self.df = df
        self.feature = feature
        self.target = target
        self.X = df[[feature]].values.reshape(-1, 1)
        self.y = df[target].values

    def correlation_analysis(self):
        pearson_corr, _ = pearsonr(self.df[self.feature], self.df[self.target])
        spearman_corr, _ = spearmanr(self.df[self.feature], self.df[self.target])
        return {
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
        }

    def feature_importance(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        importance = model.feature_importances_[0]
        return importance

    def predictive_power(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        return {
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
        }

    def autocorrelation(self, lag=1):
        series = self.df[self.feature]
        return series.autocorr(lag)

    def plot_feature_vs_target(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[self.feature], self.df[self.target], alpha=0.6)
        plt.xlabel(self.feature)
        plt.ylabel(self.target)
        plt.title(f"Scatter Plot: {self.feature} vs {self.target}")
        plt.show()

    def rolling_correlation(self, window=30):
        rolling_corr = self.df[self.feature].rolling(window).corr(self.df[self.target])
        return rolling_corr

    def feature_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.df[self.feature], bins=30, alpha=0.7)
        plt.xlabel(self.feature)
        plt.title(f"Distribution of {self.feature}")
        plt.show()

    def signal_to_noise_ratio(self):
        mean_signal = np.mean(self.df[self.feature])
        std_noise = np.std(self.df[self.feature])
        snr = mean_signal / std_noise if std_noise != 0 else np.nan
        return snr

    def cross_correlation(self, max_lag=1440):
        lags = range(-max_lag, max_lag + 1, 30)
        cross_corr = [self.df[self.feature].shift(lag).corr(self.df[self.target]) for lag in lags]
        return pd.Series(cross_corr, index=lags)