import numpy as np
import utilities.utils as utils
from .AbstractFeature import AbstractFeature

class Carter_Beta(AbstractFeature):
    def __init__(self, symbol, benchmark_symbol, window_size):
        self.symbol = symbol
        self.benchmark_symbol = benchmark_symbol
        print(window_size)
        self.window_size = window_size

        self.parents = [
            'OpenPrice' + '__' + symbol,
            'OpenPrice' + '__' + benchmark_symbol
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        price_col_symbol = 'OpenPrice' + '__' + self.symbol
        price_col_benchmark = 'OpenPrice' + '__' + self.benchmark_symbol
        
        returns_symbol = df[price_col_symbol] - df[price_col_symbol].shift(1)
        returns_benchmark = df[price_col_benchmark] - df[price_col_benchmark].shift(1)

        beta = rolling_linear_regression(returns_benchmark, returns_symbol, int(self.window_size))

        return beta

def rolling_linear_regression(X, y, window_size):
    X_roll_sum = X.rolling(window=window_size, min_periods=1).sum()
    y_roll_sum = y.rolling(window=window_size, min_periods=1).sum()
    X2_roll_sum = (X ** 2).rolling(window=window_size, min_periods=1).sum()
    Xy_roll_sum = (X * y).rolling(window=window_size, min_periods=1).sum()

    print(X)
    print(X_roll_sum)
    # Compute beta (slope of regression line) for each window
    beta = (Xy_roll_sum - (X_roll_sum * y_roll_sum) / window_size) / (X2_roll_sum - (X_roll_sum ** 2) / window_size)
    
    # Fill NaNs in beta with 0 to avoid issues at the beginning
    beta = beta.fillna(0)

    return beta
