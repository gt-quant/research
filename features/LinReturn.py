import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class LinReturn(AbstractFeature):
    def __init__(self, symbol, timeframe):

        self.symbol = symbol
        self.timeframe = timeframe

        self.parents = [
            'OpenPrice' + '__' + symbol
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        price_col = 'OpenPrice' + '__' + self.symbol
        shift_length = utils.parse_time(self.timeframe)
        return np.log(df[price_col] - df[price_col].shift(shift_length))

