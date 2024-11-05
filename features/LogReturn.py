import numpy as np
import utilities.utils as utils

from abstract_feature import AbstractFeature

class LogReturn(AbstractFeature):
    def __init__(self, symbol, timeframe):

        self.symbol = symbol
        self.timeframe = timeframe

        parents = [
            'OpenPrice' + '__' + symbol
        ]

    def get_parents(self):
        return parents

    def get_feature(self, df):
        price_col = 'OpenPrice' + '__' + self.symbol
        shift_length = utils.parse_time(self.timeframe)
        return np.log(df[price_col] / df[price_col].shift(shift_length))

