import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class SimpleMovingAverage(AbstractFeature):
    def __init__(self, symbol, returnTimeframe, moving_average_timeframe):

        self.symbol = symbol
        self.returnTimeframe = returnTimeframe
        self.moving_average_timeframe = moving_average_timeframe

        self.parents = [
            'LinReturn' + '__' + symbol + '_' + self.returnTimeframe
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        returnCol = 'LinReturn' + '__' + self.symbol + '_' + self.returnTimeframe
        window = utils.parse_time(self.moving_average_timeframe)
        # return np.log(df[price_col] / df[price_col].shift(shift_length))
        return df[returnCol].rolling(window=window).mean()