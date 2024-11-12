import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class ZScore(AbstractFeature):
    def __init__(self, symbol, timeframe, moving_average_timeframe):

        self.symbol = symbol
        self.timeframe = timeframe
        self.moving_average_timeframe = moving_average_timeframe

        self.parents = [
            'OpenPrice' + '__' + symbol,
            'SimpleMovingAverage' + '__' + symbol + '_' + self.timeframe + "_" + self.moving_average_timeframe
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        price_col = 'OpenPrice' + '__' + self.symbol
        moving_average_col = 'SimpleMovingAverage' + '__' + self.symbol + '_' + self.timeframe + "_" + self.moving_average_timeframe
        # returnTimeFrame = utils.parse_time(self.timeframe)
        window = utils.parse_time(self.moving_average_timeframe)
        # return np.log(df[price_col] / df[price_col].shift(shift_length))
        return (df[price_col] - df[moving_average_col]) / df[price_col].rolling(window=window).std()
