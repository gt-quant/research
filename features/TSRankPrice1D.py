import numpy as np
import utilities.utils as utils
from scipy import stats
from .AbstractFeature import AbstractFeature

class TSRankPrice1D(AbstractFeature):
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
        # shift_length = utils.parse_time(self.timeframe)
        mwin = utils.parse_time('1M')
        return df[price_col].rolling(window=60 * mwin).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))

