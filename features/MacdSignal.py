import numpy as np
import utilities.utils as utils
import ta
from .AbstractFeature import AbstractFeature

class Macd(AbstractFeature):
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
        mwin = utils.parse_time('1M')
        macd = ta.trend.MACD(df[price_col], window_slow=26*60*mwin, window_fast=12*60*mwin, window_sign=9*60*mwin)
        return macd.macd_signal()

