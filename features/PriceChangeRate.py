import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class PriceChangeRate(AbstractFeature):
    def __init__(self, symbol, timeframe, retInterval):

        self.symbol = symbol
        self.timeframe = timeframe
        self.retInterval = retInterval

        self.parents = [
            'OpenPrice' + '__' + symbol
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        price_col = 'OpenPrice' + '__' + self.symbol
        window = utils.parse_time(self.retInterval)
        return df[price_col].pct_change(window)

