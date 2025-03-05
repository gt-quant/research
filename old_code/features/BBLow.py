import numpy as np
import utilities.utils as utils
import ta
from .AbstractFeature import AbstractFeature

class BBLow(AbstractFeature):
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
        window = utils.parse_time(self.timeframe)

        bollinger = ta.volatility.BollingerBands(df[price_col], window=window)

        return bollinger.bollinger_lband()

