import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class RSI(AbstractFeature):
    def __init__(self, symbol, timeframe, window):

        self.symbol = symbol
        self.timeframe = timeframe
        self.window = window
        self.parents = [
            'LinReturn' + '__' + symbol + "_" + self.timeframe + "_" + self.window
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        returnCol = 'LinReturn' + '__' + self.symbol + "_" + self.timeframe
        shift_length = utils.parse_time(self.window)
        gain = (df[returnCol].where(df[returnCol] > 0, 0)).rolling(window=self.window).mean()
        loss = (-df[returnCol].where(df[returnCol] < 0, 0)).rolling(window=self.window).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

