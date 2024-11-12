import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class VolumeMA(AbstractFeature):
    def __init__(self, symbol, timeframe):

        self.symbol = symbol
        self.timeframe = timeframe

        self.parents = [
            'Volume' + '__' + symbol
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        volume_col = 'Volume' + '__' + self.symbol
        window = utils.parse_time(self.timeframe)
        return df[volume_col].rolling(window).mean()

