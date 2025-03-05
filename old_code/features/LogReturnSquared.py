import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class LogReturnSquared(AbstractFeature):
    def __init__(self, symbol, timeframe):

        self.parents = [
            'LogReturn' + '__' + symbol + '_' + timeframe
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        log_return_col = self.parents[0]
        return df[log_return_col] ** 2

