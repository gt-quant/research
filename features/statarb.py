import numpy as np
import utilities.utils as utils

from .AbstractFeature import AbstractFeature

class Statarb(AbstractFeature):
    def __init__(self, symbol1, symbol2, beta, intercept, std):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.beta = float(beta)
        self.intercept = float(intercept)
        self.std = float(std)

        self.parents = [
            'OpenPrice' + '__' + symbol1,
            'OpenPrice' + '__' + symbol2,
        ]

    def get_parents(self):
        return self.parents

    def get_feature(self, df):
        prices1_col, prices2_col = self.get_parents()
        signal = df[prices1_col] - (self.beta*df[prices2_col] + self.intercept) # we expect signal == 0 on avg.

        position_dollars = abs(df[prices1_col]) + abs(df[prices2_col]*self.beta) # dollars Transacted for 1 synthetic
        signal = signal / position_dollars # Edge per dollars of position

        return signal
    
        # Desired position based on signal
        position_volume = signal * (100/self.std) # Volume such that we do $100 worth of synthetic per std.
        desired_position = {symbol1: -position_volume, symbol2: self.beta*position_volume}