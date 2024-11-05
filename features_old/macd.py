import numpy as np
import utilities.utils as utils
import ta

def macd(df, symbol):
    price_col = symbol + '__' + 'open_price'
    macd = ta.trend.MACD(df['mn_close'], window_slow=26*60, window_fast=12*60, window_sign=9*60)
    return macd.macd()
