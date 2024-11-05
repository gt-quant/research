import numpy as np
from .lin_return import lin_return
import utilities.utils as utils

def rsi(df, symbol, timeframe, window_size_str):
    delta = lin_return(df, symbol, timeframe)
    window_size = utils.parse_time(window_size_str)

    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))