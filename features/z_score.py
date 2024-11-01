import numpy as np
from .simple_moving_average import simple_moving_average
import utilities.utils as utils

def z_score(df, symbol, moving_average_timeframe, return_timeframe):
    price_col = symbol + '__' + 'open_price'
    moving_average_col = simple_moving_average(df, symbol, moving_average_timeframe, return_timeframe)
    window = utils.parse_time(moving_average_timeframe)


    return (df[price_col] - moving_average_col) / df[price_col].rolling(window=window).std()


