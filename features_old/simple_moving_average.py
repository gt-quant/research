import numpy as np
from .lin_return import lin_return
import utilities.utils as utils
def simple_moving_average(df, symbol, moving_average_timeframe, return_timeframe):
    lin_ret = lin_return(df, symbol, return_timeframe)
    # return_col = symbol + '__' + 'log_return__' + return_timeframe
    window = utils.parse_time(moving_average_timeframe)
 
    return lin_ret.rolling(window=window).mean()

