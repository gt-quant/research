import numpy as np
import utilities.utils as utils
def simple_moving_average(df, symbol, moving_average_timeframe, return_timeframe):
    return_col = symbol + '__' + 'log_return__' + return_timeframe

    window = utils.parse_time(moving_average_timeframe)
 
    return df[return_col].rolling(window=window).mean()

