import numpy as np
import utilities.utils as utils
from .lin_return import lin_return
def volatility(df, symbol, vol_timeframe, return_timeframe):
    lin_ret = lin_return(df, symbol, return_timeframe)
    window = utils.parse_time(vol_timeframe)
    return lin_ret.rolling(window=window).std()