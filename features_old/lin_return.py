import numpy as np
import utilities.utils as utils

def lin_return(df, symbol, timeframe):
    price_col = symbol + '__' + 'open_price'

    shift_length = utils.parse_time(timeframe)
    return df[price_col] - df[price_col].shift(shift_length)
