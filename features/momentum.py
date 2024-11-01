import numpy as np
import utilities.utils as utils

def momentum(df, symbol, timeframe):
    price_col = symbol + '__' + 'open_price'

    shift_length = utils.parse_time(timeframe)

    return df[price_col].pct_change(shift_length)
