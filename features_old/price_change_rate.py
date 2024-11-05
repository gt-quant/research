import numpy as np
import utilities.utils as utils

def price_change_rate(df, symbol, retInterval):
    price_col = symbol + '__' + 'open_price'
    window = utils.parse_time(retInterval)
    return df[price_col].pct_change(window)
