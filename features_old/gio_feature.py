import numpy as np

def gio_feature(df, symbol):
    return_col = symbol + '__' + 'log_return__1M'
    close_price_col = symbol + '__' + 'close_price'

    return df[return_col] ** 2 + df[close_price_col]
