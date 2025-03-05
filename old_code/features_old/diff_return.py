import numpy as np

def diff_return(df, symbol, timeframe):
    price_col = symbol + '__' + 'open_price'

    if timeframe == "1M":
        shift_length = 1
    elif timeframe == "6H":
        shift_length = 360
    elif timeframe == "1D":
        shift_length = 1440
    else:
        raise Exception("Unknown timeframe.")

    return (df[price_col] / df[price_col].shift(shift_length)) - 1.0
