import numpy as np

def log_return(df, symbol, timeframe):
    price_col = symbol + '__' + 'open_price'

    if timeframe == "1M":
        shift_length = 1
    elif timeframe == "1D":
        shift_length = 1440
    elif timeframe == "6H":
        shift_length = 360
    else:
        raise Exception("Unknown timeframe.")

    return np.log(df[price_col] / df[price_col].shift(shift_length))
