import numpy as np

def log_return(df, symbol, timeframe):
    price_col = symbol + '__' + 'open_price'

    number = int(timeframe[:-1])
    unit = timeframe[-1]

    if unit == "M":
        shift_length = number
    elif unit == "D":
        shift_length = 1440 * number
    elif unit == "H":
        shift_length = 60 * number
    else:
        raise Exception("Unknown timeframe.")

    return np.log(df[price_col] / df[price_col].shift(shift_length))
