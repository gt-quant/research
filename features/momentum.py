import numpy as np

def momentum(df, symbol, timeframe):
    price_col = symbol + '__' + 'open_price'

    if timeframe == "1M":
        shift_length = 1
    elif timeframe == "1D":
        shift_length = 1440
    elif timeframe == "6H":
        shift_length = 360
    else:
        raise Exception("Unknown timeframe.")

    return df[price_col].pct_change(shift_length)
