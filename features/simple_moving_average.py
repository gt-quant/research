import numpy as np

def simple_moving_average(df, symbol, moving_average_timeframe, return_timeframe):
    return_col = symbol + '__' + 'log_return__' + return_timeframe

    if moving_average_timeframe == "1D":
        window = 1440
    elif moving_average_timeframe == "6H":
        window = 360
    elif moving_average_timeframe == "12H":
        window = 720
    elif moving_average_timeframe == "4D":
        window = 5760
    else:
        raise Exception("Unknown timeframe.")

    return df[return_col].rolling(window=window).mean()

