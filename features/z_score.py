import numpy as np

def z_score(df, symbol, moving_average_timeframe, return_timeframe):
    price_col = symbol + '__' + 'open_price'
    moving_average_col = symbol + '__' + 'simple_moving_average__' + moving_average_timeframe

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


    return (df[price_col] - df[moving_average_col]) / df[price_col].rolling(window=window).std()


