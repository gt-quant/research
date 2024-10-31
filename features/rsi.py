import numpy as np
from .lin_return import lin_return

def rsi(df, symbol, timeframe):

    delta = lin_return(df, symbol, "1M")
    # print(delta)
    gain = (delta.where(delta > 0, 0)).rolling(window=100).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=100).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))