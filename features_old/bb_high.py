import numpy as np
import utilities.utils as utils
import ta

def bb_high(df, symbol, timeframe):
    price_col = symbol + '__' + 'open_price'

    window = utils.parse_time(timeframe)

    bollinger = ta.volatility.BollingerBands(df[price_col], window=window)

    return bollinger.bollinger_hband()
