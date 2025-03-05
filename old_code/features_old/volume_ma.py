import numpy as np
import utilities.utils as utils
from scipy import stats

def volume_ma(df, symbol, timeframe):
    volume_col = symbol + '__' + 'volume'
    window = utils.parse_time(timeframe)
    df[volume_col].rolling(window).mean()