import numpy as np
import utilities.utils as utils
from scipy import stats

def ts_rank_price_1d(df, symbol):
    price_col = symbol + '__' + 'open_price'
    mwin = utils.parse_time('1M')
    return df[price_col].rolling(window=60 * mwin).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
