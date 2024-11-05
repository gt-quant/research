import numpy as np
import utilities.utils as utils

def rank_price(df, symbol):
    price_col = symbol + '__' + 'open_price'
    return df[price_col].rank(pct=True)
