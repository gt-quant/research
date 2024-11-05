import numpy as np

def fee_truncated_future_log_return(df, symbol):
    prev_col = symbol + '__' + 'future_log_return__1D'
    
    def truncate(x):
        if abs(x) < 0.00023:
            return 0.0
        elif x >= 0.00023:
            return x - 0.00023
        else:
            return x + 0.00023

    return df[prev_col].map(truncate)
