def akhil_statarb(df, symbol1, symbol2, beta, intercept, std):
    close1 = symbol1 + '__' + 'close_price'
    close2 = symbol2 + '__' + 'close_price'
    signal = df[close1] - (beta*df[close2] + intercept) # we expect signal == 0 on avg.

    position_dollars = abs(df[close1]) + abs(df[close2]*beta) # $Transacted for 1 synthetic
    signal = signal / position_dollars # Edge per dollars of position

    return signal

    # Desired position based on signal
    position_volume = signal * (100/std) # Volume such that we do $100 worth of synthetic per std.
    desired_position = {symbol1: -position_volume, symbol2: beta*position_volume}