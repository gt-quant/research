EDGES = {
    'log_return': ['open_price'],
    'diff_return': ['open_price'],
    'future_log_return': ['open_price'],
    'fee_truncated_future_log_return': ['future_log_return__1D'],
    'lin_return': ['open_price'],
    'gio_feature': ['log_return__1M', 'close_price'],
    'rsi': [],
    'momentum': ['open_price'],
    'simple_moving_average': [],
    'z_score': ['open_price'],
    'volatility': [],
    'bb_low': ['open_price'],
    'bb_high': ['open_price'],
    'macd': ['open_price'],
    'macd_diff': ['open_price'],
    'macd_signal': ['open_price'],
    'price_change_rate': ['open_price'],
    'rank_price': ['open_price'],
    'ts_rank_price_1d' : ['open_price'],
    'volume_ma' : ['volume']

}
