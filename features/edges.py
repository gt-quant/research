EDGES = {
    'log_return': ['open_price'],
    'diff_return': ['open_price'],
    'gio_feature': ['log_return__1M', 'close_price'],
    'future_log_return': ['open_price'],
    'fee_truncated_future_log_return': ['future_log_return__1D']
}
