EDGES = {
    'log_return': ['open_price'],
    'gio_feature': ['log_return__1M', 'close_price'],
    'momentum': ['open_price'],
    'simple_moving_average': ['log_return__6H', 'log_return__12H', 'log_return__1D', 'log_return__4D'],
    'z_score': ['open_price', 'simple_moving_average__6H', 'simple_moving_average__12H', 'simple_moving_average__1D', 'simple_moving_average__4D']
}
