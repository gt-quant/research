from utils import *
from feature_factory import add_feature_col_inplace, get_feature_col
from backtester import BackTester

def basic_lin_regr_model1():

    DATA_PATH = "historic_bybit/data"
    trading_symbols = [
        'ALPHAUSDT',
        'SUSHIUSDT',
        'ETHUSDT',
        'SILLYUSDT',
        'ZENUSDT',
        'REQUSDT',
        'WIFUSDT'
    ]
    symbols = trading_symbols + ['BTCUSDT']

    df = get_combined_data(symbols, DATA_PATH)

    # Feature Engineering
    # add_return_cols_inplace(df, 'open', [5, 60, 1440])
    for symbol in symbols:
        df.rename(columns={f'{symbol}_open': f'{symbol}__open_price'}, inplace=True)
        add_feature_col_inplace(df, f'{symbol}__log_return__1D')
    
    # Change to use new func
    # beta_df = find_all_beta(backtest_df, train_test_cut, symbols, "FIXED", "1D", 'LOG')
    BTC_BETA = 1.05

    # y_df = get_future_return_cols(df, '_open_price', [1440])
    # mn_df = get_MN_cols(y_df, BTC_BETA)

    y_df = pd.DataFrame()
    for symbol in symbols:
        feat = f'{symbol}__future_log_return__1D'
        y_df[feat] = get_feature_col(df, feat)
    
    mn_df = pd.DataFrame()
    for symbol in trading_symbols:
        mn_df[f'{symbol}__future_log_return__1D_MN'] = y_df[f'{symbol}__future_log_return__1D'] - BTC_BETA * y_df['BTCUSDT__future_log_return__1D']

    print("y_df: ", y_df.columns)
    print("mn_df: ", mn_df.columns)

    y_col='REQUSDT__future_log_return__1D_MN'

    datasets = get_datasets(df, mn_df, y_col)

    X_train, X_test, y_train, y_test = datasets
    test_start_date = X_test.index[0]

    model = get_LR_model(datasets)

    strategy_df = df.dropna().copy()

    strategy_df.loc[:, 'REQUSDT_signal'] = pd.Series(model.predict(strategy_df), index=strategy_df.index)

    train_signal_std = strategy_df[strategy_df.index < test_start_date]['REQUSDT_signal'].std()

    strategy_df.loc[:, 'REQUSDT_signal'] = strategy_df['REQUSDT_signal'] / train_signal_std

    return strategy_df

def get_mapping_func(low_bound, high_bound):
    def get_ideal_pos(signal):
        if abs(signal) < low_bound:
            return 0.0
        elif abs(signal) > high_bound:
            return np.sign(signal)
        
        return (signal - np.sign(signal) * low_bound) / (high_bound - low_bound)
    return get_ideal_pos

def monetization_strategy1(strategy_df):

    backtest_df = pd.DataFrame()
    
    pos_mapping_func = get_mapping_func(2.5, 3.5)

    backtest_df['REQUSDT_price'] = strategy_df['REQUSDT__open_price']
    backtest_df['BTCUSDT_price'] = strategy_df['BTCUSDT__open_price']
    backtest_df['REQUSDT_pos'] = strategy_df['REQUSDT_signal'].map(pos_mapping_func)

    return backtest_df
