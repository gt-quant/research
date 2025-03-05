import numpy as np
import pandas as pd

from feature_factory import add_feature_col_inplace, get_feature_col

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def find_all_beta(df, train_test_cut, symbols, beta_mode, return_timeframe, return_mode):

    train_df = df[:train_test_cut].copy()
    # test_df = backtest_df[train_test_cut:].copy()

    beta_df = pd.DataFrame()

    if beta_mode == "FIXED":
        if return_timeframe in ['1M', '1D']:
            if return_mode == "LOG":
                feat = "LogReturn"
            elif return_mode == "DIFF":
                feat = "LinReturn"
            else:
                raise Exception("Unknown Return Mode")
            
            for symbol in symbols:
                train_df[f'OpenPrice__{symbol}'] = train_df[f'{symbol}_price']
                add_feature_col_inplace(train_df, f'{feat}__{symbol}_{return_timeframe}')

            train_df[f'OpenPrice__BTCUSDT'] = train_df[f'BTCUSDT_price']
            add_feature_col_inplace(train_df, f'{feat}__BTCUSDT_{return_timeframe}')

        else:
            raise Exception("Unknown Timeframe")

        for symbol in symbols:
            beta = find_beta(train_df, f'{symbol}__{suffix}', f'BTCUSDT__{suffix}')
            beta_df[f'{symbol}_beta'] = pd.Series(beta, index=df.index)

    elif beta_mode == "CARTER":
        temp_df = pd.DataFrame()
        temp_df[f'OpenPrice__BTCUSDT'] = df[f'BTCUSDT_price']
        for symbol in symbols:
            temp_df[f'OpenPrice__{symbol}'] = df[f'{symbol}_price']
            beta_df[f'{symbol}_beta'] = get_feature_col(temp_df, f'RollingBeta__{symbol}_BTCUSDT_10080')
    else:
        raise Exception("Unknown Mode")
    
    return beta_df

def find_beta(df, y_col, x_col):
    df_cleaned = df[[x_col, y_col]].dropna()
    X = df_cleaned[[x_col]]
    y = df_cleaned[y_col]
    model = LinearRegression()
    model.fit(X, y)
    # print(f'Intercept: {model.intercept_}')
    # print(f'Coefficient: {model.coef_[0]}') 
    return model.coef_[0]

def find_rolling_beta(df, y_col, x_col, window_size):
    df_cleaned = df[[x_col, y_col]].dropna()
    X = df_cleaned[x_col]
    y = df_cleaned[y_col]
    X_roll_sum = X.rolling(window=window_size, min_periods=1).sum()
    y_roll_sum = y.rolling(window=window_size, min_periods=1).sum()
    X2_roll_sum = (X ** 2).rolling(window=window_size, min_periods=1).sum()
    Xy_roll_sum = (X * y).rolling(window=window_size, min_periods=1).sum()

    # Compute beta (slope of regression line) for each window
    beta = (Xy_roll_sum - (X_roll_sum * y_roll_sum) / window_size) / (X2_roll_sum - (X_roll_sum ** 2) / window_size)
    
    # # Fill NaNs in beta with 0 to avoid issues at the beginning
    # beta = beta.fillna(0)

    return beta

class BackTester():
    # Assuming backtest_df has columns with X_price, X_pos, and BTCUSDT_price
    # X_pos has unit of dollars (which is the same as percentage bc we have total balance of 1 dollar)
    # All abs(X_pos) in the same row sum to less than 1
    def __init__(self, backtest_df, train_test_cut=None):
        FEE_RATE = 0.00055
        BETA_MODE = "CARTER"

        if train_test_cut is None:
            train_test_cut = backtest_df.index[int(0.8 * len(backtest_df))]
        
        self.train_test_cut = train_test_cut

        trading_symbols = [col[:-4] for col in backtest_df.columns if col[-4:] == '_pos']

        train_df = backtest_df[:train_test_cut]
        test_df = backtest_df[train_test_cut:]
        
        # Calculate Beta
        beta_df = find_all_beta(backtest_df, train_test_cut, trading_symbols, BETA_MODE, "1D", 'DIFF')
        beta_df.plot()

        # display(beta_df.info())
        # display(beta_df.describe())
        # print((beta_df==0.0).sum())

        df = backtest_df
        result_df = pd.DataFrame()

        # print(f"NaN 0: {df[[col for col in df.columns if col[-4:] in ['_pos', 'rice']]].isna().sum().sum()}")
        # print(f"NaN 0.1: {beta_df.isna().sum().sum()}")
        # print(df.shape, beta_df.shape)

        for symbol in trading_symbols:

            # Unit is number of shares
            result_df[f'{symbol}_actual_pos'] = df[f'{symbol}_pos'] * (1.0 / df[f'{symbol}_price']) * (1.0 / (1.0+abs(beta_df[f'{symbol}_beta'])))
            # print(f"NaN 1: {result_df.isna().sum().sum()}")
            result_df[f'{symbol}_hedge_pos'] = df[f'{symbol}_pos'] * (1.0 / df['BTCUSDT_price']) * (1.0 / (1.0+abs(beta_df[f'{symbol}_beta']))) * -1.0 * beta_df[f'{symbol}_beta']
            # print(f"NaN 2: {result_df.isna().sum().sum()}")

        result_df['BTCUSDT_actual_pos'] = result_df.filter(items=[f'{s}_hedge_pos' for s in trading_symbols]).sum(axis=1)
        # print(f"NaN 3: {result_df.isna().sum().sum()}")

        # Trade Cost Calculation
        for symbol in trading_symbols:
            result_df[f'{symbol}_traded'] = abs(result_df[f'{symbol}_actual_pos'] - result_df[f'{symbol}_actual_pos'].shift(1, fill_value=0.0))
            result_df[f'{symbol}_trade_cost'] = result_df[f'{symbol}_traded'] * df[f'{symbol}_price'] * FEE_RATE
        result_df['main_trade_cost'] = result_df.filter(items=[f'{s}_trade_cost' for s in trading_symbols]).sum(axis=1)
        result_df['hedge_traded'] = abs(result_df['BTCUSDT_actual_pos'] - result_df['BTCUSDT_actual_pos'].shift(1, fill_value=0.0))
        result_df['hedge_trade_cost'] = result_df['hedge_traded'] * df['BTCUSDT_price'] * FEE_RATE

        result_df['total_trade_cost'] = result_df['main_trade_cost'].cumsum() + result_df['hedge_trade_cost'].cumsum()


        # PnL without cost
        for symbol in trading_symbols:
            result_df[f'{symbol}_pnl'] = result_df[f'{symbol}_actual_pos'] * (df[f'{symbol}_price'].shift(-1) - df[f'{symbol}_price'])
            result_df[f'{symbol}_cum_pnl'] = result_df[f'{symbol}_pnl'].cumsum()
        result_df['hedge_pnl'] = result_df['BTCUSDT_actual_pos'] * (df['BTCUSDT_price'].shift(-1) - df['BTCUSDT_price'])
        result_df['hedge_cum_pnl'] = result_df['hedge_pnl'].cumsum()

        result_df['total_pnl'] = result_df.filter(items=[f'{s}_cum_pnl' for s in trading_symbols]).sum(axis=1) + result_df['hedge_cum_pnl']

        # df['total_pnl'] = df['main_cum_pnl'] + df['hedge_cum_pnl']

        result_df['trade_cost_pnl'] = result_df['total_pnl'] - result_df['total_trade_cost']

        self.trading_symbols = trading_symbols
        self.result_df = result_df
    
    def plot_individual_pnl(self):
        """self.result_df.index = pd.to_datetime(self.result_df.index)

        plt.figure(figsize=(100, 80))  # Change the size of the figure to make it more readable
        self.result_df[[f'{s}_cum_pnl' for s in self.trading_symbols] + ['hedge_cum_pnl', 'total_pnl', 'trade_cost_pnl']].plot()
        plt.xticks(rotation=45, ha='right')  # Rotate labels by 45 degrees and align to the right

        plt.axvline(x=pd.to_datetime(self.train_test_cut), color='red', linestyle='--')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()"""
        self.result_df.index = pd.to_datetime(self.result_df.index)
        self.calculate_cumulative_sharpe()  # Add cumulative Sharpe ratios

        # Plot cumulative PnL
        plt.figure(figsize=(12, 8))
        self.result_df[['total_pnl', 'trade_cost_pnl']].plot()
        plt.axvline(x=pd.to_datetime(self.train_test_cut), color='red', linestyle='--', label='Train-Test Split')
        plt.xticks(rotation=45, ha='right')
        plt.title("Total PnL vs Trade-Cost-Adjusted PnL")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL")
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
        
        # Plot cumulative Sharpe ratios
        plt.figure(figsize=(12, 6))
        self.result_df[['cumulative_sharpe_total', 'cumulative_sharpe_cost_adjusted']].plot()
        plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
        plt.xticks(rotation=45, ha='right')
        plt.title("Cumulative Sharpe Ratios (Up to Each Time Point)")
        plt.xlabel("Date")
        plt.ylabel("Sharpe Ratio")
        plt.legend(['Before Cost (Total PnL)', 'After Cost (Trade-Cost-Adjusted PnL)', 'Zero Line'], loc='best')
        plt.grid(True)
        plt.show()

    def calculate_cumulative_sharpe(self):
        """
        Add cumulative Sharpe ratio columns for total PnL and trade-cost-adjusted PnL to result_df.
        """
        # Calculate hourly returns for total PnL and trade-cost-adjusted PnL
        self.result_df['hourly_return_total'] = self.result_df['total_pnl'].diff()
        self.result_df['hourly_return_cost_adjusted'] = self.result_df['trade_cost_pnl'].diff()

        # Cumulative mean and standard deviation of returns for cumulative Sharpe calculation
        cum_mean_total = self.result_df['hourly_return_total'].expanding().mean()
        cum_std_total = self.result_df['hourly_return_total'].expanding().std()

        cum_mean_cost = self.result_df['hourly_return_cost_adjusted'].expanding().mean()
        cum_std_cost = self.result_df['hourly_return_cost_adjusted'].expanding().std()

        # Cumulative Sharpe ratios (annualized for 8760 hours/year)
        self.result_df['cumulative_sharpe_total'] = (cum_mean_total / cum_std_total) * np.sqrt(8760)
        self.result_df['cumulative_sharpe_cost_adjusted'] = (cum_mean_cost / cum_std_cost) * np.sqrt(8760)
        
    def get_result_df(self):
        return self.result_df
