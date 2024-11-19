import numpy as np
import pyarrow as pa
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error
from types import SimpleNamespace
import statsmodels.api as sm


def get_combined_data(symbols, data_path):
    dfs = []
    for symbol in symbols:
        # print(symbol)
        df = pd.read_csv(f"{data_path}/{symbol}.csv")
        df = df.iloc[::-1].reset_index(drop=True).set_index("time", inplace=False)
        df.columns = [f"{symbol}_{col}" for col in df.columns]
        # print(df)
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=1)
    return combined_df

def get_t_stat(df, x_col, y_df, y_col):

    df_combined = pd.concat([df[[x_col]], y_df[[y_col]]], axis=1)
    df_cleaned = df_combined.dropna()

    X = df_cleaned[[x_col]]
    y = df_cleaned[y_col]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # print(model.summary())

    # Access t-statistics directly
    t_stats = model.tvalues
    # print("T-statistics:\n", t_stats)
    return t_stats[x_col]


def find_beta(df, y_col, x_col):
    df_cleaned = df[[x_col, y_col]].dropna()
    X = df_cleaned[[x_col]]
    y = df_cleaned[y_col]
    model = LinearRegression()
    model.fit(X, y)
    # print(f'Intercept: {model.intercept_}')
    # print(f'Coefficient: {model.coef_[0]}') 
    return model.coef_[0]

def add_return_cols_inplace(df, price_col, intervals):
    for col in df.columns:
        symbol, *_, val = col.split("_")
        if val == price_col:
            for interval in intervals:
                df[symbol + "_return_" + str(interval)] = np.log(df[col]/df[col].shift(interval))

def get_future_return_cols(df, price_col, intervals):
    new_df = pd.DataFrame()
    for col in df.columns:
        symbol, *_, val = col.split("_")
        if val == price_col:
            for interval in intervals:
                new_df[symbol + "_future_return_" + str(interval)] = np.log(df[col].shift(-1 * interval)/df[col])
    return new_df

def get_btc_beta_dict(df, symbols, BETA_RETURN_LEN=3*1440):
    df = df.copy()
    add_return_cols_inplace(df, 'open', [BETA_RETURN_LEN])

    # BETA_RETURN_LEN = 4320
    BTC_BETA = {}

    for symbol in symbols:
        if symbol == "BTCUSDT": continue
        beta = find_beta(df, f'{symbol}_return_{BETA_RETURN_LEN}', f'BTCUSDT_return_{BETA_RETURN_LEN}')
        BTC_BETA[symbol] = beta
    # print(BTC_BETA)
    return BTC_BETA

def get_MN_cols(y_df, btc_beta):
    mn_df = pd.DataFrame()
    for col in y_df.columns:
        symbol = col.split("_")[0]
        col_suffix = "_".join(col.split("_")[1:])
        if symbol != "BTCUSDT":
            new_col_name = col + "_MN"
            mn_df[new_col_name] = y_df[col] - btc_beta * y_df["BTCUSDT_"+col_suffix]
    return mn_df

def get_datasets(features, mn_df, y_col='REQUSDT_future_return_1440_MN'):
    df_combined = pd.concat([features, mn_df[y_col]], axis=1)
    df_cleaned = df_combined.dropna()
    y = df_cleaned[y_col]
    X = df_cleaned.drop(columns=[y_col])

    datasets = train_test_split(X, y, test_size=0.2, shuffle=False)
    return datasets

def get_LR_model(datasets):
    X_train, X_test, y_train, y_test = datasets

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
    print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"R^2 score: {r2_score(y_test, y_test_pred):.4f}")

    y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
    print(f"train MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"train R^2 score: {r2_score(y_train, y_train_pred):.4f}")
    
    return model

    backtest_df = pd.concat([X_test, y_test], axis=1)
    # backtest_df["PRED_" + y_col] = y_test_pred

    backtest_df['signal'] = y_test_pred / y_train_pred.std()

    info = SimpleNamespace()
    info.train_pred_return_std = y_train_pred.std()

    return model, backtest_df, info


