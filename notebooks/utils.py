import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from tqdm import tqdm

def show_heat_map(df):
    corr_matrix = df.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(20, 16))  # Set the figure size
    
    # Define a custom colormap using LinearSegmentedColormap
    colors = ["red", "white", "blue"]  # Red for negative, white for zero, blue for positive
    cdict = {
        'red':   [(0.0, 1.0, 1.0),   # negative values -> red
                  (0.5, 1.0, 1.0),   # 0 (white)
                  (1.0, 0.0, 0.0)],  # positive values -> blue
        'green': [(0.0, 0.0, 0.0),   # no green for negative
                  (0.5, 1.0, 1.0),   # 0 (white)
                  (1.0, 0.0, 0.0)],  # no green for positive
        'blue':  [(0.0, 0.0, 0.0),   # no blue for negative
                  (0.5, 1.0, 1.0),   # 0 (white)
                  (1.0, 1.0, 1.0)]   # positive values -> blue
    }

    # Create the LinearSegmentedColormap
    cmap = LinearSegmentedColormap("RedWhiteBlue", cdict)
    
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f', center=0, cbar=True)

    # Display the heatmap
    plt.title('Correlation Heatmap')
    plt.show()

def nice_hist_series(series, thresh = 0.1):
    # Calculate the 0.1st and 99.9th percentiles
    lower_percentile = series.quantile(thresh * 0.01)
    upper_percentile = series.quantile(1 - thresh * 0.01)

    # Filter the DataFrame
    serie = series[(series >= lower_percentile) & (series <= upper_percentile)]
    return serie


def print_mutual_info(df, xcols, ycol):
    # Separate features (X) and target (y)
    ddf = df[xcols + [ycol]].dropna()
    X = ddf[xcols]
    y = ddf[ycol]

    # Calculate mutual information between X and y
    mi = mutual_info_regression(X, y)

    # Display the mutual information values
    mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})
    print(mi_df)

def aggregate_data(df_raw):
    df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s')

    # Group by 'time' and conditionally sum 'A' based on 'B' == 'X'
    df_buy = df_raw[df_raw['trade_side'] == 'Buy'].groupby('time', as_index=False).agg(
        trade_size_sum=('trade_size', 'sum'),
        trade_price_min=('trade_price', 'min'),
        trade_price_max=('trade_price', 'max')
    )
    df_sell = df_raw[df_raw['trade_side'] == 'Sell'].groupby('time', as_index=False).agg(
        trade_size_sum=('trade_size', 'sum'),
        trade_price_min=('trade_price', 'min'),
        trade_price_max=('trade_price', 'max')
    )

    df_buy.set_index('time', inplace=True)
    df_sell.set_index('time', inplace=True)

    df_ob = df_raw[df_raw['trade_side'].isna()].drop(columns=['trade_side', 'trade_size', 'trade_price'])
    df_ob.set_index('time', inplace=True)

    return df_buy, df_sell, df_ob

def resample_df(df_buy, df_sell, df_ob, time_step):
    df_buy_resampled = df_buy.resample(time_step).agg({
            'trade_size_sum': 'sum',  # Sum for size
            'trade_price_min': 'min',  # Min for min price
            'trade_price_max': 'max'   # Max for max price
        }).rename(columns={'trade_size_sum': 'buy_size', 'trade_price_min': 'buy_price_min',  'trade_price_max': 'buy_price_max'})
    df_sell_resampled = df_sell.resample(time_step).agg({
            'trade_size_sum': 'sum',  # Sum for size
            'trade_price_min': 'min',  # Min for min price
            'trade_price_max': 'max'   # Max for max price
        }).rename(columns={'trade_size_sum': 'sell_size', 'trade_price_min': 'sell_price_min',  'trade_price_max': 'sell_price_max'})
    df_ob_resampled = df_ob.resample(time_step).last()
    df = df_ob_resampled.join(df_buy_resampled.join(df_sell_resampled, how='outer'), how='outer')

    for col in ['buy_size', 'sell_size']:
        df[col] = df[col].fillna(0)

    for col in ['mid']:
        df[col] = df[col].ffill()  

    return df

def load_big_df(start_date_str, end_date_str, name_template, time_step='1T'):
    big_df = []

    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    for date in tqdm(date_range):
        date_str = date.strftime('%Y%m%d')
        filename = name_template.format(date_str)
        df_raw = pd.read_csv(f"../simulation_data/{filename}")
        df_buy, df_sell, df_ob = aggregate_data(df_raw)
        df = resample_df(df_buy, df_sell, df_ob, time_step)
        big_df.append(df)
    
    return pd.concat(big_df, axis=0, ignore_index=False)

def two_sig_signal(df, alpha_col):
    tdf = pd.DataFrame()
    tdf['mean'] = df[alpha_col].rolling(200).mean()
    tdf['std'] = df[alpha_col].rolling(200).std()

    tdf['signal_buy'] = (df[alpha_col] - tdf['mean'] > tdf['std'] * 2) * 1.0
    tdf['signal_sell'] = (df[alpha_col] - tdf['mean'] < tdf['std'] * -2) * 1.0
    return tdf['signal_buy'] - tdf['signal_sell']

def two_sig_signal_multi(df, alpha_col_list):
    tdf = pd.DataFrame()
    for alpha_col in alpha_col_list:
        tdf[f'mean_{alpha_col}'] = df[alpha_col].rolling(200).mean()
        tdf[f'std_{alpha_col}'] = df[alpha_col].rolling(200).std()

        tdf[f'pval_{alpha_col}'] = (df[alpha_col] - tdf[f'mean_{alpha_col}']) / tdf[f'std_{alpha_col}']
    # print(tdf.columns)
    tdf['norm_sum_pval'] = tdf[[f'pval_{col}' for col in alpha_col_list]].sum(axis=1) / np.sqrt(len(alpha_col_list))

    tdf['signal_buy'] = (tdf['norm_sum_pval'] > 2) * 1.0
    tdf['signal_sell'] = (tdf['norm_sum_pval'] < -2) * 1.0
    return tdf['signal_buy'] - tdf['signal_sell']

def get_yulu_df():
    def process(df_raw):
        df_raw = df_raw[['time', 'mid', 'trade_side', 'trade_size', 'trade_price']]
        # display(df_raw)

        df_raw['mul'] = df_raw['trade_size'] * df_raw['trade_price']
        df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s')

        # Group by 'time' and conditionally sum 'A' based on 'B' == 'X'
        df_buy = df_raw[df_raw['trade_side'] == 'Buy'].groupby('time', as_index=False).agg(
            trade_mul_sum=('mul', 'sum'),
            trade_size_sum=('trade_size', 'sum'),
        )
        df_sell = df_raw[df_raw['trade_side'] == 'Sell'].groupby('time', as_index=False).agg(
            trade_mul_sum=('mul', 'sum'),
            trade_size_sum=('trade_size', 'sum'),
        )

        df_buy.set_index('time', inplace=True)
        df_sell.set_index('time', inplace=True)

        df_ob = df_raw[df_raw['trade_side'].isna()].drop(columns=['trade_side', 'trade_size', 'trade_price'])
        df_ob.set_index('time', inplace=True)

        time_step = '1T'
        # SUM_TRADE_SIZE = df_raw['trade_size'].sum()
        df_buy_resampled = df_buy.resample(time_step).agg({
                'trade_size_sum': 'sum',  # Sum for size
                'trade_mul_sum': 'sum',  # Min for min price
            }).rename(columns={'trade_size_sum': 'buy_size', 'trade_mul_sum': 'buy_mul'})
        df_sell_resampled = df_sell.resample(time_step).agg({
                'trade_size_sum': 'sum',  # Sum for size
                'trade_mul_sum': 'sum',  # Min for min price
            }).rename(columns={'trade_size_sum': 'sell_size', 'trade_mul_sum': 'sell_mul'})
        df_ob_resampled = df_ob.resample(time_step).last()
        df = df_ob_resampled.join(df_buy_resampled.join(df_sell_resampled, how='outer'), how='outer')

        for col in ['buy_size', 'sell_size']:
            df[col] = df[col].fillna(0)

        for col in ['mid']:
            df[col] = df[col].ffill()  
        return df

    from tqdm import tqdm
    start_date_str = "20250213"
    end_date_str = "20250314"
    name_template = "output_2025-03-15_{}_HYPEUSDTOB500_HYPEUSDTtrades_VX.csv"

    big_df = []

    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    for date in tqdm(date_range):
        date_str = date.strftime('%Y%m%d')
        filename = name_template.format(date_str)
        df_raw = pd.read_csv(f"../simulation_data/{filename}")
        df1 = process(df_raw)
        # df_buy, df_sell, df_ob = aggregate_data(df_raw)
        # df = resample_df(df_buy, df_sell, df_ob, '30s')
        big_df.append(df1)

    df2 = pd.concat(big_df, axis=0, ignore_index=False)
    df2.to_csv('dataframes/HYPEUSDT_mar16_yulu.csv', index=True)

    df2['vwap'] = (df2['buy_mul'] + df2['sell_mul']) / (df2['buy_size'] + df2['sell_size'])
    df2['alpha1'] = 1.0 * (df2['vwap'] - df2['mid'])
    df2['alpha2'] = df2['alpha1'].ewm(span=3, adjust=False).mean()
    return df2

def plot_box_plot(df, xcol, ycol, q=30):
    # Bin the x values into quantiles using qcut without modifying the original df
    bin_labels, bin_edges = pd.qcut(df[xcol], q=q, labels=False, retbins=True, duplicates='drop')

    # Group by the bin labels and calculate the mean of y for each bin
    bin_means = df.groupby(bin_labels)[ycol].mean()

    # Plot the means
    plt.figure(figsize=(40, 32))
    plt.plot(bin_means.index, bin_means.values, marker='o', linestyle='-', color='b')

    # Add the quantile bin edges to the plot
    for i, edge in enumerate(bin_edges[:-1]):
        # Add a label for each quantile range
        plt.text(i, bin_means[i], f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]', ha='center', va='bottom')

    plt.xlabel('Binned X')
    plt.ylabel('Mean of Y')
    plt.title(f'Mean of {ycol} for Each Quantile Bin of {xcol}')
    plt.xticks(bin_means.index)  # Optional: to label the x-axis with the bin labels
    plt.grid(True)
    plt.show()

def plot_box_plot_a(df, xcol, ycol, q=30):
    # Bin the x values into quantiles using qcut without modifying the original df
    bin_labels, bin_edges = pd.qcut(df[xcol], q=q, labels=False, retbins=True, duplicates='drop')

    # Group by the bin labels and calculate the required statistics
    bin_stats = df.groupby(bin_labels)[ycol].agg(['mean', 'std', 'quantile'])
    
    # Calculate the 25th and 75th percentiles (lower and upper quantiles)
    bin_stats['lower'] = df.groupby(bin_labels)[ycol].quantile(0.25)
    bin_stats['upper'] = df.groupby(bin_labels)[ycol].quantile(0.75)

    bin_stats['lower_1'] = df.groupby(bin_labels)[ycol].quantile(0.1)
    bin_stats['upper_9'] = df.groupby(bin_labels)[ycol].quantile(0.9)

    bin_stats['max'] = df.groupby(bin_labels)[ycol].max()
    bin_stats['min'] = df.groupby(bin_labels)[ycol].min()

    bin_stats['x_mean'] = df.groupby(bin_labels)[xcol].mean()
    
    # Plot the data using candlestick-like plotting (mean, lower, upper, and std)
    plt.figure(figsize=(10, 6))

    # Loop through each quantile bin
    for i, (bin_id, stats) in enumerate(bin_stats.iterrows()):
        # Plot the mean value (center of the candlestick)
        plt.plot([i, i], [stats['lower_1'], stats['upper_9']], color='yellow', lw=16)  # Plot std as red line
        plt.plot([i, i], [stats['mean'] - stats['std'], stats['mean'] + stats['std']], color='red', lw=12)  # Plot std as red line
        plt.plot([i, i], [stats['lower'], stats['upper']], color='black', lw=8)  # Candlestick line
        plt.plot([i, i], [stats['mean'] - stats['std']/10, stats['mean'] + stats['std']/10], color='white', lw=4)  # Plot std as red line
        # plt.plot(i, stats['mean'], 'blue', label=f'Mean Bin {bin_id}')  # Plot the mean as a blue dot

    # Set labels and title
    plt.xlabel('Quantile Bins')
    plt.ylabel(f'{ycol}')
    plt.title(f'{ycol} vs Quantile Bins of {xcol}')
    plt.xticks(range(len(bin_stats)), [f'{i:.4f}' for i in bin_stats['x_mean']], rotation=45)  # Label the x-axis with bin IDs
    plt.grid(True)
    plt.show()

def get_future_df(df_raw, interval='1s'):
    df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s')
    df_raw['round_time'] = df_raw['time'].dt.round(interval)

    rdf = df_raw.set_index('time', inplace=False).resample(interval).last()
    rdf.loc[:,'mid'] = (rdf['bbid'] + rdf['bask']) / 2
    
    rdf.loc[:,'future_1m_mid'] = rdf['mid'].shift(-60)
    rdf.loc[:,'future_1m_return'] = np.log(rdf['mid'].shift(-60)/rdf['mid'])
    rdf.loc[:,'future_1m_std'] = rdf['mid'].shift(-60).rolling(60).std()
    rdf.loc[:,'future_1m_range'] = rdf['mid'].shift(-60).rolling(60).max() - rdf['mid'].shift(-60).rolling(60).min()

    rdf.loc[:,'past_1m_return'] = np.log(rdf['mid']/rdf['mid'].shift(60))

    rrdf = rdf[['future_1m_mid', 'future_1m_return', 'future_1m_std', 'future_1m_range', 'past_1m_return']].reset_index()
    rrdf = rrdf.rename(columns={'time': 'r_time'})

    df_raw_r = pd.merge(df_raw, rrdf, left_on='round_time', right_on='r_time', how='left') #.drop(columns=['r_time'])
    return df_raw_r

def get_future_smoothed_df(df_raw, step='1s', future_steps=60, past_rolling_steps=15, future_rolling_steps=30):
    df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s')
    df_raw['round_time'] = df_raw['time'].dt.round(step)

    rdf = df_raw.set_index('time', inplace=False).resample(step).last()
    rdf.loc[:,'past_smoothed_mid'] = ((rdf['bbid'] + rdf['bask']) / 2).rolling(past_rolling_steps).mean()
    
    future_mid = f'future_smoothed_{str(future_steps)}_mid'
    
    rdf.loc[:, future_mid] = ((rdf['bbid'] + rdf['bask']) / 2).rolling(future_rolling_steps).mean().shift(-future_steps)
    rdf.loc[:,f'future_smoothed_{str(future_steps)}_return'] = np.log(rdf[future_mid]/rdf['past_smoothed_mid'])
    rdf.loc[:,f'future_smoothed_{str(future_steps)}_std'] = rdf[future_mid].rolling(future_steps).std()
    rdf.loc[:,f'future_smoothed_{str(future_steps)}_range'] = rdf[future_mid].rolling(future_steps).max() - rdf[future_mid].rolling(future_steps).min()

    # rdf.loc[:,'past_1m_return'] = np.log(rdf['mid']/rdf['mid'].shift(60))

    rrdf = rdf[[
        'past_smoothed_mid', 
        f'future_smoothed_{str(future_steps)}_mid',
        f'future_smoothed_{str(future_steps)}_std',
        f'future_smoothed_{str(future_steps)}_range',
        f'future_smoothed_{str(future_steps)}_return'
    ]].reset_index()
    rrdf = rrdf.rename(columns={'time': 'r_time'})

    df_raw_r = pd.merge(df_raw, rrdf, left_on='round_time', right_on='r_time', how='left') #.drop(columns=['r_time'])
    return df_raw_r

def process_yulu(df_raw, time_step='1T'):
    df_raw = df_raw[['time', 'mid', 'trade_side', 'trade_size', 'trade_price']]
    # display(df_raw)

    df_raw['mul'] = df_raw['trade_size'] * df_raw['trade_price']
    df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s')

    # Group by 'time' and conditionally sum 'A' based on 'B' == 'X'
    df_buy = df_raw[df_raw['trade_side'] == 'Buy'].groupby('time', as_index=False).agg(
        trade_mul_sum=('mul', 'sum'),
        trade_size_sum=('trade_size', 'sum'),
    )
    df_sell = df_raw[df_raw['trade_side'] == 'Sell'].groupby('time', as_index=False).agg(
        trade_mul_sum=('mul', 'sum'),
        trade_size_sum=('trade_size', 'sum'),
    )

    df_buy.set_index('time', inplace=True)
    df_sell.set_index('time', inplace=True)

    df_ob = df_raw[df_raw['trade_side'].isna()].drop(columns=['trade_side', 'trade_size', 'trade_price'])
    df_ob.set_index('time', inplace=True)

    # time_step = '1T'
    # SUM_TRADE_SIZE = df_raw['trade_size'].sum()
    df_buy_resampled = df_buy.resample(time_step).agg({
            'trade_size_sum': 'sum',  # Sum for size
            'trade_mul_sum': 'sum',  # Min for min price
        }).rename(columns={'trade_size_sum': 'buy_size', 'trade_mul_sum': 'buy_mul'})
    df_sell_resampled = df_sell.resample(time_step).agg({
            'trade_size_sum': 'sum',  # Sum for size
            'trade_mul_sum': 'sum',  # Min for min price
        }).rename(columns={'trade_size_sum': 'sell_size', 'trade_mul_sum': 'sell_mul'})
    df_ob_resampled = df_ob.resample(time_step).last()
    df = df_ob_resampled.join(df_buy_resampled.join(df_sell_resampled, how='outer'), how='outer')

    for col in ['buy_size', 'sell_size']:
        df[col] = df[col].fillna(0)

    for col in ['mid']:
        df[col] = df[col].ffill()  
    return df

def get_future_df_yulu(df_raw, ref_df, interval='1S'):
    df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s')
    df_raw['round_time'] = df_raw['time'].dt.round(interval)
    
    rrdf = ref_df.rename(columns={'time': 'r_time'})

    df_raw_r = pd.merge(df_raw, rrdf, left_on='round_time', right_on='r_time', how='left') #.drop(columns=['r_time'])
    return df_raw_r
