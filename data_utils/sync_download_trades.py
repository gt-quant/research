import os
import requests
import zipfile
from io import BytesIO
from datetime import datetime, timedelta
import gzip
import io

def download_and_save_data(url, ticker, date, save_directory):
    try:
        response = requests.get(url, timeout=30)
        # print(content)
        # exit(0)
        response.raise_for_status()

        # with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        #     data_content = f.read().decode('utf-8')
        #     save_path = os.path.join(save_directory, ticker, f"{date.replace('-', '')}.data")
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     with open(save_path, 'wb') as f:
        #         f.write(data_content)
        #     print(f"Saved {ticker} data for {date}")
        
        save_path = os.path.join(save_directory, ticker, f"{date.replace('-', '')}.data")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
            with open(save_path, 'wb') as csv_file:
                csv_file.write(gz_file.read())

        # save_path = os.path.join(save_directory, ticker, f"{date.replace('-', '')}.data")
        # with gzip.open(save_path, 'rb') as gz_file, open(save_path, 'wb') as f:
        #     f.write(gz_file.read())
        # print(f"Decompressed file saved to {decompressed_path}")

        
        # with zipfile.ZipFile(BytesIO(response.content)) as z:
        #     data_file_name = f"{date}_{ticker}_ob500.data"
        #     if data_file_name in z.namelist():
        #         data_content = z.read(data_file_name)
                
        #         save_path = os.path.join(save_directory, ticker, f"{date.replace('-', '')}.data")
        #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
        #         with open(save_path, 'wb') as f:
        #             f.write(data_content)
                
        #         print(f"Saved {ticker} data for {date}")
        #     else:
        #         print(f"No .data file found in the zip for {ticker} on {date}")
    except Exception as e:
        print(f"Failed to download/save data for {ticker} on {date}: {str(e)}")

def get_symbols():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get("https://api.bybit.com/v5/market/tickers?category=linear", headers=headers)
        response.raise_for_status()
        data = response.json()
        return [res['symbol'] for res in data['result']['list']]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching symbols: {str(e)}")
        return []

def download_orderbook_data(start_date, end_date, symbols, save_directory):
    # save_directory = "test_historical_orderbook"
    # symbols = get_symbols()
    # Use specific symbols for testing
    # symbols = ['BTCUSDT']

    base_url = "https://public.bybit.com/trading/{symbol}/{symbol}{date}.csv.gz"
    # base_url = "https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{date}_{symbol}_ob500.data.zip"
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    for symbol in symbols:
        for current_date in date_range:
            date_str = current_date.strftime('%Y-%m-%d')
            save_path = os.path.join(save_directory, symbol, f"{date_str.replace('-', '')}.data")
            
            if not os.path.exists(save_path):
                download_url = base_url.format(symbol=symbol, date=date_str)
                print(f"Downloading {symbol} data for {date_str}...")
                download_and_save_data(download_url, symbol, date_str, save_directory)
            else:
                print(f"Data for {symbol} on {date_str} already exists. Skipping download.")

if __name__ == "__main__":
    start_date = datetime(2025, 3, 2)
    end_date = datetime(2025, 3, 3)
    symbols = ['BANUSDT']
    save_directory = "data/trades"
    download_orderbook_data(start_date, end_date, symbols, save_directory)
