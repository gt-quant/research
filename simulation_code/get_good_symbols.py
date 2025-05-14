import requests

# Define the base URLs for the Bybit API
instruments_url = "https://api.bybit.com/v5/market/instruments-info"
tickers_url = "https://api.bybit.com/v5/market/tickers"

# Function to get all perpetual contract tickers
def get_perp_tickers():
    params = {
        "category": "linear",  # Use 'linear' for USDT perpetual contracts
        "status": "Trading"    # Get only active trading symbols
    }
    response = requests.get(instruments_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data['retCode'] == 0:
            # Extracting symbols for perpetual contracts
            return [(instrument['symbol'], instrument['priceFilter']['tickSize']) for instrument in data['result']['list']]
        else:
            print(f"Error: {data['retMsg']}")
            return []
    else:
        print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
        return []

# Function to get the relevant data (volume24h, tickSize, and bid1Price) for each ticker
def get_ticker_data():
    params = {
        "category": "linear",  # Use 'linear' for USDT perpetual contracts
        # "symbol": symbol       # Get data for a specific symbol
    }
    response = requests.get(tickers_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data['retCode'] == 0 and data['result']['list']:
            ticker = data['result']['list'][0]

            return {ticker['symbol']: {'turnover24h': ticker['turnover24h'], 'bid1Price': ticker['bid1Price']} for ticker in data['result']['list']}
            # return {
            #     "symbol": symbol,
            #     "volume24h": volume_24h,
            #     "tickSize": tick_size,
            #     "bid1Price": bid_1_price
            # }
        else:
            print(f"Error: Data not found for")
            return None
    else:
        print(f"Failed to retrieve data for . HTTP Status Code: {response.status_code}")
        return None

# Main function to fetch and display all relevant data for each perpetual contract
def main():
    tickers = get_perp_tickers()
    data_map = get_ticker_data()

    symbols = []
    volumes = []
    ticksizes = []
    price = []
    # sprd = []
    if tickers:
        for symbol, tick_size in tickers:
            # data = get_ticker_data(symbol)
            if symbol in data_map:
                data = data_map[symbol]
                symbols.append(symbol)
                volumes.append(int(float(data['turnover24h'])))
                ticksizes.append(float(tick_size))
                price.append(float(data['bid1Price']))

                print(f"Symbol: {symbol}")
                print(f"Volume (24h): {data['turnover24h']}")
                print(f"Tick Size: {tick_size}")
                print(f"Best Bid Price: {data['bid1Price']}")
                print(f"Spread: {float(tick_size)/float(data['bid1Price']) * 10000:.2f} bps")
                print("-" * 40)
            else:
                print("none")
    import pandas as pd
    df = pd.DataFrame({
        'symbol': symbols,
        'volume': volumes,
        'ticksize': ticksizes,
        'price': price
    })
    df['spread'] = df['ticksize'] / df['price'] * 10000

    df_sorted = df.sort_values(by='spread', ascending=False)
    print(df_sorted)
    df_sorted.to_csv("symbol_info.csv", index=False)

# Run the main function
if __name__ == "__main__":
    main()
