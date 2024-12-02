import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Any

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

class DataLibrary():

    def __init__(self, coins, DIRECTORY = 'data', BATCH_SIZE = 20, interval = 5, 
                 start_time = datetime(2024, 6, 1, 0, 0), end_time = datetime(2024, 12, 1, 0, 0)):
        self.symbols = coins
        self.DIRECTORY = DIRECTORY
        self.BATCH_SIZE = BATCH_SIZE
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.dataframes = []
    
    
    def processDFS(self):
        # Check if the number of dataframes matches the number of symbols
        if len(self.dataframes) != len(self.symbols):
            # print(len(dataframes), len(symbols))
            raise ValueError("The number of dataframes must match the number of symbols.")
        
        # Prepare a list to hold the renamed DataFrames
        renamed_dfs = []
        
        # Rename columns for each dataframe and append to the list
        for i in range(len(self.dataframes)):
            df = self.dataframes[i].copy()
            df.columns = ['time', f'open_{self.symbols[i]}', f'high_{self.symbols[i]}', 
                        f'low_{self.symbols[i]}', f'close_{self.symbols[i]}', 
                        f'volume_{self.symbols[i]}', f'turnover_{self.symbols[i]}']
            renamed_dfs.append(df)
        
        # Concatenate all DataFrames on the 'time' column
        combined_df = pd.concat(renamed_dfs, axis=1)
        
        # Drop duplicate 'time' columns if necessary
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

        # Sort by time if needed
        # combined_df.sort_values(by='time', inplace=True)

        return combined_df.reset_index(drop=True)

    async def get_data_async(self, session: aiohttp.ClientSession, symbol: str, start_time: int, end_time: int) -> List[List[Any]]:
        all_data = []
        while start_time <= end_time:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": self.interval,
                "start": start_time,
                "end": end_time,
                "limit": 1000
            }
            async with session.get("https://api.bybit.com/v5/market/kline", params=params) as response:
                data = await response.json()
                ll = data['result']['list']
                if not ll:
                    break
                last_time = int(ll[-1][0]) - 1000
                end_time = last_time
                all_data.extend(ll)
        return all_data

    async def process_symbol(self, session: aiohttp.ClientSession, symbol: str, start_time: datetime, end_time: datetime) -> None:
        path = os.path.join(self.DIRECTORY, f"{symbol}.pq")
        
        if os.path.exists(path):
            df = pd.read_parquet(path)  # Removed parse_dates argument
            df['time'] = pd.to_datetime(df['time'])  # Convert 'time' to datetime
            start_time = df['time'].max() + timedelta(minutes=1)
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        all_data = await self.get_data_async(session, symbol, start_timestamp, end_timestamp)
        
        if all_data:
            col_names = ['time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            new_df = pd.DataFrame(all_data, columns=col_names).astype(float)
            new_df['time'] = pd.to_datetime(new_df['time'], unit='ms') - timedelta(hours=4)
            
            if os.path.exists(path):
                existing_df = pd.read_parquet(path)  # Removed parse_dates argument
                existing_df['time'] = pd.to_datetime(existing_df['time'])  # Convert 'time' to datetime
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['time']).sort_values('time')
            else:
                combined_df = new_df
            
            self.dataframes.append(combined_df)
            combined_df.to_parquet(path, index=False)

    async def get_symbols_async(self, session: aiohttp.ClientSession) -> List[str]:
        async with session.get("https://api.bybit.com/v5/market/tickers?category=linear") as response:
            data = await response.json()
            return [res['symbol'] for res in data['result']['list']]

    async def get_symbols(self):
        os.makedirs(self.DIRECTORY, exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            # symbols = await self.get_symbols_async(session)
            symbols = self.symbols
            # symbols = ["ALPHAUSDT"]
            # symbs = symbols
            # print(symbols[:200])
            tasks = []
            for i in range(0, len(symbols), self.BATCH_SIZE):
                batch = symbols[i:i+self.BATCH_SIZE]
                batch_tasks = [self.process_symbol(session, symbol, self.start_time, self.end_time) for symbol in batch]
                tasks.extend(batch_tasks)
            
            await tqdm_asyncio.gather(*tasks, desc="Processing symbols")

        combined_df = self.processDFS(self)
        print(combined_df)
        return combined_df