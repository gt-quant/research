import numpy as np
import pandas as pd
import json
import sys
import os
import csv
from datetime import datetime

from OrderBook import OrderBook
from Trades import Trades
from tqdm import tqdm

import heapq

class Listener:
    def __init__(self, DIR, symbol, date, category):
        self.file_path = f"{DIR}/{category}/{symbol}/{date}.data"
        self.symbol = symbol
        self.category = category

    def packets_iter(self):
        if self.category in ['OB1', 'OB50']:
            with open(self.file_path, 'r') as file:
                for line in file:
                    try:
                        ob = json.loads(line)
                        ob['ts'] = ob['ts'] / 1000.0
                        ob['symbol'] = self.symbol
                        ob['category'] = self.category
                        # ob['data']['s'] = self.symbol
                        # print(ob)
                        yield ob
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
        elif self.category == 'trades':
            with open(self.file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    row['ts'] = float(row.pop('timestamp'))
                    row['symbol'] = self.symbol
                    row['category'] = self.category
                    yield row
    
class EventLoop:
    def __init__(self, DIR, date, symbols, categories):
        self.listeners = [
            self._wrap_with_key(Listener(DIR, symbol, date, category).packets_iter())
            for symbol, category in zip(symbols, categories)
        ]
        pass
    
    # def packets_iter(self):
    #     return heapq.merge(*self.listeners) # ! TODO: comp for packets
    
    def _wrap_with_key(self, iterator):
        """
        Wraps each packet from the iterator with a key for sorting in heapq.merge().
        Assume the packets contain a 'timestamp' field used for ordering.
        """
        priority = {
            'OB1': 2, 
            'OB50': 1, 
            'trades': 0
        }
        for packet in iterator:
            yield (packet['ts'], priority[packet['category']], packet['symbol'], packet)

    def packets_iter(self):
        # Merge iterators using the timestamp as the comparison key
        for _, _, _, packet in heapq.merge(*self.listeners):
            # print(packet)
            yield packet

class Simulator:
    def __init__(self, DIR, date, symbols, categories, ob_symbols):
        self.eventLoop = EventLoop(DIR, date, symbols, categories)
        self.state = State(ob_symbols)
    
    def simulate(self, processFunc):
        cnt = 0
        # prev_cat = None
        for packet in tqdm(self.eventLoop.packets_iter()):
            # cnt += 1
            symbol = packet['symbol']
            # print(symbol, packet['category'])
            should_process = True
            is_trade = False
            self.state.time = packet['ts']
            if packet['category'] in ['OB1']:
                # timestamp = datetime.utcfromtimestamp(packet['ts']).strftime('%m-%d %H:%M:%S.%f')
                # print( "|", timestamp)
                self.state.orderbooks[symbol].update(packet)
                should_process = self.state.orderbooks[symbol].has_new_mid()

            elif packet['category'] in ['OB50']:
                self.state.orderbooks[symbol].update(packet)

            elif packet['category'] == 'trades':
                self.state.trades.update(packet)
                # timestamp = datetime.utcfromtimestamp(packet['ts']).strftime('%m-%d %H:%M:%S.%f')

                # print(packet['side'], packet['size'], packet['price'], "|", timestamp)
            # if prev_cat != packet['category']:
            #     prev_cat = packet['category']
                # input()
                is_trade = True
                pass

            if should_process:
                processFunc(self.state, is_trade)
                cnt += 1
            # if cnt > 1000:
            #     break
        
        print("Total:", cnt)

class State:
    def __init__(self, symbols):
        self.orderbooks = {symbol: OrderBook() for symbol in symbols}
        self.trades = Trades()
        self.time = 0.0
        pass
    
    def print(self):
        print("---------------------------")
        for symbol, orderbook in self.orderbooks.items():
            print(symbol, orderbook.last_time)
            orderbook.print_best_bid_ask()
            print("")


class DataLogger:
    def __init__(self, filename, header_line):
        self.output_file = open(filename, "w")
        self.output_file.write(header_line)
        # print(dir(self.output_file))
    
    def add_line(self, outputs):
        outputs = ["" if x is None else f"{x}" for x in outputs]
        line = ','.join(outputs)+"\n"
        self.output_file.write(line)
    
    def close(self):
        self.output_file.close()

# 1. trades Data Structure
#     single trade
#     trades within one sec queue combined stats: 50usdt price, midean size price..., combined size
# 2. DataLogger
# 3. Start doing research

if __name__ == "__main__":
    # DIR = 'gio/historic_bybit/historical_orderbook'
    DIR = 'data'
    date = "20250111"
    symbols = [
        'BTCUSDT', 
        'ETHUSDT', 
        'ARCUSDT', 
        'ARCUSDT'
    ]
    ob_symbols = symbols[:3]
    categories = [
        'OB1', 
        'OB1', 
        'OB50', 
        'trades'
    ]
    simulator = Simulator(DIR, date, symbols, categories, ob_symbols)

    dataLogger = DataLogger(
        f'output_{date}.csv',
        "time,btc_mid,eth_mid,arc_mid,arc_bp50,arc_ap50,arc_trade_side,arc_trade_size,arc_trade_price\n"
    )
    def process(state, is_trade):
        # state.print()
        ob_btc = state.orderbooks['BTCUSDT']
        ob_eth = state.orderbooks['ETHUSDT']
        ob_arc = state.orderbooks['ARCUSDT']

        if is_trade:
            dataLogger.add_line([
                state.time,
                None, None, None, None, None,
                *state.trades.last_trade
            ])
        else:
            arc_bp50, arc_ap50 = ob_arc.get_size_price(50.0)
            dataLogger.add_line([
                state.time,
                ob_btc.get_mid(),
                ob_eth.get_mid(),
                ob_arc.get_mid(),
                arc_bp50,
                arc_ap50,
                None, None, None
            ])
                        
    simulator.simulate(process)
    dataLogger.close()
