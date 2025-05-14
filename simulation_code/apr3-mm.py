from Simulation import Simulator, OrderBook, get_filename, DataLogger, get_random_suffix
from datetime import datetime, timedelta

DIR = 'data'
# date = "20250308"

SYMBOL = 'NEARUSDT'

symbols = [
    SYMBOL,
    SYMBOL
]
ob_symbols = symbols[:1]
categories = [ 
    'OB500',
    'trades'
]
suffix = get_random_suffix()

curr_date = datetime.strptime("20250408", "%Y%m%d")
for i in range(1):
    date = curr_date.strftime("%Y%m%d")
    print("Date:", date)

    simulator = Simulator(DIR, date, symbols, categories, ob_symbols)

    filename = get_filename(date, symbols, categories, suffix)

    dataLogger = DataLogger(
        f'simulation_data/{filename}.csv',
        "time,bbid,bask," + 
        "bsize,asize," +
        "trade_side,trade_size,trade_price\n"
    )
    def process(state, is_trade):
        ob1 = state.orderbooks[SYMBOL]
        if is_trade:
            dataLogger.add_line([
                state.time,
                None, None,
                None, None,
                *state.trades.last_trade
            ])
        else:
            best_bid, bid_size = ob1.get_best_bid()
            best_ask, ask_size = ob1.get_best_ask()
            dataLogger.add_line([
                state.time,
                best_bid,
                best_ask,
                bid_size,
                ask_size,
                None, None, None
            ])

    simulator.simulate(process)
    dataLogger.close()

    curr_date += timedelta(days=1)
                  