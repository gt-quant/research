from Simulation import Simulator, OrderBook, get_filename, DataLogger, get_random_suffix
from datetime import datetime, timedelta

DIR = 'data'
# date = "20250308"

symbols = [
    'LINKUSDT',
    'ETHUSDT'
    
]
ob_symbols = symbols[:1]
categories = [ 
    'OB500',
    'trades'
]
suffix = get_random_suffix()

curr_date = datetime.strptime("20250314", "%Y%m%d")
for i in range(1):
    date = curr_date.strftime("%Y%m%d")
    print("Date:", date)

    simulator = Simulator(DIR, date, symbols, categories, ob_symbols)

    filename = get_filename(date, symbols, categories, suffix)

    dataLogger = DataLogger(
        f'simulation_data/{filename}.csv',
        "time,mid," + 
        "trade_side,trade_size,trade_price\n"
    )
    def process(state, is_trade):
        ob1 = state.orderbooks['LINKUSDT']
        if is_trade:
            dataLogger.add_line([
                state.time,
                None, 
                *state.trades.last_trade
            ])
        else:
            dataLogger.add_line([
                state.time,
                ob1.get_mid(),
                None, None, None
            ])

    simulator.simulate(process)
    dataLogger.close()

    curr_date += timedelta(days=1)
                  