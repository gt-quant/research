from Simulation import Simulator, OrderBook, get_filename, DataLogger, get_random_suffix
from datetime import datetime, timedelta

DIR = 'data'
# date = "20250308"

symbols = [
    'HYPEUSDT', 
    'HYPEUSDT'
]
ob_symbols = symbols[:3]
categories = [
    'OB500', 
    'trades'
]
suffix = get_random_suffix()

curr_date = datetime.strptime("20250213", "%Y%m%d")
for i in range(30):
    date = curr_date.strftime("%Y%m%d")
    print("Date:", date)

    simulator = Simulator(DIR, date, symbols, categories, ob_symbols)

    filename = get_filename(date, symbols, categories, suffix)

    dataLogger = DataLogger(
        f'simulation_data/{filename}.csv',
        "time,mid," + 
        "bidsize_1," + "bidsize_2," + "bidsize_3," + "bidsize_4," + "bidsize_5," +
        "asksize_1," + "asksize_2," + "asksize_3," + "asksize_4," + "asksize_5," +
        "bidprice_1," + "bidprice_2," + "bidprice_3," + "bidprice_4," + "bidprice_5," +
        "askprice_1," + "askprice_2," + "askprice_3," + "askprice_4," + "askprice_5," +
        "trade_side,trade_size,trade_price\n"
    )
    def process(state, is_trade):
        ob1 = state.orderbooks['HYPEUSDT']
        if is_trade:
            dataLogger.add_line([
                state.time,
                None, 
                None, None, None, None, None, 
                None, None, None, None, None, 
                None, None, None, None, None, 
                None, None, None, None, None, 
                *state.trades.last_trade
            ])
        else:
            bid_size_list, ask_size_list, bid_price_list, ask_price_list = ob1.get_bps_size([1, 2, 3, 4, 5])
            dataLogger.add_line([
                state.time,
                ob1.get_mid(),
                *bid_size_list,
                *ask_size_list,
                *bid_price_list,
                *ask_price_list,
                None, None, None
            ])

    simulator.simulate(process)
    dataLogger.close()

    curr_date += timedelta(days=1)
                  