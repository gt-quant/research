import numpy as np
import pandas as pd
import json
import sys
import os

def read_book(path, dt = 1000):
    bids = {}
    asks = {}

    max_bid = -1
    min_ask = np.inf

    with open("output.txt", "w") as output_file:
        output_file.write(f"Date,Value,Lowest Ask Volume,Highest Bid Volume")
        for date in sorted(os.listdir(path)):
            with open(f"historical_orderbook/{perp}/{date}") as file:
                start_time = None
                dif = np.nan

                for i, line in enumerate(file):
                    try:
                        json_data = json.loads(line)
                    except json.JSONDecodeError:
                        print("Invalid JSON encountered and skipped.")
                        continue

                    if json_data["type"] == "snapshot":
                        snapshot = json_data["data"]
                        if i == 0:
                            date_time = pd.to_datetime(date[:-5])
                            start_time = json_data["ts"]
                        bids = {}
                        asks = {}
                        max_bid = -1
                        min_ask = np.inf
                        dif = float(snapshot["b"][0][0]) - float(snapshot["b"][1][0])
                        for order in snapshot["b"]:
                            bids[bid_p := float(order[0])] = int(order[1])
                            max_bid = max(max_bid, bid_p)
                        for order in snapshot["a"]:
                            asks[ask_p := float(order[0])] = int(order[1])
                            min_ask = min(min_ask, ask_p)
                    
                    elif json_data["type"] == "delta":
                        row = json_data["data"]

                        for order in row["b"]:
                            bid, vol = float(order[0]), int(order[1])

                            while vol > 0:
                                if bid >= min_ask:
                                    if vol > asks[min_ask]:
                                        vol -= asks[min_ask]
                                        del asks[min_ask]
                                        min_ask = min(asks.keys())
                                    else:
                                        asks[min_ask] -= vol
                                        break

                                else:
                                    if bid in bids:
                                        bids[bid] += vol
                                    else:
                                        bids[bid] = vol
                                        max_bid = max(max_bid, bid)
                                    break
                        
                        for order in row["a"]:
                            ask, vol = float(order[0]), int(order[1])

                            while vol > 0:
                                if ask <= max_bid:
                                    if vol > bids[max_bid]:
                                        vol -= bids[max_bid]
                                        del bids[max_bid]
                                        max_bid = max(bids.keys())
                                    else:
                                        bids[max_bid] -= vol
                                        break
                                else:
                                    if ask in asks:
                                        asks[ask] += vol
                                    else:
                                        asks[ask] = vol
                                        min_ask = min(min_ask, ask)
                                    break
                    
                    if i % dt == 0:
                        actual_time = date_time + pd.Timedelta(milliseconds=json_data["ts"] - start_time)
                        # Calculate output statistics

                        output_line = f"{actual_time}, {(max_bid+min_ask)/2},{asks[min_ask]}, {bids[max_bid]}\n"
                        # print(output_line.strip())
                        output_file.write(output_line)       


if __name__ == "__main__":
    for perp in os.listdir("historical_orderbook/"):
        read_book(f"historical_orderbook/{perp}")        


#%%
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("output.txt")
plt.figure()
plt.plot(df.iloc[:, 1])
plt.tight_layout()
plt.show()