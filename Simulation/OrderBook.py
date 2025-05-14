from sortedcontainers import SortedDict

class OrderBook():
    def __init__(self):
        self.asks = SortedDict() 
        self.bids = SortedDict()
        # def ggg(x):
        #     print(x)
        #     return -1.0 * x
        # self.bids = SortedDict(ewewe=lambda x: x)
        self.last_time = 0
        self._has_new_mid = False
        self.mid = None
    
    def print(self):
        print("ASKS:", list(self.asks.items())[:10])
        print("BIDS:", list(self.bids.items())[-1:-10:-1])

    def process_snapshot(self, snapshot):
        """
        Processes a snapshot message to reset the order book.
        """
        self.bids.clear()
        self.asks.clear()
        for price, size in snapshot['b']:
            self.bids[float(price)] = float(size)
        for price, size in snapshot['a']:
            self.asks[float(price)] = float(size)

    def process_delta(self, delta):
        """
        Processes a delta message to update the order book.
        """
        for price, size in delta['b']:
            price = float(price)
            size = float(size)
            if size == 0:
                if price in self.bids:
                    del self.bids[price]
            else:
                self.bids[price] = size
        for price, size in delta['a']:
            price = float(price)
            size = float(size)
            if size == 0:
                if price in self.asks:
                    del self.asks[price]
            else:
                self.asks[price] = size
    
    def update(self, json_data):
        self.last_time = json_data["ts"]
        if json_data["type"] == "snapshot":
            self.process_snapshot(json_data["data"])
        elif json_data["type"] == "delta":
            self.process_delta(json_data["data"])
        else:
            raise Exception("Unknown json_data type")

        new_mid = self.get_mid()
        if new_mid != self.mid:
            self._has_new_mid = True
            self.mid = new_mid
        else:
            self._has_new_mid = False
    
    def has_new_mid(self):
        return self._has_new_mid

    def get_best_bid(self):
        return self.bids.peekitem(-1) if self.bids else (None, None)
    
    def get_best_ask(self):
        return self.asks.peekitem(0) if self.asks else (None, None)
    
    def get_mid(self):
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()

        if best_bid is not None and best_ask is not None:
            # print(list(self.bids.items()))
            # print(best_bid, best_ask)
            return (best_bid + best_ask) / 2.0
        return None
    
    def get_size_price(self, size_level):
        # bid, ask
        bid_size_price = None
        bid_cum_size = 0
        for i in range(-1, -len(self.bids), -1):
            bid_size_price, size = self.bids.peekitem(i)
            bid_cum_size += size
            if bid_cum_size >= size_level:
                break
        
        ask_size_price = None
        ask_cum_size = 0
        for price, size in self.asks.items():
            # print("lllll", price, size)
            ask_size_price = price
            ask_cum_size += size
            if ask_cum_size >= size_level:
                break

        return bid_size_price, ask_size_price
    
    def get_bps_size(self, bps_list):
        mid = self.get_mid()
        if mid is None:
            return [None] * len(bps_list), [None] * len(bps_list)

        # bid, ask
        bid_size_list = []
        bid_price_list = []
        bid_cum_size = 0
        bps_list_ptr = 0
        old_price = mid
        for i in range(-1, -len(self.bids), -1):
            price, size = self.bids.peekitem(i)
            bps = (mid - price) / mid * 10000.0

            if bps > bps_list[bps_list_ptr]:
                bid_size_list.append(bid_cum_size)
                bid_price_list.append(old_price)
                if bps_list_ptr == len(bps_list) - 1:
                    break
                bps_list_ptr += 1
            bid_cum_size += size
            old_price = price
        
        ask_size_list = []
        ask_price_list = []
        ask_cum_size = 0        
        bps_list_ptr = 0
        old_price = mid
        for price, size in self.asks.items():
            bps = (price - mid) / mid * 10000.0
            if bps > bps_list[bps_list_ptr]:
                ask_size_list.append(ask_cum_size)
                ask_price_list.append(old_price)
                if bps_list_ptr == len(bps_list) - 1:
                    break
                bps_list_ptr += 1
            ask_cum_size += size
            old_price = price
        
        # self.print()
        # print(bid_size_list)
        # print(ask_size_list)

        return bid_size_list, ask_size_list, bid_price_list, ask_price_list
    
    def print_best_bid_ask(self):
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        print(f"Best Bid: {best_bid[0]} at size {best_bid[1]}")
        print(f"Best Ask: {best_ask[0]} at size {best_ask[1]}")

if __name__ == '__main__':
    # Example usage
    order_book = OrderBook()

    # Process a snapshot message
    snapshot = {
        'b': [
            ['50000.0', '1.0'],
            ['49900.0', '2.0'],
        ],
        'a': [
            ['50100.0', '1.5'],
            ['50200.0', '2.5'],
        ],
    }
    order_book.process_snapshot(snapshot)

    # Process a delta message
    delta = {
        'b': [
            ['50000.0', '0.5'],
        ],
        'a': [
            ['50100.0', '0.5'],
        ],
    }
    order_book.process_delta(delta)
    order_book.print()

    best_bid, best_ask = order_book.get_best_bid_ask()
    print(f"Best Bid: {best_bid[0]} at size {best_bid[1]}")
    print(f"Best Ask: {best_ask[0]} at size {best_ask[1]}")
