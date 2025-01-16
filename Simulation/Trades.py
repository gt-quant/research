class Trades():
    def __init__(self):
        # side, size, price 
        self.last_trade = (None, None, None)

    def update(self, json_data):
        # print(json_data)
        # input()
        self.last_trade = (
            json_data['side'],
            json_data['size'],
            json_data['price'],
        )
        
if __name__ == '__main__':
    pass
