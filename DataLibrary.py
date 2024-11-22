class DataLibrary():

    def __init__(self, coins, DIRECTORY = 'data', BATCH_SIZE = 20, interval = 1):
        self.coins = coins
        self.DIRECTORY = DIRECTORY
        self.BATCH_SIZE = BATCH_SIZE
        self.interval = interval
        self.interval = interval