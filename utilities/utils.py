import numpy as np

def parse_time(str):
    if str[-1] == "M":
        return 1 * int(str[0:-1])
    elif str[-1] == "H":
        return 60 * int(str[0:-1])
    elif str[-1] == "D":
        return 1440 * int(str[0:-1])
    elif str[-1] == "W":
        return 7 * 1440 * int(str[0:-1])
    else:
        raise Exception("Unknown Timeframe")
