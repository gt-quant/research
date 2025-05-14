
def get_random_suffix():
    import random
    import string
    return''.join(random.choices(string.ascii_uppercase, k=2))

def get_filename(date, symbols, categories, suffix):
    if suffix is None:
        suffix = get_random_suffix()
    from datetime import datetime
    today_date = datetime.today().strftime('%Y-%m-%d')
    return f"output_{today_date}_{date}_" + \
        "_".join([x + y for x, y in zip(symbols, categories)]) + \
        "_" + suffix

def get_short_filename(date, symbols, categories, suffix):
    if suffix is None:
        suffix = get_random_suffix()
    from datetime import datetime
    return f"output_{date}_" + \
        "_".join([x + y for x, y in zip(symbols, categories)]) + \
        "_" + suffix
