from features.edges import EDGES
import features
import pandas as pd

# Not done yet, it should use EDGES to populate all parent features
def add_feature_col_inplace(df, feature_name):
    symbol, feature, param_str = feature_name.split('__')
    params = param_str.split('_')

    feature_function = getattr(features, feature)
    df[feature_name] = feature_function(df, symbol, *params)

df = pd.DataFrame({
    'ETHUSDT__open_price': [100, 105, 102, 110, 108]
})

add_feature_col_inplace(df, "ETHUSDT__log_return__1M")
print(df)
