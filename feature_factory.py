from features.edges import EDGES
import features
import pandas as pd

def _get_feature_col(df, feature_name):

    if feature_name in df.columns:
        return

    if len(feature_name.split('__')) == 3:
        symbol, feature, param_str = feature_name.split('__')
        params = param_str.split('_')
    elif len(feature_name.split('__')) == 2:
        symbol, feature = feature_name.split('__')
        params = []
    else:
        raise Exception("Feature Name Error.")

    if not feature in EDGES:
        raise Exception(f"Unregistered feature or Input not loaded: {feature}")

    for parent in EDGES[feature]:
        parent_feature_name = symbol + '__' + parent
        if not parent_feature_name in df.columns:
            add_feature_col_inplace(df, parent_feature_name)

    feature_function = getattr(features, feature)
    return feature_function(df, symbol, *params)

def add_feature_col_inplace(df, feature_name):
    df[feature_name] = _get_feature_col(df, feature_name)

def get_feature_col(df, feature_name):
    return _get_feature_col(df, feature_name)
    

if __name__ == "__main__":
    df = pd.DataFrame({
        'ETHUSDT__open_price': [100, 105, 102, 110, 108],
        'ETHUSDT__close_price': [100, 105, 102, 110, 108]
    })

    add_feature_col_inplace(df, "ETHUSDT__gio_feature")
    print(df)
