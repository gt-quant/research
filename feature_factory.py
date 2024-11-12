# from features.edges import EDGES
import features
import pandas as pd

def _get_feature_col(df, feature_name):

    if feature_name in df.columns:
        return

    feature, param_str = feature_name.split('__')
    params = param_str.split('_')

    # if len(feature_name.split('__')) == 3:
    #     symbol, feature, param_str = feature_name.split('__')
    #     params = param_str.split('_')
    # elif len(feature_name.split('__')) == 2:
    #     symbol, feature = feature_name.split('__')
    #     params = []
    # else:
    #     raise Exception("Feature Name Error.")

    # if not feature in EDGES:
    #     raise Exception(f"Unregistered feature or Input not loaded: {feature}")

    # for parent in EDGES[feature]:
    #     parent_feature_name = symbol + '__' + parent
    #     if not parent_feature_name in df.columns:
    #         add_feature_col_inplace(df, parent_feature_name)

    feature_class = getattr(features, feature)
    feature_obj = feature_class(*params)

    parents = feature_obj.get_parents()
    
    for parent in parents:
        if not parent in df.columns:
            add_feature_col_inplace(df, parent)

    return feature_obj.get_feature(df)

def add_feature_col_inplace(df, feature_name):
    df[feature_name] = _get_feature_col(df, feature_name)

def get_feature_col(df, feature_name):
    return _get_feature_col(df, feature_name)
    

if __name__ == "__main__":
    df = pd.DataFrame({
        'OpenPrice__ETHUSDT': [100, 105, 102, 110, 108],
        'OpenPrice__BTCUSDT': [1,2,3,4,5],
    })

    add_feature_col_inplace(df, "Statarb__ETHUSDT_BTCUSDT_1_1_1")
    print(df)