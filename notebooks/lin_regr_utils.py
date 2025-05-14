import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
import scipy.stats as stats

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.metrics import mean_squared_error, r2_score

def get_price_df(days=3):
    all_df = []
    for i in [-2,-1, 0][:days]:
        df = pd.read_csv(f'data/prices_{i}.csv', sep=';')
        all_df.append(df)
    df = pd.concat(all_df, axis=0,  ignore_index=True)
    df['time'] = df['day'] * 1000000 + df['timestamp'] + 2_000_000
    
    return df

def show_heat_map(df):
    corr_matrix = df.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(20, 16))  # Set the figure size
    
    # Define a custom colormap using LinearSegmentedColormap
    colors = ["red", "white", "blue"]  # Red for negative, white for zero, blue for positive
    cdict = {
        'red':   [(0.0, 1.0, 1.0),   # negative values -> red
                  (0.5, 1.0, 1.0),   # 0 (white)
                  (1.0, 0.0, 0.0)],  # positive values -> blue
        'green': [(0.0, 0.0, 0.0),   # no green for negative
                  (0.5, 1.0, 1.0),   # 0 (white)
                  (1.0, 0.0, 0.0)],  # no green for positive
        'blue':  [(0.0, 0.0, 0.0),   # no blue for negative
                  (0.5, 1.0, 1.0),   # 0 (white)
                  (1.0, 1.0, 1.0)]   # positive values -> blue
    }

    # Create the LinearSegmentedColormap
    cmap = LinearSegmentedColormap("RedWhiteBlue", cdict)
    
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f', center=0, cbar=True)

    # Display the heatmap
    plt.title('Correlation Heatmap')
    plt.show()

# def lin_regr(df, y_col):

#     df = df.dropna()

#     X = df[[x for x in df.columns if x!= y_col]]  # 2D array for sklearn
#     Y = df[y_col]    # 1D array

#     # Initialize the model
#     model = LinearRegression()

#     # Fit the model
#     model.fit(X, Y)

#     # Get results
#     print(f"Coefficient (slope): {model.coef_[0]}")
#     print(f"Intercept: {model.intercept_}")

#     # Make predictions
#     Y_pred = model.predict(X)
#     print(f"Predicted Y: {Y_pred}")

#     # Plot the results
#     plt.scatter(X, Y, color='blue', label='Actual data')
#     plt.plot(X, Y_pred, color='red', label='Regression line')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Linear Regression')
#     plt.legend()
#     plt.show()

def lin_regr(df, y_col):
    # Drop rows with NaN values
    df = df.dropna()

    # Define independent (X) and dependent (Y) variables
    X = df[[x for x in df.columns if x != y_col]]  # All columns except the dependent variable
    Y = df[y_col]  # Dependent variable (Y)

    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X, Y)

    # Get results
    print(f"Coefficients (slope): {model.coef_}")
    print(f"Intercept: {model.intercept_:.7f}")

    # Make predictions
    Y_pred = model.predict(X)
    # print(f"Predicted Y: {Y_pred[:5]}")  # Show first 5 predictions for brevity

    # Plotting for the first feature (just to visualize the relationship)
    # If you want to plot for multiple features, you can create a pair plot or use a 3D plot
    if X.shape[1] == 1:  # If only one feature, we can plot the regression line
        plt.scatter(X.iloc[:, 0], Y, color='blue', label='Actual data')
        plt.plot(X.iloc[:, 0], Y_pred, color='red', label='Regression line')
        plt.xlabel(X.columns[0])
        plt.ylabel(y_col)
        plt.title(f'Linear Regression with {X.columns[0]}')
        plt.legend()
        plt.show()
    else:
        pass
        # print("Regression model fitted with multiple variables. Visualization for multiple features is not supported in 2D.")
    
    # Optionally, you can show a DataFrame of the features and their corresponding coefficients
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
    print("\nFeature Coefficients:")
    print(coeff_df)

    return Y_pred, Y, X, df

def linearity_test(df, x_cols, y):
    for col in x_cols:
        sns.scatterplot(x=df[col], y=y)
        # plt.title(f'{col} vs {y_col}')
        plt.xlabel(col)
        # plt.ylabel(y_col)
        plt.show()

def residual_independence_test(Y, Y_pred):
    # Calculate residuals
    residuals = Y - Y_pred

    # Plot residuals vs. fitted values
    sns.scatterplot(x=Y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

def homoscedasticity_test(Y, Y_pred):
    # Plot residuals vs fitted values
    residuals = Y - Y_pred
    plt.scatter(Y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

def residual_normality_test(Y, Y_pred):
    residuals = Y - Y_pred
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.show()

def multicollinearity_test(X):
    X_with_const = add_constant(X)

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_with_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

    print(vif_data)

def get_stats(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Print error metrics
    print(f"Mean Squared Error (MSE): {mse:.7f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.7f}")
    print(f"R-squared (R²): {r2:.7f}")
