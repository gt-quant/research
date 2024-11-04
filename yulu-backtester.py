from numba import jit
import numpy as np
from datetime import datetime

class backtester:
    def __init__(self, data_path, start_time, amount):
        # start_date: start date of the backtesting dataset
        self.data = data_path
        self.start_time = start_time
        self.amount = amount
        self.taker_fee = 0.00055
        self.maker_fee = 0.0002
        self.starting_capital = 10000 # in USD
        self.threshold = 0.4

    def get_index(self, time):
        # Todo: return the index in the matrix
        # time given in format (string date, string hour)
        start_datetime = datetime.strptime(f"{self.start_time[0]} {self.start_time[1]}", "%Y-%m-%d %H:%M:%S")
        given_datetime = datetime.strptime(f"{time[0]} {time[1]}", "%Y-%m-%d %H:%M:%S")
        time_difference = given_datetime - start_datetime
        hours_difference = time_difference.total_seconds() // 3600
        index = int(hours_difference)
        return index


    @jit(nopython=True)
    def run_sim(self, alpha, start_time, end_time, id_num, time="hour"):
        # Load data
        raw_data = np.load(self.data)
        start_ind = self.get_index(start_time)
        end_ind = self.get_index(end_time)

        if raw_data.ndim != 3 or raw_data.shape[2] != 2:
            raise ValueError("Expected a numpy array with shape (n, k, 2) where n=hours, k=coins, and last dim is (bid, ask)")
        
        data_test_back = raw_data[start_ind : end_ind + 1]

        num_hours = data_test_back.shape[0]      # Number of hours trading
        num_coins = data_test_back.shape[1]      # Number of coins

        # Initialize variables
        total_money = self.starting_capital  # Starting capital
        holding = np.zeros(num_coins)  # Current holdings
        pnl_array = np.zeros((num_hours, 5))  # Time, profit, return, position_value, transaction_cost

        for i in range(num_hours):
            # Current and next hour prices
            current_prices = data_test_back[i][:, :]    # bid, ask Prices at time i
            next_prices = data_test_back[i+1][:, :]     # bid, ask Prices at time i+1

            # Alpha values at current time
            alpha_values = alpha[i]

            # black box begins
            # input an alpha values, returns a holding array

            # Identify top 3 longs and top 3 shorts
            sorted_indices = np.argsort(alpha_values)
            top_short_indices = sorted_indices[:3]      # Bottom 3 alpha values (shorts)
            top_long_indices = sorted_indices[-3:]      # Top 3 alpha values (longs)

            top_long_alphas = alpha_values[top_long_indices]
            top_short_alphas = alpha_values[top_short_indices]

            considered_long_indices = []

            considered_long_alphas = []

            for idx, alpha in zip(top_long_indices, top_long_alphas):
                if abs(alpha) > 0.4:
                    considered_long_indices.append(idx)
                    considered_long_alphas.append(alpha)

            considered_short_indices = []
            considered_short_alphas = []
            for idx, alpha in zip(top_short_indices, top_short_alphas):
                if abs(alpha) > 0.4:
                    considered_short_indices.append(idx)
                    considered_short_alphas.append(alpha)  

            all_considered_indices = considered_long_indices + considered_short_indices
            all_considered_alphas = considered_long_alphas + considered_short_alphas
            
            # Calculate the absolute alphas and total absolute alpha value
            all_abs_alphas = [abs(alpha) for alpha in all_considered_alphas]
            total_abs_alpha = sum(all_abs_alphas)

            # Initialize new holdings array
            new_holding = np.zeros(num_coins, dtype=np.float32)

            if total_abs_alpha > 0:
                # Determine the sign for longs (+1) and shorts (-1)
                all_considered_signs = [1]*len(considered_long_indices) + [-1]*len(considered_short_indices)
                
                # Calculate holdings for each considered coin
                for idx, alpha, sign in zip(all_considered_indices, all_considered_alphas, all_considered_signs):
                    abs_alpha = abs(alpha)
                    # Allocate money proportionally based on absolute alpha value
                    allocated_money = (abs_alpha / total_abs_alpha) * total_money
                    # Calculate units (holdings), considering the sign for long/short
                    units = (allocated_money / current_prices[idx]) * sign
                    # Round down to 6 decimal places
                    units = np.floor(units * 1e6) / 1e6
                    new_holding[idx] = np.float32(units)
                
                # Profit calculation
                profit = np.sum(holding * (next_prices - current_prices))
                position_value = np.sum(np.abs(new_holding * current_prices))
                ret = profit / position_value if position_value != 0 else 0.0

                # Record PnL information
                pnl_array[i, 0] = data_test_back[i][0, 0]  # Time
                pnl_array[i, 1] = profit                   # Profit before transaction costs
                pnl_array[i, 2] = ret                      # Return before transaction costs
                pnl_array[i, 3] = position_value


                # Transaction cost calculation
                delta_holding = new_holding - holding
                delta_volume = np.abs(delta_holding * current_prices)
                transaction_cost = np.sum(delta_volume) * self.taker_fee  # Assuming taker fee for trades

                # Update holdings
                holding = new_holding.copy()

                # Record transaction cost
                pnl_array[i, 4] = transaction_cost

        # Compute cumulative returns and metrics
        net_profits = pnl_array[:, 1] - pnl_array[:, 4]  # Profit after transaction costs
        returns = net_profits / total_money
        cumulative_returns = np.cumsum(returns)

        # Sharpe Ratio calculation (annualized)
        hourly_return = returns
        avg_return = np.mean(hourly_return)
        std_return = np.std(hourly_return)
        sharpe_ratio = (avg_return / std_return) * np.sqrt(8760) if std_return != 0 else 0

        # Max Drawdown calculation
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns)

        # Metrics before transaction costs
        profits_before_tc = pnl_array[:, 1]
        returns_before_tc = profits_before_tc / initial_capital
        cumulative_returns_before_tc = np.cumsum(returns_before_tc)
        avg_return_before_tc = np.mean(returns_before_tc)
        std_return_before_tc = np.std(returns_before_tc)
        sharpe_ratio_before_tc = (avg_return_before_tc / std_return_before_tc) * np.sqrt(8760) if std_return_before_tc != 0 else 0
        running_max_before_tc = np.maximum.accumulate(cumulative_returns_before_tc)
        drawdowns_before_tc = running_max_before_tc - cumulative_returns_before_tc
        max_drawdown_before_tc = np.max(drawdowns_before_tc)

        # Prepare results
        results = {
            'pnl_array': pnl_array,
            'return': cumulative_returns,
            'return_before_tc': returns_before_tc,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_before_tc': sharpe_ratio_before_tc,
            'max_drawdown': max_drawdown,
            'max_drawdown_before_tc': max_drawdown_before_tc
        }

        np.savetxt(f"pnl_{id_num}.csv", pnl_array, delimiter=",", header="Time,Profit,Return,PositionValue,TransactionCost", comments='')

        return results