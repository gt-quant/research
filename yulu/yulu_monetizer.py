import pandas as pd
from tqdm import tqdm  # Import tqdm for progress tracking
from monetizer import Monetizer

def indicator(lob, upb):
    if pd.isna(lob) or pd.isna(upb):  # Handle NaN values
        return 0
    if (lob > 0 and upb > 0) or (lob < 0 and upb < 0):
        return 1
    else:
        return 0

class yulumonetizer(Monetizer):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def monetize(self, predictions, cutOff):
        """
        Monetize predictions into positions based on the objective function.
        
        Args:
            predictions (pd.DataFrame): DataFrame where rows are time, columns are symbol predictions,
                                        each entry is a tuple (predicted return, predicted lower bound,
                                        predicted upper bound, prediction std_error).
            cutOff (float): Threshold for absolute predicted return to filter positions.
        
        Returns:
            pd.DataFrame: DataFrame of positions with columns as `X_pos`.
        """
        # Correct column naming logic
        pos_columns = [col.split("_")[0] + "_pos" for col in predictions.columns]
        positions = pd.DataFrame(index=predictions.index, columns=pos_columns)

        # Add tqdm for tracking the progress through time indices
        for time in tqdm(predictions.index, desc="Processing Time Steps", unit="step"):
            # Extract the prediction data for this time
            data = predictions.loc[time]

            # Drop NaN entries
            data = data.dropna()

            if data.empty:
                # If all predictions are NaN, set all positions to 0
                positions.loc[time, :] = 0
                continue
            
            # Compute the objective function based on the mode
            if self.mode == "standard":
                # Standard mode: Objective is predicted return / std_error
                values = data.apply(lambda x: x[0] / x[3] if isinstance(x, tuple) and len(x) == 4 and all(pd.notna(i) for i in x) else 0)
            elif self.mode == "risk_minimized":
                # Risk-minimized mode: Objective is predicted return / predicted upper bound
                values = data.apply(
                    lambda x: indicator(x[1], x[2]) * x[0] / x[3]
                    if isinstance(x, tuple) and len(x) == 4 and all(pd.notna(i) for i in x) else 0
                )
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            # Filter based on absolute predicted return threshold
            filtered_values = values[values.abs() > cutOff]
            
            if filtered_values.empty:
                # If no predictions pass the filter, set all positions to 0
                positions.loc[time, :] = 0
                continue
            
            # Rank values, pick top positive and bottom negative
            positive_values = filtered_values[filtered_values > 0].nlargest(10)
            negative_values = filtered_values[filtered_values < 0].nsmallest(10)
            selected_values = pd.concat([positive_values, negative_values])
            
            # Assign positions as absolute percentages
            total_value = selected_values.abs().sum()
            if total_value > 0:
                # Assign the normalized weights to the corresponding "_pos" columns
                for symbol in selected_values.index:
                    pos_col = symbol.split("_")[0] + "_pos"
                    positions.loc[time, pos_col] = selected_values[symbol] / total_value
            
            # Set all non-selected positions to 0 for this time step
            positions.loc[time, :] = positions.loc[time, :].fillna(0).infer_objects(copy=False)
        return positions
