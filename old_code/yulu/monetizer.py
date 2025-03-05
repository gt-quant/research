from abc import ABC, abstractmethod
import pandas as pd

class Monetizer(ABC):
    """
    Superclass for all monetizers.
    """
    def __init__(self):
        """
        Initialize the monetizer
        """
        self.mode = set()

    @abstractmethod
    def monetize(self, signal_df):
        """
        Abstract method to process signals into positions. Must be implemented by subclasses.
        :param signal_df: DataFrame with trading signals (columns in format X_signal).
        :return: DataFrame with positions (columns in format X_positions).
        """
        pass