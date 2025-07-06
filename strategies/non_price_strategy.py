"""
non_price_strategy.py

A sample implementation of a basic non-price-based trading strategy.
This strategy uses on-chain metrics (e.g., active addresses) instead of price movements
to generate trading signals and run backtests.

:precondition: Data must include 'timestamp', 'price', and a non-price metric
:postcondition: Prepares signal data and evaluates strategy using the BacktestEngine
"""

import pandas as pd
import numpy as np
from core.engine import BacktestEngine
from typing import Dict


class NonPriceStrategy(BacktestEngine):
    def __init__(self, param_method: str = 'pct_change', initial_capital: float = 10000):
        """
        Initialize the non-price strategy.

        :param param_method: Method to calculate the metric change ('pct_change' or 'absolute')
        :param initial_capital: Starting capital for backtesting
        :return: None
        """
        super().__init__(initial_capital)
        self.param_method = param_method

    def generate_signal(self, row: pd.Series, params: Dict) -> int:
        """
        Generate trading signal for a given row.

        :param row: A row of merged time-series data
        :param params: Dictionary of strategy parameters
        :return: 1 for long, -1 for short, 0 for no action
        """
        # Ensure required parameters are present
        required = ['rolling_window', 'long_threshold', 'short_threshold']
        if any(p not in params for p in required):
            raise ValueError(f"Missing required parameters: {required}")

        # Extract rolling return
        rolling_return = row.get('rolling_return', 0)

        # Signal logic
        if rolling_return > params['long_threshold']:
            return 1
        elif rolling_return < params['short_threshold']:
            return -1
        return 0

    def prepare_data(self, price_data: pd.DataFrame, param_data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        ...
        window = params.get('rolling_window', 20)

        """
        Merge and preprocess price and non-price data for backtesting.

        :param price_data: DataFrame containing price with columns ['timestamp', 'value']
        :param param_data: DataFrame containing a non-price metric with columns ['timestamp', 'value']
        :return: DataFrame with merged and transformed columns including 'rolling_return'
        """
        df = pd.merge(
            price_data.rename(columns={'value': 'price'}),
            param_data.rename(columns={'value': 'param'}),
            on='timestamp',
            how='inner'
        )

        # Compute changes in the non-price metric
        df['param_pct_change'] = df['param'].pct_change()
        df['param_absolute_change'] = df['param'] - df['param'].shift(1)

        # Choose the rolling return method
        if self.param_method == 'pct_change':
            df['rolling_return'] = df['param_pct_change'].rolling(
                window=params.get('rolling_window', 20)
            ).mean()
        else:  # absolute
            df['rolling_return'] = df['param_absolute_change'].rolling(
                window=params.get('rolling_window', 20)
            ).mean()

        return df.dropna()
