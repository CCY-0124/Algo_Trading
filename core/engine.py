"""
engine.py

Defines a base backtesting engine for simulating and optimizing trading strategies using historical data.

:precondition: Input data must contain at least 'price' and 'timestamp' columns.
:postcondition: Provides performance metrics and trade history for backtesting results.
"""
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Union
import math
from abc import ABC, abstractmethod


class BacktestEngine(ABC):
    """
    Abstract base class for a backtesting engine.

    :param initial_capital: Starting capital for the backtest
    :precondition: Subclasses must implement the `generate_signal` method
    :postcondition: Allows execution and optimization of trading strategies
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Initialize the backtest engine with a specified initial capital.

        :param initial_capital: Initial portfolio value to start the backtest with
        :precondition: The value should be a positive float
        :postcondition: Sets up initial capital and configures logging
        :return: None
        """
        self.initial_capital = initial_capital
        self._setup_logging()

    def _setup_logging(self):
        """
        Configure the logging format for the backtest engine.

        :precondition: Logging must be configured before use
        :postcondition: Logs will output with timestamps and severity levels
        :return: None
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @abstractmethod
    def generate_signal(self, row: pd.Series, params: Dict) -> int:
        """
        Abstract method to generate a trading signal from a single row of data.

        :param row: A single row of market data
        :param params: Strategy-specific parameters
        :precondition: Must be implemented in a subclass
        :postcondition: Returns a signal for each row
        :return: 1 for long, -1 for short, 0 for hold
        """
        pass

    def run_backtest(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Run a backtest on the provided historical data.

        :param data: DataFrame containing historical data with 'timestamp' and 'price'
        :param params: Dictionary of strategy parameters
        :precondition: DataFrame must have 'timestamp' and 'price' columns
        :postcondition: Calculates equity curve, trade history, and performance metrics
        :return: Dictionary containing backtest results and metrics
        """
        # Initialize variables
        equity = self.initial_capital
        position = 0  # Current position size
        equity_curve = [equity]
        trades = []
        data = data.copy()

        # Generate signals
        data['signal'] = data.apply(
            lambda row: self.generate_signal(row, params),
            axis=1
        )

        # Core backtesting loop
        for idx in range(1, len(data)):
            row = data.iloc[idx]
            prev_row = data.iloc[idx - 1]

            # Calculate unrealized P&L if in a position
            if position != 0:
                pnl = position * (row['price'] - prev_row['price'])
                equity += pnl

            # Signal handling
            if row['signal'] == 1 and position == 0:  # 开多仓
                position = params.get('lot_size', 0.001)
                trades.append({
                    'timestamp': row['timestamp'],
                    'type': 'buy',
                    'price': row['price'],
                    'position': position
                })
            elif row['signal'] == -1 and position > 0:  # 平多仓
                trades.append({
                    'timestamp': row['timestamp'],
                    'type': 'sell',
                    'price': row['price'],
                    'position': position
                })
                position = 0

            equity_curve.append(equity)

        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, trades)
        return {
            **metrics,
            'equity_curve': equity_curve,
            'trades': trades,
            'params': params
        }

    def _calculate_metrics(self, equity_curve: List[float], trades: List[dict]) -> Dict:
        """
        Calculate performance metrics from equity and trade history.

        :param equity_curve: List of portfolio equity values over time
        :param trades: List of executed trades
        :precondition: Both equity curve and trades should be complete
        :postcondition: Calculates total return, Sharpe ratio, max drawdown, win rate
        :return: Dictionary of backtest performance metrics
        """
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = equity_curve[-1] / equity_curve[0] - 1

        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)
        else:
            sharpe_ratio = 0

        # Calculate maximum drawdown
        max_drawdown = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            'total_return': total_return,
            'annual_return': (1 + total_return) ** (365 / len(equity_curve)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': self._calculate_win_rate(trades)
        }

    def _calculate_win_rate(self, trades: List[dict]) -> float:
        """
        Calculate the win rate from trade history.

        :param trades: List of trade dictionaries
        :precondition: Trades must include matching buy and sell pairs
        :postcondition: Computes percentage of profitable trades
        :return: Win rate as a float between 0 and 1
        """
        if len(trades) < 2:
            return 0.0

        winning_trades = 0
        total_trades = 0

        # Iterate through trade history to calculate profit for each buy/sell pair
        for i in range(0, len(trades) - 1, 2):
            buy_trade = trades[i]
            sell_trade = trades[i + 1]

            if buy_trade['type'] == 'buy' and sell_trade['type'] == 'sell':
                profit = (sell_trade['price'] - buy_trade['price']) * buy_trade['position']
                if profit > 0:
                    winning_trades += 1
                total_trades += 1

        return winning_trades / total_trades if total_trades > 0 else 0.0

    def optimize(self, data: pd.DataFrame, param_ranges: Dict) -> Dict:
        """
        Find the best strategy parameters using Sharpe ratio as the objective.

        :param data: DataFrame of backtest data
        :param param_ranges: Dictionary mapping parameter names to value lists
        :precondition: All parameter combinations must be valid for `run_backtest`
        :postcondition: Identifies and returns the best performing parameter set
        :return: Dictionary of best backtest result and parameters
        """
        best_sharpe = -np.inf
        best_result = None

        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        param_combinations = self._generate_param_combinations(param_values)

        # Progress bar optimization
        for combination in tqdm(param_combinations, desc="Optimizing parameters"):
            params = dict(zip(param_names, combination))
            result = self.run_backtest(data, params)

            if result['sharpe_ratio'] > best_sharpe:
                best_sharpe = result['sharpe_ratio']
                best_result = result

        return best_result

    def _generate_param_combinations(self, values_list: List[List[Any]]) -> List[Tuple]:
        """
        Generate all possible parameter combinations from value lists.

        :param values_list: A list of lists, where each sublist contains values for one parameter
        :precondition: Each sublist must be non-empty
        :postcondition: Produces Cartesian product of all parameter values
        :return: A list of tuples, each representing a parameter combination
        """
        from itertools import product
        return list(product(*values_list))