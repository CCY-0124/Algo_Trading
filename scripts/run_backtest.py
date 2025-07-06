"""
run_backtest.py

This script performs a backtest using a non-price strategy based on Glassnode on-chain data.
It fetches price and address activity metrics, optimizes strategy parameters, and reports performance metrics.

:precondition: Requires Glassnode API key and proper DataLoader/Strategy modules
:postcondition: Executes backtest and logs results to console
"""
from core.dataloader import DataLoader
from strategies.non_price_strategy import NonPriceStrategy
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_backtest(asset: str = 'BTC', param_method: str = 'pct_change'):
    """
    Run backtest on a given crypto asset using a non-price indicator strategy.

    :param asset: Asset symbol, e.g., 'BTC'
    :param param_method: Method to calculate signal input ('pct_change' or 'absolute')
    :precondition: DataLoader and strategy classes must be properly implemented
    :postcondition: Logs backtest results, including performance metrics and best parameters
    :return: Dictionary of results from the best-performing parameter set
    """
    # Initialize DataLoader - prefers local database
    loader = DataLoader(api_key="YOUR_GLASSNODE_API_KEY", use_database=True)

    # Set date range: last 90 days ending 2 days ago
    end_date = datetime.now() - timedelta(days=2)  # 2天前
    start_date = end_date - timedelta(days=90)  # 90天前

    logging.info(f"Fetching data for {asset} from {start_date.date()} to {end_date.date()}")

    try:
        # Fetch price data
        price_data = loader.get_data(
            asset=asset,
            metric='market/price_usd_close',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

        # Fetch non-price metric: active addresses
        param_data = loader.get_data(
            asset=asset,
            metric='addresses/active_count',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

        # Initialize strategy
        strategy = NonPriceStrategy(param_method=param_method)

        # Prepare merged dataset for backtesting
        data = strategy.prepare_data(price_data, param_data)

        if len(data) < 30:
            raise ValueError("Insufficient data for backtesting")

        # Define parameter grid for optimization
        param_ranges = {
            'rolling_window': [5, 10, 20, 30, 50],
            'long_threshold': np.linspace(0.01, 0.1, 5).tolist(),
            'short_threshold': np.linspace(-0.1, -0.01, 5).tolist(),
            'lot_size': [0.001, 0.005, 0.01]
        }

        logging.info("Starting parameter optimization...")

        # Run parameter optimization
        best_result = strategy.optimize(data, param_ranges)

        # Output results
        logging.info("\n=== OPTIMIZATION RESULTS ===")
        logging.info(f"Best parameters: {best_result['params']}")
        logging.info(f"Total return: {best_result['total_return'] * 100:.2f}%")
        logging.info(f"Annual return: {best_result['annual_return'] * 100:.2f}%")
        logging.info(f"Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
        logging.info(f"Max drawdown: {best_result['max_drawdown'] * 100:.2f}%")
        logging.info(f"Win rate: {best_result['win_rate'] * 100:.2f}%")
        logging.info(f"Number of trades: {best_result['num_trades']}")

        return best_result

    except Exception as e:
        logging.error(f"Backtest failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example: Run backtest for BTC
    run_backtest(asset='BTC')

    # Example: Run backtest for ETH
    # run_backtest(asset='ETH')

    # Example: Use absolute change method
    # run_backtest(asset='SOL', param_method='absolute')