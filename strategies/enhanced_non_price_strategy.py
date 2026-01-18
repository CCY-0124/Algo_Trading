"""
enhanced_non_price_strategy.py

Enhanced non-price strategy using the new EnhancedBacktestEngine.
Demonstrates the improved features:
- Dynamic lot size calculation
- Position tracking with pos_opened and pnl_list
- Proper position closing and opening for direction changes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from core.enhanced_engine import EnhancedBacktestEngine
from utils.local_data_loader import LocalDataLoader
from config.trading_config import DEFAULT_INITIAL_CAPITAL


class EnhancedNonPriceStrategy(EnhancedBacktestEngine):
    """
    Enhanced non-price strategy with improved position management.
    
    Features:
    - Uses factor data for signal generation
    - Dynamic position sizing based on available capital
    - Proper position tracking and P&L management
    - Realistic trading simulation
    """
    
    def __init__(self, 
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 min_lot_size: float = 0.001,
                 api_key: str = None,
                 use_api: bool = False,
                 param_method: str = 'pct_change'):
        """
        Initialize enhanced strategy
        
        :param initial_capital: Starting capital
        :param min_lot_size: Minimum lot size for trading
        :param api_key: Glassnode API key for using API data (optional, will auto-load if None)
        :param use_api: Whether to use API data instead of local files
        """
        super().__init__(
            initial_capital=initial_capital,
            min_lot_size=min_lot_size
        )
        self.use_api = use_api
        self.param_method = param_method  # Store the parameter method
        
        # Auto-load API key if not provided and API is enabled
        if use_api and api_key is None:
            self.api_key = self._load_api_key()
        else:
            self.api_key = api_key
            
        self.data_loader = LocalDataLoader()
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key using the same method as simple_price_check.py
        Tries encrypted storage first, then environment variables
        """
        import os
        import sys
        
        # Add the project root to Python path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        api_keys = []
        
        # Try to load from encrypted storage first
        try:
            from config.secrets import get_api_key, get_api_keys
            CONFIG_AVAILABLE = True
        except ImportError:
            CONFIG_AVAILABLE = False
            print("Warning: Configuration modules not available, using fallback settings")
        
        if CONFIG_AVAILABLE:
            # Try encrypted storage first
            for service in ['glassnode']:
                for key_name in ['main', 'backup']:
                    key = get_api_key(service, key_name)
                    if key:
                        api_keys.append(key)
                        print(f"Loaded {service}_{key_name} API key")
        
        # Fallback to environment variables
        if not api_keys:
            env_keys = [
                os.getenv('GLASSNODE_API_KEY', ''),
                os.getenv('GLASSNODE_API_KEY_BACKUP', ''),
            ]
            api_keys = [key for key in env_keys if key]
            if api_keys:
                print(f"Loaded {len(api_keys)} API keys from environment variables")
        
        if not api_keys:
            print("ERROR: No API keys found")
            return None
        
        # Return the first available key
        return api_keys[0]
    
    def load_api_data(self, asset: str, factor_name: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Load data from Glassnode API (matching legacy algorithm data source)
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :return: Prepared DataFrame or None if failed
        """
        if not self.api_key:
            logging.error("API key not provided")
            return None
            
        try:
            import requests
            import datetime
            
            # Convert dates to unix timestamps (EXACT same as legacy algorithm)
            unix_time_start = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
            unix_time_end = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
            
            # API endpoints - use dynamic factor_name
            crypto_price_link = 'https://api.glassnode.com/v1/metrics/market/price_usd_close'
            
            # Map factor names to correct Glassnode endpoints
            factor_endpoint_map = {
                'sopr_account_based': 'indicators/sopr_account_based',
                'sopr': 'indicators/sopr',
                'nvt': 'indicators/nvt',
                'nvt_ratio': 'indicators/nvt_ratio',
                'mcap_sopr': 'indicators/mcap_sopr',
                'sopr_less_155': 'indicators/sopr_less_155',
                'sopr_more_155': 'indicators/sopr_more_155'
            }
            
            # Use mapped endpoint or fallback to original factor_name
            endpoint_path = factor_endpoint_map.get(factor_name, factor_name)
            crypto_parameter_link = f'https://api.glassnode.com/v1/metrics/{endpoint_path}'
            
            logging.info(f"Using API endpoint: {crypto_parameter_link}")
            
            # Get price data
            result_of_price = requests.get(crypto_price_link,
                                           params={'a': asset, 's': unix_time_start, 'u': unix_time_end, 'api_key': self.api_key})
            
            # Debug API response
            logging.info(f"Price API Status: {result_of_price.status_code}")
            logging.info(f"Price API Response: {result_of_price.text[:200]}...")
            
            if result_of_price.status_code != 200:
                logging.error(f"Price API failed with status {result_of_price.status_code}: {result_of_price.text}")
                return None
                
            try:
                df_crypto_price = pd.read_json(result_of_price.text, convert_dates=['t'])
            except Exception as e:
                logging.error(f"Failed to parse price JSON: {e}")
                logging.error(f"Price JSON content: {result_of_price.text}")
                return None
            
            # Get parameter data
            result_of_param = requests.get(crypto_parameter_link,
                                           params={'a': asset, 's': unix_time_start, 'u': unix_time_end, 'api_key': self.api_key})
            
            # Debug API response
            logging.info(f"Parameter API Status: {result_of_param.status_code}")
            logging.info(f"Parameter API Response: {result_of_param.text[:200]}...")
            
            if result_of_param.status_code != 200:
                logging.error(f"Parameter API failed with status {result_of_param.status_code}: {result_of_param.text}")
                return None
                
            try:
                df_crypto_parameter = pd.read_json(result_of_param.text, convert_dates=['t'])
            except Exception as e:
                logging.error(f"Failed to parse parameter JSON: {e}")
                logging.error(f"Parameter JSON content: {result_of_param.text}")
                return None
            
            # Rename columns (EXACT same as legacy algorithm)
            for i in range(len(df_crypto_price.columns)):
                col_name = df_crypto_price.columns[i]
                df_crypto_price = df_crypto_price.rename(columns={col_name: col_name + '_' + asset})
            
            # Extract the endpoint name from factor_name for column naming
            endpoint_name = factor_name.split('/')[-1] if '/' in factor_name else factor_name
            
            for i in range(len(df_crypto_parameter.columns)):
                col_name = df_crypto_parameter.columns[i]
                df_crypto_parameter = df_crypto_parameter.rename(columns={col_name: col_name + '_' + endpoint_name})
            
            # Merge data (EXACT same as legacy algorithm)
            df = pd.concat([df_crypto_price, df_crypto_parameter], axis=1)
            
            # Shift parameter data by 1 day (EXACT same as legacy algorithm)
            for i in range(len(df.columns)):
                col_name = df.columns[i]
                if endpoint_name in col_name:
                    df[col_name] = df[col_name].shift(1)
            
            # Clean and prepare data
            df = df.dropna()
            df = df.reset_index(drop=True)
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                f't_{asset}': 'timestamp',
                f'v_{asset}': 'price',
                f'v_{endpoint_name}': 'factor_value'
            })
            
            # Debug: Log data consistency (after renaming)
            logging.info(f"Loaded {len(df)} rows of data for {asset} - {factor_name}")
            logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logging.info(f"Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")
            logging.info(f"Factor range: {df['factor_value'].min():.8f} to {df['factor_value'].max():.8f}")
            
            # Add required columns
            df['last_price'] = df['price'].shift(1)
            df['price_pct_change'] = df['price'].pct_change()
            df['factor_last_value'] = df['factor_value'].shift(1)
            df['factor_pct_change'] = df['factor_value'].pct_change()
            
            # Calculate rolling return based on param_method
            if self.param_method == 'pct_change':
                # Percentage Change
                df['rolling_return'] = df['factor_pct_change'].rolling(window=1).mean()
            elif self.param_method == 'absolute':
                # Absolute Change
                df['factor_absolute_change'] = df['factor_value'].diff()
                df['rolling_return'] = df['factor_absolute_change'].rolling(window=1).mean()
            elif self.param_method == 'raw':
                # Raw Number
                df['rolling_return'] = df['factor_value'].rolling(window=1).mean()
            else:
                # Default to percentage change
                df['rolling_return'] = df['factor_pct_change'].rolling(window=1).mean()
            
            # Clean NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to load API data: {e}")
            return None
    
    def generate_signal(self, row: pd.Series, params: Dict) -> int:
        """
        Generate trading signal using EXACT legacy algorithm logic
        
        Correct logic (matching manual simulation):
        rolling_return > long_param (0.02) = LONG signal (1)
        rolling_return < short_param (-0.02) = SHORT signal (-1)
        
        :param row: Data row containing rolling_return
        :param params: Strategy parameters
        :return: 1 (long), -1 (short), 0 (hold)
        """
        # Get rolling return (calculated in prepare_data)
        if 'rolling_return' not in row or pd.isna(row['rolling_return']):
            return 0
        
        rolling_return = row['rolling_return']
        
        # Get thresholds from parameters (EXACT same names as legacy algorithm)
        long_param = params.get('long_param', 0.02)   # Default from legacy algorithm
        short_param = params.get('short_param', -0.02)  # Default from legacy algorithm
        
        # EXACT same logic as legacy algorithm:
        # rolling_return < long_param (0.02) = LONG signal (1)
        # rolling_return > short_param (-0.02) = SHORT signal (-1)
        
        if rolling_return < long_param:
            return 1  # Long signal
        elif rolling_return > short_param:
            return -1  # Short signal
        else:
            return 0  # Hold (neutral)
    
    def prepare_data(self, asset: str, factor_name: str, 
                    clean_method: str = 'interpolate', rolling: int = 20,
                    start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Prepare data for backtesting with original algo_trading_backtest.py logic
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param clean_method: Method to clean NaN values
        :param rolling: Rolling window size (default: 20)
        :param start_date: Start date for API data (YYYY-MM-DD)
        :param end_date: End date for API data (YYYY-MM-DD)
        :return: Prepared DataFrame or None if failed
        """
        try:
            # Use API data if requested
            if self.use_api and self.api_key and start_date and end_date:
                logging.info(f"Loading data from Glassnode API for {asset}")
                data = self.load_api_data(asset, factor_name, start_date, end_date)
                if data is not None:
                    # Update rolling return with correct window size
                    if self.param_method == 'pct_change':
                        # Percentage Change
                        data['rolling_return'] = data['factor_pct_change'].rolling(window=rolling).mean()
                    elif self.param_method == 'absolute':
                        # Absolute Change
                        data['factor_absolute_change'] = data['factor_value'].diff()
                        data['rolling_return'] = data['factor_absolute_change'].rolling(window=rolling).mean()
                    elif self.param_method == 'raw':
                        # Raw Number
                        data['rolling_return'] = data['factor_value'].rolling(window=rolling).mean()
                    else:
                        # Default to percentage change
                        data['rolling_return'] = data['factor_pct_change'].rolling(window=rolling).mean()
                    return data
                else:
                    logging.warning("API data loading failed, falling back to local data")
            
            # Load price data
            price_data = self.data_loader.load_price_data(
                asset, 
                clean_method=clean_method
            )
            
            if price_data is None:
                logging.error(f"Failed to load price data for {asset}")
                return None
            
            # Load factor data (support both old and new parameter-based folder structure)
            factor_data = None
            
            # First try parameter-based folder structure
            if hasattr(self.data_loader, 'load_factor_data_by_param'):
                factor_data = self.data_loader.load_factor_data_by_param(
                    asset, 
                    factor_name,  # Use factor_name as param_name
                    clean_method=clean_method
                )
            
            # Fallback to old structure if parameter-based loading fails
            if factor_data is None:
                factor_data = self.data_loader.load_factor_data(
                    asset, 
                    factor_name, 
                    clean_method=clean_method
                )
            
            if factor_data is None:
                logging.error(f"Failed to load factor data for {asset}/{factor_name}")
                return None
            
            # FIX: Ensure daily frequency and unique timestamps for price data
            price_data = (price_data
                .dropna(subset=['timestamp', 'value'])
                .sort_values('timestamp')
                .drop_duplicates(subset=['timestamp'])
                .set_index('timestamp')
                .resample('1D').last()
                .dropna()
                .reset_index())
            
            # FIX: Ensure daily frequency and unique timestamps for factor data
            factor_data = (factor_data
                .dropna(subset=['timestamp', 'value'])
                .sort_values('timestamp')
                .drop_duplicates(subset=['timestamp'])
                .set_index('timestamp')
                .resample('1D').last()
                .dropna()
                .reset_index())
            
            # Merge price and factor data
            merged_data = pd.merge(
                price_data, 
                factor_data, 
                on='timestamp', 
                how='inner',
                suffixes=('', '_factor')
            )
            
            # FIX: Defensively ensure one row per day after merge
            merged_data['date'] = merged_data['timestamp'].dt.date
            merged_data = (merged_data
                .sort_values('timestamp')
                .groupby('date', as_index=False).last())
            merged_data.drop(columns=['date'], inplace=True)
            
            # Rename columns for consistency with original logic
            merged_data = merged_data.rename(columns={
                'value': 'price',
                'value_factor': 'factor_value'
            })
            
            # Sort by timestamp
            merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)
            
            # Add yesterday's close price logic (exactly like original)
            merged_data['last_price'] = merged_data['price'].shift(1)
            merged_data['price_pct_change'] = merged_data['price'].pct_change()
            
            # Add factor yesterday's value and percentage change (exactly like original)
            merged_data['factor_last_value'] = merged_data['factor_value'].shift(1)
            merged_data['factor_pct_change'] = merged_data['factor_value'].pct_change()
            
            # Calculate rolling return based on param_method
            if self.param_method == 'pct_change':
                # Percentage Change
                # Formula: (x_t - x_{t-1}) / x_{t-1} * 100%
                # Logic: Standardized volatility for cross-asset comparison
                merged_data['rolling_return'] = merged_data['factor_pct_change'].rolling(rolling).mean()
                
            elif self.param_method == 'absolute':
                # Absolute Change
                # Formula: x_t - x_{t-1}
                # Logic: Focus on short-term volatility magnitude
                merged_data['factor_absolute_change'] = merged_data['factor_value'].diff()
                merged_data['rolling_return'] = merged_data['factor_absolute_change'].rolling(rolling).mean()
                
            elif self.param_method == 'raw':
                # Raw Number
                # Formula: x_t (raw factor value)
                # Logic: Direct reflection of absolute indicator level
                merged_data['rolling_return'] = merged_data['factor_value'].rolling(rolling).mean()
                
            else:
                # Default to percentage change
                merged_data['rolling_return'] = merged_data['factor_pct_change'].rolling(rolling).mean()
            
            # Remove rows with NaN values (first row will have NaN for shifted columns)
            merged_data = merged_data.dropna().reset_index(drop=True)
            
            # FIX: Verify we have one row per day
            daily_counts = merged_data['timestamp'].dt.date.value_counts()
            if daily_counts.max() > 1:
                logging.warning(f"Still have multiple rows per day: {daily_counts.head()}")
            else:
                # Only log in verbose mode
                from utils.log_config import should_log
                if should_log('debug'):
                    logging.debug(f"Successfully ensured one row per day. Total days: {len(daily_counts)}")
            
            # Only log data preparation details in verbose mode
            from utils.log_config import should_log
            if should_log('debug'):
                logging.debug(f"Prepared data with rolling={rolling}: {len(merged_data)} records for {asset}/{factor_name}")
                logging.debug(f"Price range: ${merged_data['price'].min():.2f} to ${merged_data['price'].max():.2f}")
                logging.debug(f"Factor range: {merged_data['factor_value'].min():.6f} to {merged_data['factor_value'].max():.6f}")
                logging.debug(f"Rolling return range: {merged_data['rolling_return'].min():.6f} to {merged_data['rolling_return'].max():.6f}")
            
            return merged_data
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None
    
    def run_backtest(self, asset: str, factor_name: str, params: Dict) -> Optional[Dict]:
        """
        Run backtest with EXACT legacy algorithm logic
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param params: Strategy parameters (should include 'rolling', 'long_param', 'short_param', 'date_range')
        :return: Backtest results or None if failed
        """
        # Prepare data with rolling parameter
        rolling = params.get('rolling', 1)  # Default rolling=1 like legacy algorithm
        
        # Get date range for API data
        start_date = None
        end_date = None
        if 'date_range' in params:
            start_date, end_date = params['date_range']
        
        data = self.prepare_data(asset, factor_name, rolling=rolling, 
                                start_date=start_date, end_date=end_date)
        if data is None:
            return None
        
        # Apply date range filter if provided
        if 'date_range' in params:
            start_date, end_date = params['date_range']
            
            # Convert date strings to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data by date range
            data = data[(data['timestamp'] >= start_dt) & (data['timestamp'] <= end_dt)].copy()
            
            if len(data) == 0:
                from utils.log_config import should_log
                if should_log('warning'):
                    logging.warning(f"No data found in date range {start_date} to {end_date}")
                return None
        
        # Execute backtest
        results = self.execute_backtest(data, params)
        
        # Add additional information
        results['asset'] = asset
        results['factor_name'] = factor_name
        results['data_points'] = len(data)
        
        return results
    
    def run_optimization(self, asset: str, factor_name: str, 
                        param_grid: Dict, objective: str = 'sharpe_ratio',
                        param_method: str = 'pct_change', date_range: tuple = None) -> Optional[Dict]:
        """
        Run parameter optimization with EXACT legacy algorithm logic
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param param_grid: Parameter search space
        :param objective: Optimization objective
        :param param_method: Parameter method ('pct_change' or 'absolute')
        :param date_range: Date range tuple (start_date, end_date) for filtering data
        :return: Best result or None if failed
        """
        # Use first parameter combination to prepare data (rolling parameter needed)
        first_params = {}
        for key, values in param_grid.items():
            if values:
                first_params[key] = values[0]
        
        rolling = first_params.get('rolling', 1)  # Default rolling=1 like legacy algorithm
        
        # Get date range for API data
        start_date = None
        end_date = None
        if date_range is not None:
            start_date, end_date = date_range
        
        data = self.prepare_data(asset, factor_name, rolling=rolling, 
                                start_date=start_date, end_date=end_date)
        if data is None:
            return None
        
        # Apply date range filter if provided
        if date_range is not None:
            start_date, end_date = date_range
            print(f"Filtering optimization data by date range: {start_date} to {end_date}")
            
            # Convert date strings to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data by date range
            data = data[(data['timestamp'] >= start_dt) & (data['timestamp'] <= end_dt)].copy()
            
            if len(data) == 0:
                print(f"No data found in date range {start_date} to {end_date}")
                return None
            
            print(f"Optimization data filtered to {len(data)} rows in date range")
        
        # Add param_method to all parameter combinations
        enhanced_param_grid = param_grid.copy()
        enhanced_param_grid['param_method'] = [param_method]
        
        # Run optimization
        best_result = self.optimize_parameters(data, enhanced_param_grid, objective)
        
        # Add additional information
        best_result['asset'] = asset
        best_result['factor_name'] = factor_name
        
        return best_result
    
    def run_standardized_optimization(self, asset: str, factor_name: str,
                                    rolling_windows: List[int] = [1],  # Default rolling=1 like legacy algorithm
                                            long_params: List[float] = None,   # Will use legacy algorithm ranges
        short_params: List[float] = None,  # Will use legacy algorithm ranges
                                    param_method: str = 'pct_change',
                                    objective: str = 'sharpe_ratio',
                                    date_range: tuple = None) -> Optional[Dict]:
        """
        Run standardized optimization with EXACT legacy algorithm parameter ranges
        
        :param asset: Asset symbol
        :param factor_name: Factor name
        :param rolling_windows: List of rolling window sizes (default [1] like legacy algorithm)
        :param long_params: List of long parameters (will use legacy algorithm range if None)
        :param short_params: List of short parameters (will use legacy algorithm range if None)
        :param param_method: Parameter method ('pct_change' or 'absolute')
        :param objective: Optimization objective
        :param date_range: Date range tuple (start_date, end_date) for filtering data
        :return: Best result or None if failed
        """
        # Use EXACT parameter ranges from legacy algorithm if not provided
        if long_params is None:
            # From legacy algorithm: long_param_list = numpy.arange(start=0.02, stop=-0.2, step=-0.02)
            # This creates: [0.02, 0.00, -0.02, -0.04, ..., -0.18]
            long_params = list(np.arange(start=0.02, stop=-0.2, step=-0.02))
        
        if short_params is None:
            # From legacy algorithm: short_param_list = numpy.arange(start=-0.02, stop=0.2, step=0.02)
            # This creates: [-0.02, 0.00, 0.02, 0.04, ..., 0.18]
            short_params = list(np.arange(start=-0.02, stop=0.2, step=0.02))
        
        # Create parameter grid with EXACT same names as legacy algorithm
        param_grid = {
            'rolling': rolling_windows,
            'long_param': long_params,    # EXACT name from legacy algorithm
            'short_param': short_params,  # EXACT name from legacy algorithm
            'param_method': [param_method],
            'lot_size': [0.001]  # Use same lot_size as single backtest for consistency
        }
        
        # Update the strategy's param_method for this optimization
        original_param_method = self.param_method
        self.param_method = param_method
        
        try:
            result = self.run_optimization(asset, factor_name, param_grid, objective, param_method, date_range)
            return result
        finally:
            # Restore original param_method
            self.param_method = original_param_method
    
    def print_results_summary(self, results: Dict):
        """
        Print comprehensive results summary with fixed legacy algorithm logic
        
        :param results: Backtest results
        """
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("(Fixed to match legacy algorithm Logic)")
        print("="*60)
        
        # Basic info
        print(f"Asset: {results.get('asset', 'N/A')}")
        print(f"Factor: {results.get('factor_name', 'N/A')}")
        print(f"Data Processing Method: {self.param_method}")
        print(f"Data Points: {results.get('data_points', 0):,}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results.get('final_capital', 0):,.2f}")
        
        # Performance metrics
        print(f"\nPERFORMANCE METRICS:")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Annual Return: {results.get('annual_return', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        # Trading statistics
        print(f"\nTRADING STATISTICS:")
        print(f"Total Trades: {results.get('num_trades', 0)}")
        print(f"Total Positions: {results.get('num_positions', 0)}")
        print(f"Max Consecutive Losses: {results.get('max_consecutive_losses', 0)}")
        
        # Parameters used
        params = results.get('params', {})
        print(f"\nPARAMETERS USED:")
        print(f"Data Processing Method: {params.get('param_method', self.param_method)}")
        print(f"Rolling: {params.get('rolling', 'N/A')}")
        print(f"Long Param: {params.get('long_param', 'N/A')}")
        print(f"Short Param: {params.get('short_param', 'N/A')}")
        print(f"Lot Size: {params.get('lot_size', 'N/A')}")
        
        print("="*60)


def main():
    """Test the enhanced strategy with standardized optimization output"""
    # Initialize strategy
    strategy = EnhancedNonPriceStrategy(
        initial_capital=10000,
        min_lot_size=0.001
    )
    
    # Get available assets and factors
    assets = strategy.data_loader.get_available_assets()
    if not assets:
        print("No assets available")
        return
    
    asset = assets[0]  # Use first available asset
    factors = strategy.data_loader.get_available_factors(asset)
    
    if not factors:
        print(f"No factors available for {asset}")
        return
    
    factor_name = factors[0]  # Use first available factor
    
    print(f"Testing with {asset}/{factor_name}")
    
            # Define parameters for single backtest (using legacy algorithm defaults)
    params = {
        'long_param': 0.02,     # EXACT name and value from legacy algorithm
        'short_param': -0.02,   # EXACT name and value from legacy algorithm
        'rolling': 1,           # EXACT value from legacy algorithm
        'param_method': 'pct_change',
        'lot_size': 0.001       # Match legacy algorithm hardcoded value
    }
    
    # Run single backtest
    results = strategy.run_backtest(asset, factor_name, params)
    
    if results:
        strategy.print_results_summary(results)
        
        # Run standardized optimization
        print(f"\nRunning standardized optimization...")
        print("This will generate the required output format:")
        print("- Parameter search results table")
        print("- Buy & Hold summary")
        print("- 2x2 plots (Strategy vs BnH, Trade distribution, Performance comparison, Sharpe heatmap)")
        
        best_result = strategy.run_standardized_optimization(
            asset=asset,
            factor_name=factor_name,
            rolling_windows=[1],  # EXACT match to legacy algorithm
                    long_params=None,     # Will use legacy algorithm ranges
        short_params=None,    # Will use legacy algorithm ranges
            param_method='pct_change',
            objective='sharpe_ratio'
        )
        
        if best_result:
            print(f"\nOPTIMIZATION COMPLETED!")
            print(f"Best result found with Sharpe ratio: {best_result.get('sharpe_ratio', 0):.4f}")
            print(f"Check the 'reports' directory for detailed results and plots.")
        else:
            print("Optimization failed")
    else:
        print("Backtest failed")


if __name__ == "__main__":
    main()
