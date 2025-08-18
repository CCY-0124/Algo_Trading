"""
interactive_trading.py

Interactive trading script with user input for:
- Asset selection (BTC, ETH, SOL, TON, TRX, USDC, USDT)
- Factor selection with search functionality
- Parameter input with validation
- Backtest and optimization options
- Enhanced engine integration
"""

import os
import sys
import logging
import glob
import re
import signal
import warnings
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Filter out requests dependency warning
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", category=UserWarning, message=".*urllib3.*chardet.*charset_normalizer.*")

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.enhanced_non_price_strategy import EnhancedNonPriceStrategy
from config.trading_config import DEFAULT_INITIAL_CAPITAL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InteractiveTrading:
    """Interactive trading interface with user input"""
    
    # Supported assets
    SUPPORTED_ASSETS = ['BTC', 'ETH', 'SOL', 'TON', 'TRX', 'USDC', 'USDT']
    
    # Default data directory
    DATA_DIR = r"D:\Trading_Data\glassnode_data2"
    
    # Fixed parameters
    INITIAL_CAPITAL = DEFAULT_INITIAL_CAPITAL
    LOT_SIZE = 0.001
    
    def __init__(self):
        self.asset = None
        self.factor_name = None
        self.factor_file = None
        self.parameters = {}
        self.backtest_ranges = {}
        self.stop_optimization = False
        self.operation_choice = None
        self.use_api = False  # Default to CSV data
        self._setup_signal_handler()
    
    def _setup_signal_handler(self):
        """Setup signal handler for graceful interruption"""
        def signal_handler(signum, frame):
            print("\n\nReceived interrupt signal. Stopping optimization...")
            self.stop_optimization = True
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*60)
        print("           INTERACTIVE TRADING SYSTEM")
        print("="*60)
        print("Welcome to the Interactive Trading System!")
        print("This tool helps you run backtests and optimizations")
        print("with interactive parameter selection.")
        print("Enhanced engine with dynamic lot sizing and position tracking.")
        print("Supports both local CSV files and API data sources.")
        print(f"Fixed parameters: Initial Capital=${self.INITIAL_CAPITAL:,}, Lot Size={self.LOT_SIZE}")
        print("="*60)
    
    def get_operation_choice_first(self) -> str:
        """Get operation choice first before other inputs"""
        print("\n--- OPERATION SELECTION ---")
        print("1. Run Backtest")
        print("2. Run Optimization")
        print("3. Both (Backtest first, then Optimization)")
        
        while True:
            choice = input("\nSelect operation (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return choice
            else:
                print("Please select 1, 2, or 3.")
    
    def get_data_source_choice(self) -> bool:
        """Get data source choice (API vs CSV)"""
        print("\n--- DATA SOURCE SELECTION ---")
        print("1. Use Local CSV Files (faster, requires downloaded data)")
        print("2. Use API Data (slower, requires API key)")
        
        while True:
            choice = input("\nSelect data source (1-2): ").strip()
            if choice == '1':
                print("SUCCESS: Using local CSV files")
                return False  # use_api = False
            elif choice == '2':
                print("SUCCESS: Using API data")
                return True   # use_api = True
            else:
                print("Please select 1 or 2.")
    
    def get_api_factor_selection(self) -> str:
        """Get factor selection for API data"""
        print(f"\n--- API FACTOR SELECTION FOR {self.asset} ---")
        print("Enter the API endpoint path (e.g., indicators/sopr_account_based)")
        print("Note: Do not include the base URL, just the endpoint path")
        
        while True:
            factor_path = input("API endpoint: ").strip()
            if factor_path:
                # Remove leading slash if present
                if factor_path.startswith('/'):
                    factor_path = factor_path[1:]
                print(f"SUCCESS: Using API endpoint: {factor_path}")
                return factor_path
            else:
                print("Please enter a valid API endpoint path.")
        

    
    def get_asset_selection(self) -> str:
        """Get asset selection from user"""
        print("\n--- ASSET SELECTION ---")
        print("Available assets:")
        for i, asset in enumerate(self.SUPPORTED_ASSETS, 1):
            print(f"  {i}. {asset}")
        
        while True:
            try:
                choice = input(f"\nSelect asset (1-{len(self.SUPPORTED_ASSETS)}) or enter asset code: ").strip().upper()
                
                # Check if it's a number
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.SUPPORTED_ASSETS):
                        return self.SUPPORTED_ASSETS[idx]
                    else:
                        print(f"Please enter a number between 1 and {len(self.SUPPORTED_ASSETS)}")
                        continue
                
                # Check if it's a valid asset code
                if choice in self.SUPPORTED_ASSETS:
                    return choice
                else:
                    print(f"Invalid asset code. Please choose from: {', '.join(self.SUPPORTED_ASSETS)}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    
    def search_factors(self, asset: str, search_term: str = "") -> List[str]:
        """Search for factor files in the asset directory"""
        asset_dir = os.path.join(self.DATA_DIR, asset)
        
        if not os.path.exists(asset_dir):
            print(f"Warning: Asset directory not found: {asset_dir}")
            return []
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(asset_dir, "*.csv"))
        
        # Extract factor names (remove .csv extension and path)
        factor_names = []
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            if filename.endswith('.csv'):
                factor_name = filename[:-4]  # Remove .csv extension
                # Skip market price files
                if not factor_name.endswith('_market_price_usd_close_tier1'):
                    factor_names.append(factor_name)
        
        # Filter by search term if provided
        if search_term:
            filtered_factors = []
            for factor in factor_names:
                if search_term.lower() in factor.lower():
                    filtered_factors.append(factor)
            return filtered_factors
        
        return factor_names
    
    def get_factor_selection(self, asset: str) -> Tuple[str, str]:
        """Get factor selection from user with search functionality"""
        print(f"\n--- FACTOR SELECTION FOR {asset} ---")
        
        while True:
            print("\nOptions:")
            print("1. List all available factors")
            print("2. Search factors by keyword")
            print("3. Enter factor name directly")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                factors = self.search_factors(asset)
                if not factors:
                    print("No factors found for this asset.")
                    continue
                
                print(f"\nAvailable factors for {asset}:")
                for i, factor in enumerate(factors, 1):
                    print(f"  {i}. {factor}")
                
                # Let user select from list
                try:
                    factor_choice = input(f"\nSelect factor (1-{len(factors)}) or enter factor name: ").strip()
                    
                    if factor_choice.isdigit():
                        idx = int(factor_choice) - 1
                        if 0 <= idx < len(factors):
                            factor_name = factors[idx]
                        else:
                            print("Invalid selection.")
                            continue
                    else:
                        factor_name = factor_choice
                    
                    # Verify factor exists
                    factor_file = os.path.join(self.DATA_DIR, asset, f"{factor_name}.csv")
                    if os.path.exists(factor_file):
                        return factor_name, factor_file
                    else:
                        print(f"Factor file not found: {factor_file}")
                        continue
                        
                except (ValueError, KeyboardInterrupt):
                    continue
            
            elif choice == "2":
                search_term = input("Enter search keyword: ").strip()
                if not search_term:
                    print("Please enter a search term.")
                    continue
                
                factors = self.search_factors(asset, search_term)
                if not factors:
                    print(f"No factors found matching '{search_term}'")
                    continue
                
                print(f"\nFactors matching '{search_term}':")
                for i, factor in enumerate(factors, 1):
                    print(f"  {i}. {factor}")
                
                # Let user select from search results
                try:
                    factor_choice = input(f"\nSelect factor (1-{len(factors)}) or enter factor name: ").strip()
                    
                    if factor_choice.isdigit():
                        idx = int(factor_choice) - 1
                        if 0 <= idx < len(factors):
                            factor_name = factors[idx]
                        else:
                            print("Invalid selection.")
                            continue
                    else:
                        factor_name = factor_choice
                    
                    # Verify factor exists
                    factor_file = os.path.join(self.DATA_DIR, asset, f"{factor_name}.csv")
                    if os.path.exists(factor_file):
                        return factor_name, factor_file
                    else:
                        print(f"Factor file not found: {factor_file}")
                        continue
                        
                except (ValueError, KeyboardInterrupt):
                    continue
            
            elif choice == "3":
                factor_name = input("Enter factor name (e.g., _breakdowns_spent_volume_sum_by_lth_sth_tier3): ").strip()
                if not factor_name:
                    print("Please enter a factor name.")
                    continue
                
                factor_file = os.path.join(self.DATA_DIR, asset, f"{factor_name}.csv")
                if os.path.exists(factor_file):
                    return factor_name, factor_file
                else:
                    print(f"Factor file not found: {factor_file}")
                    continue
            
            else:
                print("Invalid option. Please select 1, 2, or 3.")
    
    def get_parameters(self) -> Dict:
        """Get strategy parameters from user"""
        print("\n--- PARAMETER INPUT ---")
        print(f"Fixed parameters: Initial Capital=${self.INITIAL_CAPITAL:,}, Lot Size={self.LOT_SIZE}")
        
        # Show what operation we're getting parameters for
        if self.operation_choice == '1':
            print("Enter parameters for BACKTEST:")
        elif self.operation_choice == '2':
            print("Enter parameter ranges for OPTIMIZATION:")
        else:  # operation_choice == '3'
            print("Enter parameters for BACKTEST and OPTIMIZATION:")
        
        params = {}
        
        # Set fixed parameters
        params['initial_capital'] = self.INITIAL_CAPITAL
        params['lot_size'] = self.LOT_SIZE
        
        # For backtest or both, ask for individual parameters
        if self.operation_choice in ['1', '3']:
            # Rolling window
            while True:
                try:
                    rolling_input = input("Rolling window (default: 1): ").strip()
                    if not rolling_input:
                        params['rolling'] = 1
                        break
                    else:
                        rolling_window = int(rolling_input)
                        if rolling_window > 0:
                            params['rolling'] = rolling_window
                            break
                        else:
                            print("Rolling window must be positive.")
                except ValueError:
                    print("Please enter a valid integer.")
            
            # Long threshold
            while True:
                try:
                    long_input = input("Long param (default: 0.02): ").strip()
                    if not long_input:
                        params['long_param'] = 0.02
                        break
                    else:
                        long_param = float(long_input)
                        params['long_param'] = long_param
                        break
                except ValueError:
                    print("Please enter a valid number.")
            
            # Short threshold
            while True:
                try:
                    short_input = input("Short param (default: -0.02): ").strip()
                    if not short_input:
                        params['short_param'] = -0.02
                        break
                    else:
                        short_param = float(short_input)
                        params['short_param'] = short_param
                        break
                except ValueError:
                    print("Please enter a valid number.")
            
            # Date range (for API data or optimization)
            if self.use_api or self.operation_choice in ['2', '3']:
                print("\n--- DATE RANGE SELECTION ---")
                while True:
                    try:
                        start_date_input = input("Start date (YYYY-MM-DD) (default: 2023-01-01): ").strip()
                        if not start_date_input:
                            start_date = "2023-01-01"
                        else:
                            # Validate date format
                            pd.to_datetime(start_date_input)
                            start_date = start_date_input
                        
                        end_date_input = input("End date (YYYY-MM-DD) (default: 2024-12-31): ").strip()
                        if not end_date_input:
                            end_date = "2024-12-31"
                        else:
                            # Validate date format
                            pd.to_datetime(end_date_input)
                            end_date = end_date_input
                        
                        # Validate date range
                        if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
                            print("Start date must be before end date.")
                            continue
                        
                        params['date_range'] = (start_date, end_date)
                        print(f"SUCCESS: Date range set: {start_date} to {end_date}")
                        break
                        
                    except ValueError:
                        print("Please enter dates in YYYY-MM-DD format.")
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Parameter method (for all operations)
            while True:
                param_method_input = input("Parameter method (pct_change/absolute/raw) (default: pct_change): ").strip()
                if not param_method_input:
                    params['param_method'] = 'pct_change'
                    break
                elif param_method_input in ['pct_change', 'absolute', 'raw']:
                    params['param_method'] = param_method_input
                    break
                else:
                    print("Please enter 'pct_change', 'absolute', or 'raw'.")
            
            # Always set objective to sharpe_ratio (no need to ask)
            if self.operation_choice in ['2', '3']:
                params['objective'] = 'sharpe_ratio'
        
        # Only ask for optimization ranges if optimization is selected
        if self.operation_choice in ['2', '3']:
            print("\n--- OPTIMIZATION PARAMETER RANGES ---")
            print("Enter parameter ranges for optimization (press Enter for default values):")
            
            # Rolling range
            while True:
                try:
                    rolling_range_input = input("Rolling range (start,end,step) (default: 1,10,1): ").strip()
                    if not rolling_range_input:
                        params['rolling_range'] = (1, 10, 1)
                        break
                    else:
                        parts = rolling_range_input.split(',')
                        if len(parts) == 3:
                            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
                            if start > 0 and end > start and step > 0:
                                params['rolling_range'] = (start, end, step)
                                break
                            else:
                                print("Invalid range values. Start < End, all values > 0.")
                        else:
                            print("Please enter in format: start,end,step")
                except ValueError:
                    print("Please enter valid integers.")
            
            # Long param range
            while True:
                try:
                    long_range_input = input("Long param range (start,end,step) (default: -0.1,0.1,0.01): ").strip()
                    if not long_range_input:
                        params['long_param_range'] = (-0.1, 0.1, 0.01)
                        break
                    else:
                        parts = long_range_input.split(',')
                        if len(parts) == 3:
                            start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
                            if start < end and step > 0:
                                params['long_param_range'] = (start, end, step)
                                break
                            else:
                                print("Invalid range values. Start < End, step > 0.")
                        else:
                            print("Please enter in format: start,end,step")
                except ValueError:
                    print("Please enter valid numbers.")
            
            # Short param range
            while True:
                try:
                    short_range_input = input("Short param range (start,end,step) (default: -0.1,0.1,0.01): ").strip()
                    if not short_range_input:
                        params['short_param_range'] = (-0.1, 0.1, 0.01)
                        break
                    else:
                        parts = short_range_input.split(',')
                        if len(parts) == 3:
                            start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
                            if start < end and step > 0:
                                params['short_param_range'] = (start, end, step)
                                break
                            else:
                                print("Invalid range values. Start < End, step > 0.")
                        else:
                            print("Please enter in format: start,end,step")
                except ValueError:
                    print("Please enter valid numbers.")
            
            # Date range
            while True:
                try:
                    date_range_input = input("Date range (start,end) (default: 2023-01-01,2024-12-31): ").strip()
                    if not date_range_input:
                        params['date_range'] = ('2023-01-01', '2024-12-31')
                        break
                    else:
                        parts = date_range_input.split(',')
                        if len(parts) == 2:
                            start_date, end_date = parts[0].strip(), parts[1].strip()
                            # Basic date format validation
                            if len(start_date) == 10 and len(end_date) == 10:
                                params['date_range'] = (start_date, end_date)
                                break
                            else:
                                print("Please enter dates in YYYY-MM-DD format.")
                        else:
                            print("Please enter in format: start_date,end_date")
                except ValueError:
                    print("Please enter valid dates.")
        
        return params
    
    def confirm_settings(self) -> bool:
        """Display and confirm settings"""
        print("\n--- CONFIRM SETTINGS ---")
        print(f"Operation: {'Backtest' if self.operation_choice == '1' else 'Optimization' if self.operation_choice == '2' else 'Both (Backtest + Optimization)'}")
        print(f"Data Source: {'API' if self.use_api else 'Local CSV Files'}")
        print(f"Asset: {self.asset}")
        print(f"Factor: {self.factor_name}")
        if not self.use_api:
            print(f"Factor file: {self.factor_file}")
        print(f"Parameter Method: {self.parameters.get('param_method', 'pct_change')}")
        print(f"Parameters: {self.parameters}")
        if self.backtest_ranges and self.operation_choice in ['2', '3']:
            print(f"Backtest ranges: {self.backtest_ranges}")
        
        while True:
            confirm = input("\nProceed with these settings? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'.")
    
    def run_backtest(self) -> Dict:
        """Run backtest with current settings"""
        print(f"\n--- RUNNING BACKTEST FOR {self.asset} ---")
        
        try:
            # Initialize enhanced strategy with user parameters
            initial_capital = self.parameters.get('initial_capital', self.INITIAL_CAPITAL)
            lot_size = self.parameters.get('lot_size', self.LOT_SIZE)
            
            # Get param_method from parameters
            param_method = self.parameters.get('param_method', 'pct_change')
            
            strategy = EnhancedNonPriceStrategy(
                initial_capital=initial_capital,
                min_lot_size=lot_size,
                use_api=self.use_api,
                param_method=param_method
            )
            
            # For API data, we need to add date range to parameters
            if self.use_api:
                # Get date range from parameters (user should have provided it)
                date_range = self.parameters.get('date_range')
                if date_range:
                    print(f"INFO: Loading API data: {date_range[0]} to {date_range[1]}")
                    # Add date_range to parameters like simple_optimization.py does
                    self.parameters['date_range'] = date_range
                else:
                    print("WARNING: No date range provided, using default")
                    self.parameters['date_range'] = ('2023-01-01', '2024-12-31')
            
            # Execute backtest
            print("INFO: Running backtest...")
            result = strategy.run_backtest(
                asset=self.asset,
                factor_name=self.factor_name,
                params=self.parameters
            )
            
            if result:
                # Display results
                print("\n=== BACKTEST RESULTS ===")
                strategy.print_results_summary(result)
                return result
            else:
                print("Backtest failed to produce results.")
                return {}
            
        except Exception as e:
            print(f"Backtest failed: {str(e)}")
            logging.exception("Backtest error")
            return {}
    
    def run_optimization(self) -> Dict:
        """Run optimization with current settings"""
        print(f"\n--- RUNNING OPTIMIZATION FOR {self.asset} ---")
        print("Press Ctrl+C to stop optimization at any time.")
        
        # Let user choose parameter method for optimization
        print("\n=== DATA PROCESSING METHOD FOR OPTIMIZATION ===")
        print("1. Raw Number - Use raw factor values")
        print("2. Absolute Change - Use factor value differences")
        print("3. Percentage Change - Use factor percentage changes")
        
        while True:
            choice = input("\nSelect data processing method (1-3): ").strip()
            if choice == '1':
                optimization_param_method = 'raw'
                break
            elif choice == '2':
                optimization_param_method = 'absolute'
                break
            elif choice == '3':
                optimization_param_method = 'pct_change'
                break
            else:
                print("ERROR: Invalid choice. Please select 1, 2, or 3.")
        
        # Display optimization settings
        print(f"INFO: Data Processing Method: {optimization_param_method}")
        if self.use_api:
            date_range = self.parameters.get('date_range', ('2023-01-01', '2024-12-31'))
            print(f"INFO: Date Range: {date_range[0]} to {date_range[1]}")
        print(f"INFO: Optimization Objective: {self.parameters.get('objective', 'sharpe_ratio')}")
        
        try:
            # Initialize enhanced strategy with user parameters
            initial_capital = self.parameters.get('initial_capital', self.INITIAL_CAPITAL)
            lot_size = self.parameters.get('lot_size', self.LOT_SIZE)
            
            strategy = EnhancedNonPriceStrategy(
                initial_capital=initial_capital,
                min_lot_size=lot_size,
                use_api=self.use_api,
                param_method=optimization_param_method
            )
            
            # Create parameter grid from ranges
            param_grid = {}
            
            if 'rolling_range' in self.backtest_ranges:
                start, end, step = self.backtest_ranges['rolling_range']
                param_grid['rolling'] = list(range(start, end + 1, step))
            
            if 'long_param_range' in self.backtest_ranges:
                start, end, step = self.backtest_ranges['long_param_range']
                import numpy as np
                param_grid['long_param'] = list(np.arange(start, end + step, step))
            
            if 'short_param_range' in self.backtest_ranges:
                start, end, step = self.backtest_ranges['short_param_range']
                import numpy as np
                param_grid['short_param'] = list(np.arange(start, end + step, step))
            
            # Add initial capital range
            if 'initial_capital_range' in self.backtest_ranges:
                start, end, step = self.backtest_ranges['initial_capital_range']
                import numpy as np
                param_grid['initial_capital'] = list(np.arange(start, end + step, step))
            else:
                param_grid['initial_capital'] = [self.parameters.get('initial_capital', self.INITIAL_CAPITAL)]
            
            # Add lot size range
            if 'lot_size_range' in self.backtest_ranges:
                start, end, step = self.backtest_ranges['lot_size_range']
                import numpy as np
                param_grid['lot_size'] = list(np.arange(start, end + step, step))
            else:
                param_grid['lot_size'] = [self.parameters.get('lot_size', self.LOT_SIZE)]
            
            # Format parameter grid for clean display
            formatted_grid = {}
            for key, values in param_grid.items():
                if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                    formatted_grid[key] = [f"{v:.4f}" if isinstance(v, float) else v for v in values]
                else:
                    formatted_grid[key] = values
            print(f"Parameter grid: {formatted_grid}")
            
            # Calculate total combinations for progress tracking
            total_combinations = 1
            for param_values in param_grid.values():
                total_combinations *= len(param_values)
            print(f"Total parameter combinations to test: {total_combinations}")
            
            # Execute optimization
            try:
                # Get objective from parameters
                objective = self.parameters.get('objective', 'sharpe_ratio')
                
                # Get date range for API data or optimization
                date_range = None
                if self.use_api or 'date_range' in self.parameters:
                    date_range = self.parameters.get('date_range')
                    if date_range:
                        print(f"INFO: Using date range: {date_range[0]} to {date_range[1]}")
                    else:
                        print("WARNING: No date range provided, using default")
                        date_range = ('2023-01-01', '2024-12-31')
                
                # Use run_standardized_optimization like simple_optimization.py
                # Extract parameter ranges from backtest_ranges
                rolling_windows = []
                if 'rolling_range' in self.backtest_ranges:
                    start, end, step = self.backtest_ranges['rolling_range']
                    rolling_windows = list(range(start, end + 1, step))
                else:
                    rolling_windows = [self.parameters.get('rolling', 1)]
                
                long_params = []
                if 'long_param_range' in self.backtest_ranges:
                    start, end, step = self.backtest_ranges['long_param_range']
                    import numpy as np
                    long_params = list(np.arange(start, end + step, step))
                else:
                    long_params = [self.parameters.get('long_param', 0.02)]
                
                short_params = []
                if 'short_param_range' in self.backtest_ranges:
                    start, end, step = self.backtest_ranges['short_param_range']
                    import numpy as np
                    short_params = list(np.arange(start, end + step, step))
                else:
                    short_params = [self.parameters.get('short_param', -0.02)]
                
                print("INFO: Running optimization...")
                best_result = strategy.run_standardized_optimization(
                    asset=self.asset,
                    factor_name=self.factor_name,
                    rolling_windows=rolling_windows,
                    long_params=long_params,
                    short_params=short_params,
                    param_method=optimization_param_method,
                    objective=objective,
                    date_range=date_range
                )
            except KeyboardInterrupt:
                print("\nOptimization stopped by user.")
                return {}
            
            if best_result:
                # Display results
                print("\n=== OPTIMIZATION RESULTS ===")
                print(f"INFO: Data Processing Method: {optimization_param_method}")
                print(f"Best Sharpe ratio: {best_result.get('sharpe_ratio', 0):.4f}")
                print(f"Best parameters: {best_result.get('params', {})}")
                print(f"Best total return: {best_result.get('total_return', 0) * 100:.2f}%")
                print(f"Best annual return: {best_result.get('annual_return', 0) * 100:.2f}%")
                print(f"Best max drawdown: {best_result.get('max_drawdown', 0) * 100:.2f}%")
                
                # Print top 50 and bottom 50 results if available
                if 'results_df' in best_result:
                    results_df = best_result['results_df']
                    total_results = len(results_df)
                    
                    print(f"\n=== TOP 50 RESULTS (out of {total_results} total) ===")
                    top_50 = results_df.head(50)
                    for idx, row in top_50.iterrows():
                        print(f"{idx+1:3d}. Sharpe: {row['sharpe']:.4f}, "
                              f"Return: {row['total_profit']:.2f}, "
                              f"Trades: {row['num_of_trade']:3d}, "
                              f"Params: rolling={row['rolling']}, "
                              f"long={row['long_param']:.3f}, "
                              f"short={row['short_param']:.3f}")
                    
                    print(f"\n=== BOTTOM 50 RESULTS (out of {total_results} total) ===")
                    bottom_50 = results_df.tail(50)
                    for idx, row in bottom_50.iterrows():
                        print(f"{idx+1:3d}. Sharpe: {row['sharpe']:.4f}, "
                              f"Return: {row['total_profit']:.2f}, "
                              f"Trades: {row['num_of_trade']:3d}, "
                              f"Params: rolling={row['rolling']}, "
                              f"long={row['long_param']:.3f}, "
                              f"short={row['short_param']:.3f}")
                    
                    print(f"\nINFO: Full results saved to reports/optimization_results.csv")
                    print(f"INFO: Detailed plots saved to reports/")
                
                return best_result
            else:
                print("Optimization failed to produce results.")
                return {}
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            logging.exception("Optimization error")
            return {}
    
    def run(self):
        """Main interactive workflow"""
        try:
            # Display welcome
            self.display_welcome()
            
            # Get operation choice first
            self.operation_choice = self.get_operation_choice_first()
            
            # Get data source choice
            self.use_api = self.get_data_source_choice()
            
            # Get asset selection
            self.asset = self.get_asset_selection()
            
            # Get factor selection
            if not self.use_api:
                self.factor_name, self.factor_file = self.get_factor_selection(self.asset)
            else:
                self.factor_name = self.get_api_factor_selection()
                self.factor_file = None
            
            # Get parameters (including ranges if optimization is selected)
            self.parameters = self.get_parameters()
            
            # Extract ranges from parameters if they exist
            self.backtest_ranges = {}
            if 'rolling_range' in self.parameters:
                self.backtest_ranges['rolling_range'] = self.parameters['rolling_range']
            if 'long_param_range' in self.parameters:
                self.backtest_ranges['long_param_range'] = self.parameters['long_param_range']
            if 'short_param_range' in self.parameters:
                self.backtest_ranges['short_param_range'] = self.parameters['short_param_range']
            if 'date_range' in self.parameters:
                self.backtest_ranges['date_range'] = self.parameters['date_range']
                # date_range is already in self.parameters, so it will be passed to backtest
            
            # Set fixed ranges for initial capital and lot size
            self.backtest_ranges['initial_capital_range'] = (self.INITIAL_CAPITAL, self.INITIAL_CAPITAL, 1000)
            self.backtest_ranges['lot_size_range'] = (self.LOT_SIZE, self.LOT_SIZE, 0.1)
            
            # Confirm settings
            if not self.confirm_settings():
                print("Operation cancelled.")
                return
            
            # Execute operations
            if self.operation_choice == '1':
                self.run_backtest()
            elif self.operation_choice == '2':
                self.run_optimization()
            elif self.operation_choice == '3':
                self.run_backtest()
                self.run_optimization()
            
            print("\n--- OPERATION COMPLETE ---")
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            logging.exception("Unexpected error")

def main():
    """Main entry point"""
    interactive_trading = InteractiveTrading()
    interactive_trading.run()

if __name__ == "__main__":
    main() 