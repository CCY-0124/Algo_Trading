"""
local_data_loader.py

Local data loader for CSV files from Glassnode data directory.
Handles loading price data and factor data from local CSV files.

:precondition: CSV files must be in D:\\Trading_Data\\glassnode_data2\\{asset}\\ directory
:postcondition: Returns standardized DataFrames with 'timestamp' and 'value' columns
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataCleaner:
    """
    Utility class for cleaning and handling NaN values in time series data.
    
    Provides various methods to detect, handle, and clean NaN values in financial data.
    """
    
    @staticmethod
    def detect_nan_patterns(df: pd.DataFrame, value_col: str = 'value') -> Dict:
        """
        Detect patterns in NaN values to understand data quality issues.
        
        :param df: DataFrame to analyze
        :param value_col: Name of the value column
        :return: Dictionary with NaN analysis results
        """
        analysis = {
            'total_rows': len(df),
            'nan_count': df[value_col].isna().sum(),
            'nan_percentage': (df[value_col].isna().sum() / len(df)) * 100,
            'consecutive_nan_max': 0,
            'nan_gaps': []
        }
        
        if analysis['nan_count'] > 0:
            # Find consecutive NaN sequences
            nan_series = df[value_col].isna()
            consecutive_nan = 0
            max_consecutive = 0
            current_gap_start = None
            
            for i, is_nan in enumerate(nan_series):
                if is_nan:
                    consecutive_nan += 1
                    if current_gap_start is None:
                        current_gap_start = i
                    max_consecutive = max(max_consecutive, consecutive_nan)
                else:
                    if consecutive_nan > 0:
                        analysis['nan_gaps'].append({
                            'start_index': current_gap_start,
                            'end_index': i - 1,
                            'length': consecutive_nan
                        })
                    consecutive_nan = 0
                    current_gap_start = None
            
            # Handle case where NaN sequence ends at the end of data
            if consecutive_nan > 0:
                analysis['nan_gaps'].append({
                    'start_index': current_gap_start,
                    'end_index': len(df) - 1,
                    'length': consecutive_nan
                })
            
            analysis['consecutive_nan_max'] = max_consecutive
        
        return analysis
    
    @staticmethod
    def clean_nan_values(df: pd.DataFrame, value_col: str = 'value', method: str = 'drop') -> pd.DataFrame:
        """
        Clean NaN values using specified method.
        
        :param df: DataFrame to clean
        :param value_col: Name of the value column
        :param method: Cleaning method ('drop', 'forward_fill', 'backward_fill', 'interpolate', 'mean_fill')
        :return: Cleaned DataFrame
        """
        if df.empty:
            return df
        
        original_count = len(df)
        nan_count = df[value_col].isna().sum()
        
        if nan_count == 0:
            return df
        
        logging.info(f"Cleaning {nan_count} NaN values using method: {method}")
        
        if method == 'drop':
            df_clean = df.dropna(subset=[value_col])
            logging.info(f"Dropped {nan_count} rows with NaN values, remaining: {len(df_clean)} rows")
            
        elif method == 'forward_fill':
            df_clean = df.copy()
            df_clean[value_col] = df_clean[value_col].fillna(method='ffill')
            remaining_nan = df_clean[value_col].isna().sum()
            logging.info(f"Forward filled {nan_count - remaining_nan} NaN values, {remaining_nan} remaining")
            
        elif method == 'backward_fill':
            df_clean = df.copy()
            df_clean[value_col] = df_clean[value_col].fillna(method='bfill')
            remaining_nan = df_clean[value_col].isna().sum()
            logging.info(f"Backward filled {nan_count - remaining_nan} NaN values, {remaining_nan} remaining")
            
        elif method == 'interpolate':
            df_clean = df.copy()
            df_clean[value_col] = df_clean[value_col].interpolate(method='linear')
            remaining_nan = df_clean[value_col].isna().sum()
            logging.info(f"Interpolated {nan_count - remaining_nan} NaN values, {remaining_nan} remaining")
            
        elif method == 'mean_fill':
            df_clean = df.copy()
            mean_val = df_clean[value_col].mean()
            df_clean[value_col] = df_clean[value_col].fillna(mean_val)
            logging.info(f"Filled {nan_count} NaN values with mean: {mean_val:.4f}")
            
        else:
            logging.warning(f"Unknown cleaning method: {method}, using 'drop' instead")
            df_clean = df.dropna(subset=[value_col])
        
        return df_clean
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame, value_col: str = 'value', timestamp_col: str = 'timestamp') -> Dict:
        """
        Validate data quality and provide comprehensive report.
        
        :param df: DataFrame to validate
        :param value_col: Name of the value column
        :param timestamp_col: Name of the timestamp column
        :return: Dictionary with validation results
        """
        validation = {
            'total_rows': len(df),
            'has_duplicates': df.index.duplicated().any(),
            'duplicate_count': df.index.duplicated().sum(),
            'timestamp_issues': [],
            'value_issues': [],
            'recommendations': []
        }
        
        # Check timestamp issues
        if timestamp_col in df.columns:
            if df[timestamp_col].isna().any():
                validation['timestamp_issues'].append('Contains NaT values')
            if not df[timestamp_col].is_monotonic_increasing:
                validation['timestamp_issues'].append('Timestamps not in ascending order')
            if df[timestamp_col].duplicated().any():
                validation['timestamp_issues'].append('Contains duplicate timestamps')
        
        # Check value issues
        if value_col in df.columns:
            nan_analysis = DataCleaner.detect_nan_patterns(df, value_col)
            validation['nan_analysis'] = nan_analysis
            
            if nan_analysis['nan_count'] > 0:
                validation['value_issues'].append(f"Contains {nan_analysis['nan_count']} NaN values ({nan_analysis['nan_percentage']:.2f}%)")
            
            # Check for extreme values
            if not df[value_col].isna().all():
                q1 = df[value_col].quantile(0.25)
                q3 = df[value_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[value_col] < lower_bound) | (df[value_col] > upper_bound)]
                if len(outliers) > 0:
                    validation['value_issues'].append(f"Contains {len(outliers)} potential outliers")
        
        # Generate recommendations
        if validation['has_duplicates']:
            validation['recommendations'].append('Remove duplicate timestamps')
        
        if validation['timestamp_issues']:
            validation['recommendations'].append('Fix timestamp ordering and duplicates')
        
        if validation['value_issues']:
            validation['recommendations'].append('Consider cleaning NaN values and outliers')
        
        if validation['total_rows'] < 100:
            validation['recommendations'].append('Limited data points, consider longer time period')
        
        return validation


class ColumnDetector:
    """
    Utility class for detecting and selecting appropriate columns from CSV files.
    """
    
    @staticmethod
    def detect_timestamp_and_value_columns(df: pd.DataFrame, file_path: str) -> Optional[Tuple[str, str]]:
        """
        Detect timestamp and value columns from DataFrame.
        
        :param df: DataFrame to analyze
        :param file_path: Path to the CSV file for error reporting
        :return: Tuple of (timestamp_col, value_col) or None if detection failed
        """
        # Show available columns and data preview
        logging.info(f"Available columns in {file_path}: {list(df.columns)}")
        logging.info(f"Data shape: {df.shape}")
        
        # Show last 10 rows for user reference
        print(f"\n=== Data Preview (Last 10 rows) from {file_path} ===")
        print(df.tail(10).to_string())
        print("=" * 80)
        
        # Auto-detect timestamp column
        timestamp_col = None
        timestamp_candidates = ['t', 'timestamp', 'time', 'date', 'datetime']
        
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                timestamp_col = candidate
                break
        
        # If no standard timestamp column found, look for datetime-like columns
        if timestamp_col is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    timestamp_col = col
                    break
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isna().all():  # At least some valid dates
                        timestamp_col = col
                        break
                except:
                    continue
        
        # Auto-detect value column
        value_col = None
        value_candidates = ['v', 'value', 'price', 'close', 'open', 'high', 'low', 'volume']
        
        for candidate in value_candidates:
            if candidate in df.columns:
                value_col = candidate
                break
        
        # If no standard value column found, look for numeric columns
        if value_col is None:
            for col in df.columns:
                if col == timestamp_col:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    value_col = col
                    break
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():  # At least some valid numbers
                        value_col = col
                        break
                except:
                    continue
        
        if timestamp_col is None:
            logging.error(f"Could not detect timestamp column in {file_path}")
            logging.error(f"Available columns: {list(df.columns)}")
            return None
        
        if value_col is None:
            logging.error(f"Could not detect value column in {file_path}")
            logging.error(f"Available columns: {list(df.columns)}")
            return None
        
        logging.info(f"Auto-detected columns: timestamp='{timestamp_col}', value='{value_col}'")
        return timestamp_col, value_col
    
    @staticmethod
    def interactive_column_selection(df: pd.DataFrame, file_path: str) -> Optional[Tuple[str, str]]:
        """
        Interactive column selection for timestamp and value columns.
        
        :param df: DataFrame to analyze
        :param file_path: Path to the CSV file
        :return: Tuple of (timestamp_col, value_col) or None if selection failed
        """
        print(f"\n=== Interactive Column Selection for {file_path} ===")
        print(f"Available columns: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print("\nData preview (last 10 rows):")
        print(df.tail(10).to_string())
        print("=" * 80)
        
        # Auto-detect first
        auto_detected = ColumnDetector.detect_timestamp_and_value_columns(df, file_path)
        if auto_detected:
            timestamp_col, value_col = auto_detected
            print(f"\nAuto-detected columns:")
            print(f"  Timestamp: {timestamp_col}")
            print(f"  Value: {value_col}")
            
            # Ask user if they want to use auto-detected columns
            while True:
                choice = input("\nUse auto-detected columns? (y/n): ").lower().strip()
                if choice in ['y', 'yes']:
                    return timestamp_col, value_col
                elif choice in ['n', 'no']:
                    break
                else:
                    print("Please enter 'y' or 'n'")
        
        # Manual selection
        print("\nManual column selection:")
        
        # Select timestamp column
        print(f"\nAvailable columns for timestamp: {list(df.columns)}")
        while True:
            timestamp_col = input("Enter timestamp column name: ").strip()
            if timestamp_col in df.columns:
                break
            else:
                print(f"Column '{timestamp_col}' not found. Available columns: {list(df.columns)}")
        
        # Select value column
        remaining_cols = [col for col in df.columns if col != timestamp_col]
        print(f"\nAvailable columns for value: {remaining_cols}")
        while True:
            value_col = input("Enter value column name: ").strip()
            if value_col in df.columns:
                break
            else:
                print(f"Column '{value_col}' not found. Available columns: {list(df.columns)}")
        
        print(f"\nSelected columns:")
        print(f"  Timestamp: {timestamp_col}")
        print(f"  Value: {value_col}")
        
        return timestamp_col, value_col


class LocalDataLoader:
    """
    Local data loader for CSV files from Glassnode data directory.
    
    Handles loading and validating price data and factor data from local CSV files.
    Supports automatic file discovery and data validation.
    """
    
    def __init__(self, base_path: str = r"D:\Trading_Data\glassnode_data2"):
        """
        Initialize the local data loader.
        
        :param base_path: Base path to the Glassnode data directory
        """
        self.base_path = base_path
        self.price_filename = "market_price_usd_ohlc_tier2.csv"
        
        # Validate base path exists
        if not os.path.exists(base_path):
            logging.warning(f"Base path does not exist: {base_path}")
            logging.info("Please ensure the data directory exists and contains asset subdirectories")
    
    def get_available_assets(self) -> List[str]:
        """
        Get list of available assets (subdirectories) in the data directory.
        
        :return: List of asset symbols (e.g., ['BTC', 'ETH', ...])
        """
        if not os.path.exists(self.base_path):
            logging.error(f"Base path does not exist: {self.base_path}")
            return []
        
        assets = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path):
                assets.append(item)
        
        logging.info(f"Found {len(assets)} assets: {assets}")
        return sorted(assets)
    
    def get_available_factors(self, asset: str) -> List[str]:
        """
        Get list of available factor files for a specific asset.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :return: List of factor filenames (without .csv extension)
        """
        asset_path = os.path.join(self.base_path, asset)
        
        if not os.path.exists(asset_path):
            logging.error(f"Asset directory does not exist: {asset_path}")
            return []
        
        # Get all CSV files in the asset directory
        csv_files = glob.glob(os.path.join(asset_path, "*.csv"))
        
        factors = []
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            if filename.endswith('.csv'):
                factor_name = filename[:-4]  # Remove .csv extension
                # Skip the price file
                if factor_name != self.price_filename[:-4]:
                    factors.append(factor_name)
        
        logging.info(f"Found {len(factors)} factors for {asset}")
        return sorted(factors)
    
    def _validate_and_process_csv_structure(self, df: pd.DataFrame, file_path: str, 
                                          timestamp_col: str = None, value_col: str = None) -> Optional[pd.DataFrame]:
        """
        Validate and process CSV file structure with enhanced NaN handling.
        
        :param df: DataFrame to validate and process
        :param file_path: Path to the CSV file for error reporting
        :param timestamp_col: Name of timestamp column (if None, will auto-detect)
        :param value_col: Name of value column (if None, will auto-detect)
        :return: Processed DataFrame or None if validation failed
        """
        # Check if DataFrame is empty
        if df.empty:
            logging.error(f"CSV file is empty: {file_path}")
            return None
        
        # Auto-detect columns if not provided
        if timestamp_col is None or value_col is None:
            detected_cols = ColumnDetector.detect_timestamp_and_value_columns(df, file_path)
            if detected_cols is None:
                return None
            timestamp_col, value_col = detected_cols
        
        # Enhanced NaN handling for timestamp column
        try:
            # Convert timestamp column to datetime
            if timestamp_col == 't':
                # Filter out invalid timestamps (negative values, NaN, inf)
                valid_mask = (df['t'] >= 0) & (df['t'].notna()) & (df['t'] != float('inf'))
                if not valid_mask.all():
                    invalid_count = (~valid_mask).sum()
                    logging.warning(f"Found {invalid_count} invalid timestamps in {file_path}, filtering them out")
                    df = df[valid_mask].copy()
                
                if df.empty:
                    logging.error(f"No valid timestamps found in {file_path}")
                    return None
                
                # Convert to datetime with error handling
                df['t'] = pd.to_datetime(df['t'], unit='s', errors='coerce')
                # Remove any remaining NaT values
                df = df.dropna(subset=['t'])
                
                if df.empty:
                    logging.error(f"No valid datetime values after conversion in {file_path}")
                    return None
            else:
                # If using 'timestamp' column, check if it's already datetime
                if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                    # Remove any NaT values
                    df = df.dropna(subset=[timestamp_col])
                    
                    if df.empty:
                        logging.error(f"No valid datetime values after conversion in {file_path}")
                        return None
        except Exception as e:
            logging.error(f"Failed to convert timestamp column in {file_path}: {e}")
            return None
        
        # Enhanced NaN handling for value column
        try:
            # Convert to numeric with coerce to handle non-numeric values
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            
            # Check for NaN values and handle them
            nan_count = df[value_col].isna().sum()
            if nan_count > 0:
                logging.warning(f"Found {nan_count} NaN values in value column of: {file_path}")
                
                # Option 1: Remove rows with NaN values
                df = df.dropna(subset=[value_col])
                logging.info(f"Removed {nan_count} rows with NaN values, remaining: {len(df)} rows")
                
                # Option 2: Alternative - forward fill NaN values (uncomment if preferred)
                # df[value_col] = df[value_col].fillna(method='ffill')
                # logging.info(f"Forward filled {nan_count} NaN values")
                
                # Option 3: Alternative - interpolate NaN values (uncomment if preferred)
                # df[value_col] = df[value_col].interpolate(method='linear')
                # logging.info(f"Interpolated {nan_count} NaN values")
            
            if df.empty:
                logging.error(f"No valid data remaining after NaN handling in {file_path}")
                return None
                
        except Exception as e:
            logging.error(f"Failed to convert value column to numeric in {file_path}: {e}")
            return None
        
        return df
    
    def _load_csv_file(self, file_path: str, clean_method: str = 'drop', 
                      interactive: bool = False, timestamp_col: str = None, value_col: str = None) -> Optional[pd.DataFrame]:
        """
        Load and validate a CSV file with enhanced NaN handling.
        
        :param file_path: Path to the CSV file
        :param clean_method: Method to clean NaN values ('drop', 'forward_fill', 'interpolate', etc.)
        :param interactive: Whether to use interactive column selection
        :param timestamp_col: Name of timestamp column (if None, will auto-detect)
        :param value_col: Name of value column (if None, will auto-detect)
        :return: DataFrame with standardized columns or None if failed
        """
        try:
            if not os.path.exists(file_path):
                logging.error(f"File does not exist: {file_path}")
                return None
            
            # Load CSV file
            df = pd.read_csv(file_path)
            logging.info(f"Loaded {len(df)} rows from: {file_path}")
            
            # Handle duplicate column names
            if len(df.columns) != len(set(df.columns)):
                logging.warning(f"Found duplicate column names in {file_path}, using first occurrence")
                # Keep only the first occurrence of each column name
                seen_columns = set()
                unique_columns = []
                for col in df.columns:
                    if col not in seen_columns:
                        seen_columns.add(col)
                        unique_columns.append(col)
                    else:
                        logging.warning(f"Removing duplicate column: {col}")
                df = df[unique_columns]
                logging.info(f"Columns after deduplication: {list(df.columns)}")
            
            # Validate and process structure
            if interactive:
                # Use interactive column selection
                selected_cols = ColumnDetector.interactive_column_selection(df, file_path)
                if selected_cols is None:
                    return None
                timestamp_col, value_col = selected_cols
                df = self._validate_and_process_csv_structure(df, file_path, timestamp_col, value_col)
            else:
                # Use auto-detection or provided columns
                df = self._validate_and_process_csv_structure(df, file_path, timestamp_col, value_col)
            
            if df is None:
                return None
            
            # Standardize column names - prefer 't' and 'v' over 'timestamp' and 'value'
            if 't' in df.columns and 'v' in df.columns:
                # Use 't' and 'v' columns, rename them
                # First, drop any existing 'timestamp' and 'value' columns to avoid duplicates
                if 'timestamp' in df.columns:
                    df = df.drop(columns=['timestamp'])
                if 'value' in df.columns:
                    df = df.drop(columns=['value'])
                
                # Now rename 't' and 'v' to 'timestamp' and 'value'
                df = df.rename(columns={'t': 'timestamp', 'v': 'value'})
                df = df[['timestamp', 'value']]
            elif 'timestamp' in df.columns and 'value' in df.columns:
                # Use existing 'timestamp' and 'value' columns
                df = df[['timestamp', 'value']]
            else:
                logging.error(f"Cannot find required columns in: {file_path}")
                return None
            
            # Set timestamp as index for better handling
            df = df.set_index('timestamp')
            
            # Remove duplicate timestamps (keep last occurrence)
            if df.index.duplicated().any():
                duplicate_count = df.index.duplicated().sum()
                logging.warning(f"Found {duplicate_count} duplicate timestamps in {file_path}, keeping last occurrence")
                df = df[~df.index.duplicated(keep='last')]
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Apply data quality validation
            validation = DataCleaner.validate_data_quality(df, value_col='value', timestamp_col='timestamp')
            if validation['recommendations']:
                logging.info(f"Data quality recommendations for {file_path}: {validation['recommendations']}")
            
            # Clean NaN values using specified method
            if validation['nan_analysis']['nan_count'] > 0:
                df = DataCleaner.clean_nan_values(df, value_col='value', method=clean_method)
            
            # Reset index to get timestamp back as column
            df = df.reset_index()
            
            logging.info(f"Successfully processed {len(df)} rows from: {file_path}")
            return df
            
        except Exception as e:
            logging.error(f"Failed to load CSV file {file_path}: {e}")
            return None
    
    def load_price_data(self, asset: str, clean_method: str = 'drop', 
                       interactive: bool = False, timestamp_col: str = None, value_col: str = None) -> Optional[pd.DataFrame]:
        """
        Load price data for a specific asset with NaN cleaning.
        Handles OHLC format where 'o' column contains JSON with 'c' (close) price.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param clean_method: Method to clean NaN values ('drop', 'forward_fill', 'interpolate', etc.)
        :param interactive: Whether to use interactive column selection
        :param timestamp_col: Name of timestamp column (if None, will auto-detect)
        :param value_col: Name of value column (if None, will auto-detect)
        :return: DataFrame with 'timestamp' and 'value' columns, or None if failed
        """
        price_file = os.path.join(self.base_path, asset, self.price_filename)
        
        logging.info(f"Loading price data for {asset} from: {price_file}")
        
        try:
            # Load raw CSV
            df = pd.read_csv(price_file)
            logging.info(f"Loaded {len(df)} rows from: {price_file}")
            
            # Check if this is OHLC format (has 'o' column with dict/JSON)
            if 'o' in df.columns and 't' in df.columns:
                logging.info("Detected OHLC format, extracting close price from dict/JSON")
                import ast
                import json
                
                # Extract close price from dict/JSON string in 'o' column
                def extract_close_price(ohlc_str):
                    try:
                        if pd.isna(ohlc_str):
                            return None
                        if isinstance(ohlc_str, str):
                            # Try ast.literal_eval first (for Python dict format)
                            try:
                                ohlc_dict = ast.literal_eval(ohlc_str)
                                return ohlc_dict.get('c', None)
                            except (ValueError, SyntaxError):
                                # Try JSON parsing (for JSON format)
                                try:
                                    ohlc_dict = json.loads(ohlc_str)
                                    return ohlc_dict.get('c', None)
                                except json.JSONDecodeError:
                                    return None
                        elif isinstance(ohlc_str, dict):
                            return ohlc_str.get('c', None)
                        return None
                    except (TypeError, AttributeError):
                        return None
                
                # Extract close prices
                df['value'] = df['o'].apply(extract_close_price)
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['t'], unit='s', errors='coerce')
                
                # Keep only timestamp and value columns
                df = df[['timestamp', 'value']].copy()
                
                # Remove rows with invalid data
                df = df.dropna(subset=['timestamp', 'value'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logging.info(f"Successfully processed OHLC price data: {len(df)} records")
            else:
                # Use standard loading method
                df = self._load_csv_file(price_file, clean_method=clean_method, 
                                       interactive=interactive, timestamp_col=timestamp_col, value_col=value_col)
            
            if df is not None and len(df) > 0:
                logging.info(f"Successfully loaded price data for {asset}: {len(df)} records")
                logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                logging.info(f"Price range: ${df['value'].min():.2f} to ${df['value'].max():.2f}")
            else:
                logging.error(f"Failed to load price data for {asset}")
                return None
            
        except Exception as e:
            logging.error(f"Error loading price data from {price_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return df
    
    def load_factor_data(self, asset: str, factor_name: str, clean_method: str = 'drop',
                        interactive: bool = False, timestamp_col: str = None, value_col: str = None) -> Optional[pd.DataFrame]:
        """
        Load factor data for a specific asset and factor with NaN cleaning.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param factor_name: Factor name (without .csv extension)
        :param clean_method: Method to clean NaN values ('drop', 'forward_fill', 'interpolate', etc.)
        :param interactive: Whether to use interactive column selection
        :param timestamp_col: Name of timestamp column (if None, will auto-detect)
        :param value_col: Name of value column (if None, will auto-detect)
        :return: DataFrame with 'timestamp' and 'value' columns, or None if failed
        """
        factor_file = os.path.join(self.base_path, asset, f"{factor_name}.csv")
        
        logging.info(f"Loading factor data for {asset}/{factor_name} from: {factor_file}")
        
        df = self._load_csv_file(factor_file, clean_method=clean_method,
                               interactive=interactive, timestamp_col=timestamp_col, value_col=value_col)
        
        if df is not None:
            logging.info(f"Successfully loaded factor data for {asset}/{factor_name}: {len(df)} records")
            logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logging.info(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        else:
            logging.error(f"Failed to load factor data for {asset}/{factor_name}")
        
        return df
    
    def load_factor_data_by_param(self, asset: str, param_name: str, factor_name: str = None, 
                                 clean_method: str = 'drop', interactive: bool = False, 
                                 timestamp_col: str = None, value_col: str = None) -> Optional[pd.DataFrame]:
        """
        Load factor data organized by parameter folders.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param param_name: Parameter name (folder name, e.g., 'sopr_account_based', 'nvt_ratio')
        :param factor_name: Specific factor file name (if None, will auto-detect)
        :param clean_method: Method to clean NaN values
        :param interactive: Whether to use interactive column selection
        :param timestamp_col: Name of timestamp column (if None, will auto-detect)
        :param value_col: Name of value column (if None, will auto-detect)
        :return: DataFrame with 'timestamp' and 'value' columns, or None if failed
        """
        # Build path: base_path/asset/param_name/factor_name.csv
        param_folder = os.path.join(self.base_path, asset, param_name)
        
        if factor_name is None:
            # Auto-detect factor file in param folder
            if os.path.exists(param_folder):
                csv_files = glob.glob(os.path.join(param_folder, "*.csv"))
                if csv_files:
                    # Use the first CSV file found
                    factor_file = csv_files[0]
                    factor_name = os.path.basename(factor_file).replace('.csv', '')
                    logging.info(f"Auto-detected factor file: {factor_name}")
                else:
                    logging.error(f"No CSV files found in parameter folder: {param_folder}")
                    return None
            else:
                logging.error(f"Parameter folder does not exist: {param_folder}")
                return None
        else:
            factor_file = os.path.join(param_folder, f"{factor_name}.csv")
        
        logging.info(f"Loading factor data for {asset}/{param_name}/{factor_name} from: {factor_file}")
        
        df = self._load_csv_file(factor_file, clean_method=clean_method,
                               interactive=interactive, timestamp_col=timestamp_col, value_col=value_col)
        
        if df is not None:
            logging.info(f"Successfully loaded factor data for {asset}/{param_name}/{factor_name}: {len(df)} records")
            logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logging.info(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        else:
            logging.error(f"Failed to load factor data for {asset}/{param_name}/{factor_name}")
        
        return df
    
    def get_available_params(self, asset: str) -> List[str]:
        """
        Get list of available parameter folders for an asset.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :return: List of parameter folder names
        """
        asset_path = os.path.join(self.base_path, asset)
        if not os.path.exists(asset_path):
            logging.warning(f"Asset path does not exist: {asset_path}")
            return []
        
        # Get all subdirectories (parameter folders)
        param_folders = []
        for item in os.listdir(asset_path):
            item_path = os.path.join(asset_path, item)
            if os.path.isdir(item_path):
                param_folders.append(item)
        
        logging.info(f"Available parameters for {asset}: {param_folders}")
        return param_folders
    
    def get_available_factors_in_param(self, asset: str, param_name: str) -> List[str]:
        """
        Get list of available factor files within a parameter folder.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param param_name: Parameter folder name
        :return: List of factor file names (without .csv extension)
        """
        param_folder = os.path.join(self.base_path, asset, param_name)
        if not os.path.exists(param_folder):
            logging.warning(f"Parameter folder does not exist: {param_folder}")
            return []
        
        # Get all CSV files in the parameter folder
        csv_files = glob.glob(os.path.join(param_folder, "*.csv"))
        factor_names = [os.path.basename(f).replace('.csv', '') for f in csv_files]
        
        logging.info(f"Available factors in {asset}/{param_name}: {factor_names}")
        return factor_names
    
    def create_param_folder_structure(self, asset: str, param_name: str) -> bool:
        """
        Create parameter folder structure for organizing factor data.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param param_name: Parameter folder name to create
        :return: True if successful, False otherwise
        """
        param_folder = os.path.join(self.base_path, asset, param_name)
        
        try:
            os.makedirs(param_folder, exist_ok=True)
            logging.info(f"Created parameter folder: {param_folder}")
            return True
        except Exception as e:
            logging.error(f"Failed to create parameter folder {param_folder}: {e}")
            return False
    
    def move_factor_to_param_folder(self, asset: str, old_factor_name: str, param_name: str, 
                                   new_factor_name: str = None) -> bool:
        """
        Move an existing factor file to a parameter folder structure.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param old_factor_name: Current factor file name (without .csv)
        :param param_name: Parameter folder name
        :param new_factor_name: New factor file name (if None, keeps original name)
        :return: True if successful, False otherwise
        """
        old_file = os.path.join(self.base_path, asset, f"{old_factor_name}.csv")
        if not os.path.exists(old_file):
            logging.error(f"Source factor file does not exist: {old_file}")
            return False
        
        # Create parameter folder
        if not self.create_param_folder_structure(asset, param_name):
            return False
        
        # Determine new file name
        if new_factor_name is None:
            new_factor_name = old_factor_name
        
        new_file = os.path.join(self.base_path, asset, param_name, f"{new_factor_name}.csv")
        
        try:
            import shutil
            shutil.move(old_file, new_file)
            logging.info(f"Moved {old_file} to {new_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to move factor file: {e}")
            return False
    
    def load_data_pair(self, asset: str, factor_name: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Load both price data and factor data for a specific asset and factor.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param factor_name: Factor name (without .csv extension)
        :return: Tuple of (price_data, factor_data) DataFrames, or None if failed
        """
        logging.info(f"Loading data pair for {asset}/{factor_name}")
        
        # Load price data
        price_data = self.load_price_data(asset)
        if price_data is None:
            logging.error(f"Cannot proceed without price data for {asset}")
            return None
        
        # Load factor data
        factor_data = self.load_factor_data(asset, factor_name)
        if factor_data is None:
            logging.error(f"Cannot proceed without factor data for {asset}/{factor_name}")
            return None
        
        # Validate date overlap
        price_start = price_data['timestamp'].min()
        price_end = price_data['timestamp'].max()
        factor_start = factor_data['timestamp'].min()
        factor_end = factor_data['timestamp'].max()
        
        overlap_start = max(price_start, factor_start)
        overlap_end = min(price_end, factor_end)
        
        if overlap_start >= overlap_end:
            logging.error(f"No date overlap between price and factor data for {asset}/{factor_name}")
            logging.error(f"Price data: {price_start} to {price_end}")
            logging.error(f"Factor data: {factor_start} to {factor_end}")
            return None
        
        logging.info(f"Data overlap: {overlap_start} to {overlap_end}")
        
        return price_data, factor_data
    
    def get_data_info(self, asset: str, factor_name: str = None) -> Dict:
        """
        Get information about available data for an asset.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param factor_name: Optional factor name to get specific info
        :return: Dictionary with data information
        """
        info = {
            'asset': asset,
            'base_path': self.base_path,
            'available_factors': self.get_available_factors(asset)
        }
        
        # Get price data info
        price_data = self.load_price_data(asset)
        if price_data is not None:
            info['price_data'] = {
                'records': len(price_data),
                'date_range': {
                    'start': price_data['timestamp'].min().strftime('%Y-%m-%d'),
                    'end': price_data['timestamp'].max().strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': float(price_data['value'].min()),
                    'max': float(price_data['value'].max())
                }
            }
        else:
            info['price_data'] = None
        
        # Get factor data info if specified
        if factor_name:
            factor_data = self.load_factor_data(asset, factor_name)
            if factor_data is not None:
                info['factor_data'] = {
                    'factor_name': factor_name,
                    'records': len(factor_data),
                    'date_range': {
                        'start': factor_data['timestamp'].min().strftime('%Y-%m-%d'),
                        'end': factor_data['timestamp'].max().strftime('%Y-%m-%d')
                    },
                    'value_range': {
                        'min': float(factor_data['value'].min()),
                        'max': float(factor_data['value'].max())
                    }
                }
            else:
                info['factor_data'] = None
        
        return info
    
    def preview_and_load_data(self, asset: str, factor_name: str = None, 
                             interactive: bool = True, clean_method: str = 'drop') -> Optional[pd.DataFrame]:
        """
        Preview data structure and load data with optional interactive column selection.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param factor_name: Factor name (if None, loads price data)
        :param interactive: Whether to use interactive column selection
        :param clean_method: Method to clean NaN values
        :return: DataFrame with 'timestamp' and 'value' columns, or None if failed
        """
        if factor_name is None:
            # Load price data
            return self.load_price_data(asset, clean_method=clean_method, interactive=interactive)
        else:
            # Load factor data
            return self.load_factor_data(asset, factor_name, clean_method=clean_method, interactive=interactive)
    
    def get_data_preview(self, asset: str, factor_name: str = None) -> Optional[pd.DataFrame]:
        """
        Get a preview of data structure without loading the full dataset.
        
        :param asset: Asset symbol (e.g., 'BTC')
        :param factor_name: Factor name (if None, previews price data)
        :return: DataFrame with first 5 and last 5 rows, or None if failed
        """
        if factor_name is None:
            file_path = os.path.join(self.base_path, asset, self.price_filename)
        else:
            file_path = os.path.join(self.base_path, asset, f"{factor_name}.csv")
        
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return None
        
        try:
            # Load just a sample of the data
            df = pd.read_csv(file_path, nrows=10)
            print(f"\n=== Data Structure Preview for {file_path} ===")
            print(f"Columns: {list(df.columns)}")
            print(f"Data types: {df.dtypes.to_dict()}")
            print(f"Shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head().to_string())
            print("\nLast 5 rows:")
            print(df.tail().to_string())
            print("=" * 80)
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to preview data from {file_path}: {e}")
            return None


def main():
    """Test the local data loader with enhanced features"""
    loader = LocalDataLoader()
    
    # Test available assets
    assets = loader.get_available_assets()
    print(f"Available assets: {assets}")
    
    if assets:
        # Test with first available asset
        asset = assets[0]
        print(f"\nTesting with asset: {asset}")
        
        # Get available factors
        factors = loader.get_available_factors(asset)
        print(f"Available factors: {factors}")
        
        # Test data preview functionality
        print(f"\n=== Testing Data Preview ===")
        preview_df = loader.get_data_preview(asset)
        if preview_df is not None:
            print("Data preview successful")
        
        if factors:
            # Test factor data preview
            factor = factors[0]
            print(f"\n=== Testing Factor Data Preview for {factor} ===")
            factor_preview = loader.get_data_preview(asset, factor)
            if factor_preview is not None:
                print("Factor data preview successful")
        
        # Test interactive loading (commented out to avoid blocking in automated tests)
        print(f"\n=== Testing Interactive Loading (commented out) ===")
        print("To test interactive loading, uncomment the following lines:")
        print("# data = loader.preview_and_load_data(asset, interactive=True)")
        print("# if data is not None:")
        print("#     print(f'Successfully loaded {len(data)} records')")
        
        # Test automatic loading with different cleaning methods
        print(f"\n=== Testing Automatic Loading ===")
        try:
            price_data = loader.load_price_data(asset, clean_method='drop')
            if price_data is not None:
                print(f"Successfully loaded price data: {len(price_data)} records")
            
            if factors:
                factor_data = loader.load_factor_data(asset, factors[0], clean_method='interpolate')
                if factor_data is not None:
                    print(f"Successfully loaded factor data: {len(factor_data)} records")
        except Exception as e:
            print(f"Error during automatic loading: {e}")
        
        # Get data info
        info = loader.get_data_info(asset)
        print(f"\nData info for {asset}:")
        print(f"Price data: {info.get('price_data', 'Not available')}")
        print(f"Available factors: {info.get('available_factors', [])}")
        
        if factors:
            # Test loading data pair
            factor = factors[0]
            print(f"\nTesting data pair loading for {asset}/{factor}")
            
            data_pair = loader.load_data_pair(asset, factor)
            if data_pair:
                price_data, factor_data = data_pair
                print(f"Successfully loaded data pair:")
                print(f"  Price data: {len(price_data)} records")
                print(f"  Factor data: {len(factor_data)} records")
            else:
                print("Failed to load data pair")


if __name__ == "__main__":
    main() 