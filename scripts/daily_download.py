"""
daily_download.py

This script downloads Glassnode metric data daily using the API and updates local CSV files.
It supports automatic API key switching, intelligent file updating, and per-asset metric configuration.

:precondition: The user must have valid Glassnode API keys stored using secrets manager
:postcondition: Data is updated into structured CSV files under the configured BASE_PATH
"""
import os
import requests
import json
import pandas as pd
from datetime import datetime, timezone
import logging
import sys
from time import sleep
import glob
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.secrets import get_api_key  # Use your secrets manager

# Import rate limit configuration
try:
    from config.download_config import RATE_LIMIT_CONFIG
    MIN_DELAY = RATE_LIMIT_CONFIG.get('min_delay_between_requests', 3.0)
    REQUESTS_PER_MINUTE = RATE_LIMIT_CONFIG.get('requests_per_minute', 20)
except ImportError:
    # Fallback if config not available
    MIN_DELAY = 3.0
    REQUESTS_PER_MINUTE = 20

# Basic settings
BASE_PATH = r"D:\Trading_Data\glassnode_data2"
BASE_URL = "https://api.glassnode.com/v1"

# Track skip counts for each file to implement incremental thresholds
# Format: {file_path: skip_count}
file_skip_counts = {}

# Logs settings
LOG_FILE = os.path.join(BASE_PATH, "daily_download.log")
ERROR_LOG_FILE = os.path.join(BASE_PATH, "api_errors.csv")
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG to see skip check details
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)


def safe_read_csv(file_path):
    """
    Safely read CSV file with error handling for encoding and parsing issues.
    
    :param file_path: Path to the CSV file
    :return: DataFrame or None if failed
    """
    try:
        # Try UTF-8 first
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Try with different encodings
            return pd.read_csv(file_path, encoding='latin-1')
        except Exception as e:
            logging.error(f"Failed to read {file_path} with latin-1 encoding: {str(e)}")
            return None
    except pd.errors.ParserError as e:
        # Handle parsing errors (inconsistent field counts)
        logging.warning(f"Parser error in {file_path}, trying with error handling: {str(e)}")
        try:
            return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except Exception as e2:
            logging.error(f"Failed to read {file_path} even with error handling: {str(e2)}")
            return None
    except Exception as e:
        logging.error(f"Unexpected error reading {file_path}: {str(e)}")
        return None


def validate_csv_data(df, file_path):
    """
    Validate CSV data structure and content.
    
    :param df: DataFrame to validate
    :param file_path: Path to the file for error reporting
    :return: True if valid, False otherwise
    """
    if df is None or df.empty:
        logging.error(f"DataFrame is empty or None for {file_path}")
        return False
    
    # Check for required columns
    if len(df.columns) == 0:
        logging.error(f"No columns found in {file_path}")
        return False
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        logging.warning(f"Duplicate column names found in {file_path}, cleaning...")
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
    
    # Check for completely empty rows
    if df.isnull().all(axis=1).any():
        empty_rows = df.isnull().all(axis=1).sum()
        logging.warning(f"Found {empty_rows} completely empty rows in {file_path}, removing...")
        df.dropna(how='all', inplace=True)
    
    return True


def get_current_api_key():
    """
    Retrieve the main Glassnode API key.

    :precondition: The API key must be managed using the secrets module
    :postcondition: Returns the key as a string, or logs an error if retrieval fails
    :return: API key string or None
    """
    try:
        # Use the secrets module to get main API key only
        api_key = get_api_key("glassnode", "main")
        if not api_key:
            logging.error("Failed to retrieve main API key from secrets")
            return None
        return api_key
    except Exception as e:
        logging.error(f"Failed to get API key: {str(e)}")
        return None


def save_error_to_file(asset, metric_path, status_code, response_body, response_headers, url, params, error_type=''):
    """
    Save error details to CSV file for later review.
    
    :param asset: Asset symbol
    :param metric_path: Metric path
    :param status_code: HTTP status code
    :param response_body: Response body text
    :param response_headers: Response headers dict
    :param url: Request URL
    :param params: Request parameters (with masked API key)
    :param error_type: Type of error (404, 403, etc.)
    :return: None
    """
    try:
        import csv
        from datetime import datetime
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(ERROR_LOG_FILE)
        
        # Prepare error record
        error_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'asset': asset,
            'metric_path': metric_path,
            'status_code': status_code,
            'error_type': error_type,
            'response_body': response_body.replace('\n', ' ').replace('\r', ' ')[:500],  # Limit length and remove newlines
            'url': url,
            'params': str(params).replace('\n', ' ')[:200],  # Limit length
            'rate_limit_remaining': response_headers.get('X-Rate-Limit-Remaining', 'N/A'),
            'rate_limit_reset': response_headers.get('X-Rate-Limit-Reset', 'N/A'),
        }
        
        # Write to CSV file
        with open(ERROR_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'asset', 'metric_path', 'status_code', 'error_type', 
                         'response_body', 'url', 'params', 'rate_limit_remaining', 'rate_limit_reset']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(error_record)
            
    except Exception as e:
        logging.error(f"Failed to save error to file: {str(e)}")


def handle_rate_limit():
    """
    Handle rate limit by waiting before retrying.

    :precondition: Rate limit error has occurred
    :postcondition: Waits for a delay to avoid rate limit
    :return: None
    """
    try:
        from config.download_config import RATE_LIMIT_CONFIG
        wait_time = RATE_LIMIT_CONFIG.get('max_delay_between_requests', 300.0)
    except ImportError:
        wait_time = 300.0
    
    logging.warning(f"API rate limit reached, waiting {wait_time} seconds ({wait_time/60:.1f} minutes) before retry...")
    sleep(wait_time)


def load_metrics_info():
    """
    Load all metrics_info_*.csv files under BASE_PATH and extract metric configurations.

    :precondition: CSV files must follow the naming convention metrics_info_{asset}.csv
    :postcondition: Builds a nested dictionary containing metric details for each asset
    :return: Dictionary structured as {asset: {metric_path: {min_resolution, supported_assets, tier}}}
    """
    logging.info("Loading metrics_info file...")
    metrics_info = {}

    # Find all metrics_info_*.csv file
    pattern = os.path.join(BASE_PATH, "metrics_info_*.csv")
    metrics_files = glob.glob(pattern)

    if not metrics_files:
        logging.error(f"Error: No metrics_info CSV file found in BASE_PATH")
        return {}

    logging.info(f"found {len(metrics_files)} metrics_info file")

    for file_path in metrics_files:
        try:
            # Extract asset name from file name
            filename = os.path.basename(file_path)
            asset = filename.replace('metrics_info_', '').replace('.csv', '').upper()

            logging.info(f"Processing metric info for {asset}...")

            # Read CSV files with error handling
            df = safe_read_csv(file_path)
            if df is None:
                continue
            
            # Validate the data
            if not validate_csv_data(df, file_path):
                continue

            # Check if required columns exist
            required_columns = ['path', 'min_resolution', 'supported_assets']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logging.error(f"File {filename} is missing required columns: {', '.join(missing_columns)}")
                continue

            asset_metrics = {}
            for _, row in df.iterrows():
                metric_path = row['path']
                min_resolution = row['min_resolution']
                supported_assets_str = row['supported_assets']
                tier = row.get('tier', 1)  # Default tier is 1

                # Parse supported assets list
                if pd.isna(supported_assets_str) or supported_assets_str == '':
                    supported_assets = [asset]
                else:
                    # Strip quotes and split
                    supported_assets_str = str(supported_assets_str).strip('"')
                    supported_assets = [s.strip().upper() for s in supported_assets_str.split(',')]

                asset_metrics[metric_path] = {
                    'min_resolution': min_resolution,
                    'supported_assets': supported_assets,
                    'tier': int(tier)  # Ensure tier is an integer
                }

            metrics_info[asset] = asset_metrics
            logging.info(f"  {asset}: {len(asset_metrics)} metrics")

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            continue

    return metrics_info


def get_valid_combinations(metrics_info, target_assets=None):
    """
    Extract valid asset-metric combinations from metrics_info.

    :param metrics_info: Nested dictionary of metrics by asset
    :param target_assets: Optional list of asset symbols to filter
    :precondition: metrics_info must be a properly structured dictionary
    :postcondition: Filters and returns supported combinations
    :return: List of tuples (asset, metric_path, min_resolution, tier)
    """
    valid_combinations = []

    for asset, asset_metrics in metrics_info.items():
        if target_assets and asset not in target_assets:
            continue

        for metric_path, metric_info in asset_metrics.items():
            supported_assets = metric_info['supported_assets']
            min_resolution = metric_info['min_resolution']
            tier = metric_info['tier']

            # Check if asset is in supported list
            if asset in supported_assets:
                valid_combinations.append((asset, metric_path, min_resolution, tier))

    return valid_combinations


def update_metrics_info_last_update(asset, metric_path, current_time):
    """
    Update the 'last_update' field in the corresponding metrics_info CSV.

    :param asset: Asset symbol (e.g., BTC)
    :param metric_path: Metric path as used in the CSV and API
    :param current_time: Timestamp string to write as last_update
    :precondition: The file must be writable and have a 'path' column
    :postcondition: Updates the last_update value for the specified metric
    :return: None
    """
    try:
        metrics_file = os.path.join(BASE_PATH, f"metrics_info_{asset.lower()}.csv")
        if not os.path.exists(metrics_file):
            return

        # Read metrics_info CSV with error handling
        df = safe_read_csv(metrics_file)
        if df is None:
            return
        
        # Validate the data
        if not validate_csv_data(df, metrics_file):
            return

        # Create 'last_update' column if missing
        if 'last_update' not in df.columns:
            df['last_update'] = pd.NaT

        # Locate matching row and update last_update
        mask = df['path'] == metric_path
        if mask.any():
            df.loc[mask, 'last_update'] = current_time

            # Write back to file
            df.to_csv(metrics_file, index=False)
            logging.info(f"Updated last_update for {asset}")

    except Exception as e:
        logging.error(f"Failed to update metrics_info last_update: {str(e)}")


def write_data_immediately(file_path, new_data, existing_data=None):
    """
    Immediately write new metric data into a CSV file, merging with existing data if available.

    :param file_path: File path to write data
    :param new_data: New data as DataFrame
    :param existing_data: Existing data DataFrame, if any
    :precondition: Time columns must exist and be in seconds
    :postcondition: Merges and writes updated data to file
    :return: The combined DataFrame or None if an error occurs
    """
    try:
        # Write new data directly if no existing data
        if existing_data is None or existing_data.empty:
            new_data.to_csv(file_path, index=False)
            return new_data

        # Merge new and existing data
        # Determine timestamp column name
        timestamp_col = 't' if 't' in existing_data.columns else 'timestamp'

        # Ensure new data contains timestamp column
        if timestamp_col not in new_data.columns:
            logging.error(f"New data is missing timestamp column {timestamp_col}")
            return None

        # Convert timestamps to datetime for sorting
        existing_data[timestamp_col] = pd.to_datetime(existing_data[timestamp_col], unit='s')
        new_data[timestamp_col] = pd.to_datetime(new_data[timestamp_col], unit='s')

        # Merge data and remove duplicates
        combined_data = pd.concat([existing_data, new_data])
        combined_data.drop_duplicates(subset=[timestamp_col], inplace=True)
        combined_data.sort_values(by=timestamp_col, inplace=True)

        # Convert back to Unix timestamp
        combined_data[timestamp_col] = combined_data[timestamp_col].astype('int64') // 10 ** 9

        # Write data to file immediately
        combined_data.to_csv(file_path, index=False)
        return combined_data

    except Exception as e:
        logging.error(f"Failed to write data immediately: {str(e)}")
        return None


def update_metric_data(asset, metric_path, last_timestamp, resolution):
    """
    Fetch new metric data for an asset from Glassnode API starting from last known timestamp.

    :param asset: The crypto asset (e.g., 'BTC')
    :param metric_path: The Glassnode metric path (e.g., '/market/price_usd_close')
    :param last_timestamp: Last known timestamp (int or ISO string)
    :param resolution: Data resolution (e.g., '24h')
    :precondition: API key must be valid, metric must exist
    :postcondition: Data is returned or logged error
    :return: A DataFrame with new data, or None if error
    """
    url = f"{BASE_URL}/metrics{metric_path}"
    current_time = int(datetime.now().timestamp())
    api_key = get_current_api_key()

    if not api_key:
        logging.error("Unable to get a valid API key")
        return None

    # Ensure last_timestamp is an integer
    try:
        if isinstance(last_timestamp, str):
            last_timestamp = int(pd.to_datetime(last_timestamp).timestamp())
        else:
            last_timestamp = int(last_timestamp)
    except Exception as e:
        logging.error(f"Timestamp conversion failed - {last_timestamp}: {str(e)}")
        return None

    # Avoid duplicate data by starting request from last_timestamp + 1 sec
    start_time = last_timestamp + 1

    params = {
        'a': asset,
        'i': resolution,
        'f': 'json',
        's': start_time,
        'u': current_time,
        'timestamp_format': 'unix',
        'api_key': api_key
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Check if response contains data
            if response.text.strip() == '[]':
                logging.info(f"No new data - Asset: {asset}, Metric: {metric_path}")
                return pd.DataFrame()

            new_data = pd.DataFrame(response.json())

            if not new_data.empty:
                # Ensure timestamp column exists
                if 't' not in new_data.columns:
                    logging.error(f"API response missing timestamp column: {response.text[:100]}...")
                    return None

                # Update last_update in metrics_info
                update_metrics_info_last_update(asset, metric_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                return new_data
            return pd.DataFrame()

        else:
            # All errors (404, 403, 429, 500, etc.) - log details and skip (don't retry in this run)
            masked_params = str(params).replace(api_key, api_key[:8] + "***")
            
            # Determine error type
            if response.status_code == 404:
                error_type = "Endpoint Not Found"
                error_msg = "This endpoint may have been removed by Glassnode."
            elif response.status_code == 403:
                error_type = "Forbidden/Access Denied"
                error_msg = "Rate limit or access denied. May require Studio subscription."
            elif response.status_code == 429:
                error_type = "Rate Limit"
                error_msg = "Rate limit exceeded."
            else:
                error_type = f"Error {response.status_code}"
                error_msg = "Unexpected error."
            
            logging.warning(f"API request failed - Status code: {response.status_code} - Asset: {asset}, Metric: {metric_path}")
            logging.warning(f"URL: {url}")
            logging.warning(f"Params: {masked_params}")
            logging.warning(f"Response Headers: {dict(response.headers)}")
            logging.warning(f"Response Body: {response.text}")
            logging.warning(f"{error_msg} Will skip this metric in this run.")
            logging.warning(f"Skipping this metric to avoid wasting API quota.")
            
            # Save error to CSV file for later review
            save_error_to_file(
                asset=asset,
                metric_path=metric_path,
                status_code=response.status_code,
                response_body=response.text,
                response_headers=dict(response.headers),
                url=url,
                params=masked_params,
                error_type=error_type
            )
            
            return None

    except Exception as e:
        logging.error(f"Failed to fetch data - Asset: {asset}, Metric: {metric_path}: {str(e)}")
        return None


def update_single_file(file_path, asset, metric_path, min_resolution, tier):
    """
    Update the data for a specific asset-metric file.

    :param file_path: Full path to the target CSV file
    :param asset: Asset symbol (e.g., BTC)
    :param metric_path: Metric endpoint path (e.g., /market/price_usd_close)
    :param min_resolution: Resolution string (e.g., '24h')
    :param tier: Tier level for the metric
    :precondition: The metric must be supported and the file must be writable
    :postcondition: Appends new data to the file if available
    :return: Number of rows added (int)
    """
    try:
        # Check if file exists
        file_exists = os.path.exists(file_path)
        df = pd.DataFrame()
        last_timestamp = 0  # Default to download from beginning

        if file_exists:
            try:
                # Read existing data with error handling
                df = safe_read_csv(file_path)
                if df is None:
                    return 0
                
                # Validate the data
                if not validate_csv_data(df, file_path):
                    return 0

                # Fix duplicate or malformed column names
                df.columns = [str(col).strip() for col in df.columns]
                if len(set(df.columns)) != len(df.columns):
                    logging.error(f"Duplicate or invalid column names: {file_path}")
                    return 0

                # Check for timestamp column
                timestamp_col = 't' if 't' in df.columns else 'timestamp'
                if timestamp_col not in df.columns:
                    logging.error(f"Timestamp column not found: {file_path}")
                    return 0

                # Convert timestamps to numeric values
                df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
                if df[timestamp_col].dropna().empty:
                    logging.warning(f"All timestamps are invalid: {file_path}")
                    last_timestamp = 0
                else:
                    last_timestamp = df[timestamp_col].max()

            except Exception as e:
                logging.error(f"Failed to read file: {file_path}, Error: {str(e)}")
                return 0
        else:
            logging.info(f"File not found, creating new one: {file_path}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Check if data is already up to date to avoid unnecessary API calls
        # Uses UTC time for comparison and different thresholds based on resolution
        if file_exists and last_timestamp > 0:
            # Use UTC time for comparison (timezone-aware)
            current_time_utc = int(datetime.now(timezone.utc).timestamp())
            
            # Handle timestamp format: if timestamp is in milliseconds, convert to seconds
            if last_timestamp > 1e10:  # Likely in milliseconds (timestamp > year 2286)
                last_timestamp = last_timestamp / 1000
            
            last_timestamp_int = int(last_timestamp)
            time_since_last = current_time_utc - last_timestamp_int
            
            # Get skip count for this file (default to 0 for first check)
            skip_count = file_skip_counts.get(file_path, 0)
            
            # All resolutions check if within 2 days in UTC time
            threshold_seconds = 172800  # 2 days (172800 seconds) in UTC
            threshold_description = "2 days (UTC)"
            
            # Log skip check details (using UTC time)
            minutes_ago = time_since_last // 60
            hours_ago = minutes_ago // 60
            days_ago = hours_ago // 24
            seconds_ago = time_since_last % 60
            last_update_time_utc = datetime.fromtimestamp(last_timestamp_int, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            current_time_utc_str = datetime.fromtimestamp(current_time_utc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            
            if days_ago > 0:
                time_ago_str = f"{days_ago}d {hours_ago % 24}h {minutes_ago % 60}m"
            elif hours_ago > 0:
                time_ago_str = f"{hours_ago}h {minutes_ago % 60}m {seconds_ago}s"
            else:
                time_ago_str = f"{minutes_ago}m {seconds_ago}s"
            
            logging.info(f"Skip check for {asset}/{metric_path}: last_update={last_update_time_utc}, current={current_time_utc_str}, time_since_last={time_ago_str} ({time_since_last}s), threshold={threshold_description}, resolution={min_resolution}")
            
            if time_since_last < threshold_seconds:
                logging.info(f"Data is up to date (updated {time_ago_str} ago, threshold: {threshold_description}), skipping API call for {asset}/{metric_path}")
                return 0
            else:
                # Reset skip count if data is older than threshold (data needs update)
                if file_path in file_skip_counts:
                    file_skip_counts[file_path] = 0
                logging.info(f"Data needs update: time_since_last={time_ago_str} ({time_since_last}s) >= threshold={threshold_description} ({threshold_seconds}s) for {asset}/{metric_path}")
        else:
            if not file_exists:
                logging.info(f"File does not exist, will create new: {file_path}")
            elif last_timestamp == 0:
                logging.info(f"File exists but last_timestamp is 0, will download from beginning: {file_path}")

        logging.info(f"Starting update from {file_path}")

        # Download new data
        new_data = update_metric_data(asset, metric_path, last_timestamp, min_resolution)

        if new_data is not None and not new_data.empty:
            before = len(df)
            updated_data = write_data_immediately(file_path, new_data, df)
            if updated_data is not None:
                after = len(updated_data)
                added_rows = after - before
                # Reset skip count since data was successfully updated
                if file_path in file_skip_counts:
                    file_skip_counts[file_path] = 0
                logging.info(f"Added {added_rows} new rows to {file_path}")
                sleep(MIN_DELAY)  # Respect API rate limit from config
                return added_rows
            else:
                logging.error(f"Failed to write data: {file_path}")
                return 0
        else:
            # API returned no new data, but we still made the call
            # Reset skip count to 0 so next time we check again (but with 5min threshold)
            if file_path in file_skip_counts:
                file_skip_counts[file_path] = 0
            logging.info(f"No new data from API: {file_path}")
            return 0

    except Exception as e:
        logging.error(f"Error occurred while updating file: {file_path}, Error: {str(e)}")
        return 0


def update_from_metrics_info():
    """
    Update data files for all valid asset-metric combinations listed in metrics_info CSV files.

    :precondition: metrics_info CSV files must exist in BASE_PATH and be formatted correctly
    :postcondition: New data rows are downloaded and appended to corresponding files
    :return: None
    """
    logging.info("\nStarting update from metrics_info files...")

    # Load metrics info
    metrics_info = load_metrics_info()
    if not metrics_info:
        logging.error("Cant Load metrics info, exit")
        return

    # Get valid combinations
    valid_combinations = get_valid_combinations(metrics_info)
    total_combinations = len(valid_combinations)
    logging.info(f"Found {total_combinations} valid combinations")

    updated_files = 0
    updated_rows = 0
    processed = 0

    for asset, metric_path, min_resolution, tier in valid_combinations:
        try:
            processed += 1
            percentage = (processed / total_combinations) * 100
            
            # Generate file name - remove leading underscore and duplicate parts
            filename_parts = metric_path.strip('/').split('/')
            # Remove any empty parts and join
            filename_parts = [part for part in filename_parts if part]
            base_filename = f"{'_'.join(filename_parts)}_tier{tier}.csv"

            # Build file path
            asset_dir = os.path.join(BASE_PATH, asset)
            file_path = os.path.join(asset_dir, base_filename)

            # Log progress with percentage
            logging.info(f"[{processed}/{total_combinations}] ({percentage:.1f}%) Processing: {asset}/{metric_path}")

            # Update data
            new_rows = update_single_file(file_path, asset, metric_path, min_resolution, tier)
            if new_rows > 0:
                updated_files += 1
                updated_rows += new_rows
                logging.info(f"  -> Added {new_rows} rows (Progress: {percentage:.1f}%)")
            else:
                logging.info(f"  -> No new data (Progress: {percentage:.1f}%)")

        except Exception as e:
            logging.error(f"Error updating {asset}/{metric_path} : {str(e)} (Progress: {percentage:.1f}%)")
            continue

    logging.info(f"\n=== Update complete ===")
    logging.info(f"Total processed: {processed}/{total_combinations} (100.0%)")
    logging.info(f"Number of files updated: {updated_files}")
    logging.info(f"Number of rows added: {updated_rows}")


def main():
    """
    Main program entry point to start the Glassnode data update process.

    :precondition: Secrets and metrics_info CSVs must be set up
    :postcondition: Executes full data update process and logs execution time and errors
    :return: None
    """
    start_time = datetime.now()
    logging.info(f"\nStart execution time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        update_from_metrics_info()

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\nUpdate finished！")
        logging.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total runtime: {duration}")

    except Exception as e:
        logging.error(f"Runtime error: {str(e)}")

if __name__ == "__main__":
    try:
        logging.info("\n=== Glassnode data update program started ===\n")
        main()
    except Exception as e:
        logging.error(f"Runtime error: {str(e)}")
    finally:
        logging.info("\n=== Program finished ===\n")