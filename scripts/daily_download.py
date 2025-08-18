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
from datetime import datetime
import logging
import sys
from time import sleep
import glob
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from config.secrets import get_api_key  # Use your secrets manager

# Basic settings
BASE_PATH = r"D:\Trading_Data\glassnode_data2"
CURRENT_API_KEY_TYPE = "main"  # Track which key type we're using
BASE_URL = "https://api.glassnode.com/v1"

# Logs settings
LOG_FILE = os.path.join(BASE_PATH, "daily_download.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)


def get_current_api_key():
    """
    Retrieve the currently active Glassnode API key.

    :precondition: The API key must be managed using the secrets module
    :postcondition: Returns the key as a string, or logs an error if retrieval fails
    :return: API key string or None
    """
    global CURRENT_API_KEY_TYPE
    try:
        # Placeholder implementation - replace with actual API key management
        api_key = os.getenv("GLASSNODE_API_KEY", "YOUR_GLASSNODE_API_KEY")
        if api_key == "YOUR_GLASSNODE_API_KEY":
            logging.warning("Using placeholder API key. Set GLASSNODE_API_KEY in .env file")
        return api_key
    except Exception as e:
        logging.error(f"Failed to get API key: {str(e)}")
        return None


def switch_api_key():
    """
    Switch between the main and backup Glassnode API keys.

    :precondition: `CURRENT_API_KEY_TYPE` must be either 'main' or 'backup'
    :postcondition: The global API key type is switched and a delay is introduced to avoid rate limit
    :return: True after switching
    """
    global CURRENT_API_KEY_TYPE
    if CURRENT_API_KEY_TYPE == "main":
        CURRENT_API_KEY_TYPE = "backup"
        logging.info("Switched to backup API key")
    else:
        CURRENT_API_KEY_TYPE = "main"
        logging.info("Switched back to main API key")
    sleep(10)
    return True


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

            # Read CSV files
            df = pd.read_csv(file_path)

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

        # Read metrics_info CSV
        df = pd.read_csv(metrics_file)

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

        elif response.status_code in [429, 403]:
            logging.warning(f"API key limit reached - Status code: {response.status_code}")
            switch_api_key()
            return None
        else:
            # Mask API key in error message for security
            masked_params = str(params).replace(api_key, api_key[:8] + "***")
            logging.error(f"API request failed - Status code: {response.status_code}, response: {response.text[:200]}, params: {masked_params}")
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
                # Read existing data
                df = pd.read_csv(file_path)

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

        logging.info(f"Starting update from {file_path}")

        # Download new data
        new_data = update_metric_data(asset, metric_path, last_timestamp, min_resolution)

        if new_data is not None and not new_data.empty:
            before = len(df)
            updated_data = write_data_immediately(file_path, new_data, df)
            if updated_data is not None:
                after = len(updated_data)
                added_rows = after - before
                logging.info(f"Added new rows to {file_path}")
                sleep(1)  # API limit
                return added_rows
            else:
                logging.error(f"Failed to write data: {file_path}")
                return 0
        else:
            logging.info(f"No new data: {file_path}")
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
    logging.info(f"Found {len(valid_combinations)} valid combinations")

    updated_files = 0
    updated_rows = 0

    for asset, metric_path, min_resolution, tier in valid_combinations:
        try:
            # Generate file name
            filename_parts = metric_path.strip('/').split('/')
            base_filename = f"_{'_'.join(filename_parts)}_tier{tier}.csv"

            # Build file path
            asset_dir = os.path.join(BASE_PATH, asset)
            file_path = os.path.join(asset_dir, base_filename)

            # Update data
            new_rows = update_single_file(file_path, asset, metric_path, min_resolution, tier)
            if new_rows > 0:
                updated_files += 1
                updated_rows += new_rows

        except Exception as e:
            logging.error(f"Error updating {asset}/{metric_path} : {str(e)}")
            continue

    logging.info(f"\n=== Update complete ===")
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