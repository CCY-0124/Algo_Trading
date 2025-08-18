"""
dataloader.py

Provides a DataLoader class to retrieve time-series data from the Glassnode API.

:precondition: A valid API key is required. Output is not cached locally.
:postcondition: Time-series metric data is returned directly from the API.
"""

import pandas as pd
import requests
import os
from datetime import datetime
from typing import Optional
import logging
import time


class DataLoader:
    """
    A data loader that fetches time-series metric data for crypto assets from the Glassnode API.

    :param api_key: API key for accessing the Glassnode API
    :precondition: A valid API key must be provided
    :postcondition: Data is retrieved from the API and returned as a DataFrame
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = 'https://api.glassnode.com/v1/metrics/'
        logging.basicConfig(level=logging.INFO)

    def _get_from_api(self, asset: str, metric: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch metric data from the Glassnode API.

        :param asset: The crypto asset symbol
        :param metric: The metric name
        :param start: Start date in 'YYYY-MM-DD' format
        :param end: End date in 'YYYY-MM-DD' format
        :precondition: A valid API key must be set in the DataLoader instance
        :postcondition: Returns a DataFrame with timestamp and value if API call is successful
        :return: A pandas DataFrame or None if the API call fails
        """
        if not self.api_key:
            logging.error("API key is required for API access")
            return None

        # Convert dates to Unix timestamps
        start_timestamp = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end, '%Y-%m-%d').timestamp())
        
        params = {
            'api_key': self.api_key,
            'a': asset,
            'i': '24h',
            's': start_timestamp,
            'u': end_timestamp,
            'f': 'json'
        }

        try:
            response = requests.get(f"{self.base_url}{metric}", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                logging.warning(f"No data returned for {asset}/{metric}")
                return None

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df['value'] = df['v']
            return df[['timestamp', 'value']]
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    def get_data(self, asset: str, metric: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve data from the Glassnode API.

        :param asset: The crypto asset symbol
        :param metric: The metric name
        :param start: Start date in 'YYYY-MM-DD' format
        :param end: End date in 'YYYY-MM-DD' format
        :precondition: API key must be valid
        :postcondition: Returns a DataFrame of time-series data
        :return: A pandas DataFrame with 'timestamp' and 'value' columns
        """
        api_data = self._get_from_api(asset, metric, start, end)
        if api_data is None or api_data.empty:
            raise ValueError(f"Failed to fetch data for {asset}/{metric}")

        logging.info(f"Fetched {len(api_data)} records from API for {asset}/{metric}")
        return api_data

    def get_bulk_data(self, assets: list, metrics: list, start: str, end: str) -> dict:
        """
        Retrieve multiple metrics for multiple assets in bulk.

        :param assets: A list of crypto asset symbols
        :param metrics: A list of metric names
        :param start: Start date in 'YYYY-MM-DD' format
        :param end: End date in 'YYYY-MM-DD' format
        :precondition: The assets and metrics lists must be non-empty
        :postcondition: Each combination of asset and metric is fetched and organized in a nested dictionary
        :return: A nested dictionary in the form {asset: {metric: DataFrame}}
        """
        results = {}
        for asset in assets:
            results[asset] = {}
            for metric in metrics:
                try:
                    results[asset][metric] = self.get_data(asset, metric, start, end)
                except Exception as e:
                    logging.error(f"Error fetching {asset}/{metric}: {str(e)}")
                time.sleep(0.5)  # Respect API rate limits
        return results
