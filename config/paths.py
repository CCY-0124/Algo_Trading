"""
paths.py

This module provides utility functions for locating key directory
paths for trading-related data, including raw data folders.
"""

import os
import warnings


def get_data_path():
    """
    Get the root directory for trading data on a Windows system.

    :precondition: Trading data directories may or may not already exist on common drive locations
    :postcondition: Return the first valid existing path, or fallback to a new directory in the current working
    directory
    :return: a string representing the absolute path to the root trading data directory
    """
    # Default path
    primary_path = "D:\\Trading_Data"

    # Backup paths
    candidate_paths = [
        primary_path,
        "E:\\Trading_Data",
        "C:\\Trading_Data",
        os.path.join(os.path.expanduser("~"), "Trading_Data")
    ]

    # Check if the path exists
    for path in candidate_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path

    # Final fallback
    fallback = os.path.join(os.getcwd(), "Trading_Data")
    warnings.warn(f"No valid Trading_Data folder found, fallback to current working directory: {fallback}")
    os.makedirs(fallback, exist_ok=True)
    return fallback
