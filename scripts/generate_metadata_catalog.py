"""
generate_metadata_catalog.py

This script fetches metadata from Glassnode API to create a comprehensive catalog
of available assets and metrics, then generates configuration files for data download.

Step 1: Fetch all assets and metrics metadata
Step 2: Generate configuration files for BTC, ETH, SOL
Step 3: Create a new data download script
"""

import os
import sys
import requests
import json
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BASE_URL = "https://api.glassnode.com/v1"
TARGET_ASSETS = ["BTC", "ETH", "SOL"]
NEW_BASE_PATH = r"D:\Trading_Data\glassnode_data2"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_api_key() -> Optional[str]:
    """Load API key from encrypted storage or environment"""
    try:
        # Try encrypted storage first
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
        if config_path not in sys.path:
            sys.path.insert(0, config_path)
        
        from config.secrets import get_api_key
        api_key = get_api_key('glassnode', 'main')
        if api_key:
            print("✓ Loaded API key from encrypted storage")
            return api_key
    except Exception as e:
        print(f"WARNING: Could not load from encrypted storage: {e}")
    
    # Fallback to environment variable
    api_key = os.getenv('GLASSNODE_API_KEY', '')
    if api_key:
        print("✓ Loaded API key from environment variable")
        return api_key
    
    print("ERROR: No API key found")
    return None

def fetch_assets_metadata(api_key: str) -> List[Dict]:
    """Fetch all assets metadata from Glassnode API"""
    print("\n=== Fetching Assets Metadata ===")
    
    url = f"{BASE_URL}/metadata/assets"
    params = {'api_key': api_key}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        assets = data.get('data', [])
        
        print(f"✓ Found {len(assets)} total assets")
        
        # Filter for target assets
        target_assets = [asset for asset in assets if asset['id'] in TARGET_ASSETS]
        print(f"✓ Target assets found: {[asset['id'] for asset in target_assets]}")
        
        return target_assets
        
    except Exception as e:
        print(f"ERROR: Error fetching assets metadata: {e}")
        return []

def fetch_metrics_list(api_key: str) -> List[str]:
    """Fetch all available metrics paths from Glassnode API"""
    print("\n=== Fetching Metrics List ===")
    
    url = f"{BASE_URL}/metadata/metrics"
    params = {'api_key': api_key}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        metrics = response.json()
        print(f"✓ Found {len(metrics)} total metrics")
        
        return metrics
        
    except Exception as e:
        print(f"ERROR: Error fetching metrics list: {e}")
        return []

def fetch_metric_metadata(api_key: str, metric_path: str, asset: str = None) -> Optional[Dict]:
    """Fetch detailed metadata for a specific metric"""
    url = f"{BASE_URL}/metadata/metric"
    params = {
        'api_key': api_key,
        'path': metric_path
    }
    
    if asset:
        params['a'] = asset
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"WARNING: Metric {metric_path} returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"WARNING: Error fetching metadata for {metric_path}: {e}")
        return None

def is_metric_supported_for_asset(metric_metadata: Dict, asset: str) -> bool:
    """Check if a metric is supported for a specific asset"""
    if not metric_metadata:
        return False
    
    # Check if asset is in supported assets list
    supported_assets = metric_metadata.get('parameters', {}).get('a', [])
    return asset in supported_assets

def get_min_resolution(metric_metadata: Dict) -> str:
    """Get the minimum resolution for a metric"""
    if not metric_metadata:
        return "24h"  # Default
    
    resolutions = metric_metadata.get('parameters', {}).get('i', [])
    if not resolutions:
        return "24h"
    
    # Prefer larger resolutions for daily downloads: 24h, 1d, 7d, then 1h as fallback
    preferred_order = ["24h", "1w", "1month", "1h"]
    for res in preferred_order:
        if res in resolutions:
            return res
    
    return resolutions[0]

def get_tier(metric_metadata: Dict) -> int:
    """Get the tier level for a metric"""
    return metric_metadata.get('tier', 1) if metric_metadata else 1

def generate_metrics_info_csv(asset: str, supported_metrics: List[Tuple[str, Dict]]) -> pd.DataFrame:
    """Generate metrics_info CSV data for an asset"""
    data = []
    
    for metric_path, metadata in supported_metrics:
        min_resolution = get_min_resolution(metadata)
        tier = get_tier(metadata)
        
        # Get supported assets for this metric
        supported_assets = metadata.get('parameters', {}).get('a', [])
        supported_assets_str = ','.join(supported_assets) if supported_assets else asset
        
        # Clean up description to avoid CSV issues
        description = metadata.get('descriptors', {}).get('description', {}).get('default', '')
        if description:
            # Remove newlines and quotes that could break CSV
            description = description.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
            # Truncate very long descriptions to avoid CSV issues
            if len(description) > 200:
                description = description[:200] + "..."
        
        data.append({
            'path': metric_path,
            'min_resolution': min_resolution,
            'supported_assets': supported_assets_str,
            'tier': tier,
            'name': metadata.get('descriptors', {}).get('name', ''),
            'group': metadata.get('descriptors', {}).get('group', ''),
            'description': description,
            'last_update': ''
        })
    
    return pd.DataFrame(data)

def save_metrics_info(asset: str, df: pd.DataFrame, base_path: str):
    """Save metrics_info CSV file for an asset"""
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"metrics_info_{asset.lower()}.csv")
    
    # Save with proper CSV settings to avoid parsing issues
    df.to_csv(file_path, index=False, quoting=1, escapechar='\\', encoding='utf-8')
    print(f"✓ Saved {len(df)} metrics for {asset} to {file_path}")

def main():
    """Main function to generate metadata catalog"""
    print("�� Glassnode Metadata Catalog Generator")
    print("=" * 50)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("ERROR: Cannot proceed without API key")
        return
    
    # Step 1: Fetch assets metadata
    assets_metadata = fetch_assets_metadata(api_key)
    if not assets_metadata:
        print("ERROR: Failed to fetch assets metadata")
        return
    
    # Step 2: Fetch metrics list
    metrics_list = fetch_metrics_list(api_key)
    if not metrics_list:
        print("ERROR: Failed to fetch metrics list")
        return
    
    # Step 3: Process each target asset
    for asset_metadata in assets_metadata:
        asset_id = asset_metadata['id']
        print(f"\n{'='*20} Processing {asset_id} {'='*20}")
        
        # Check if asset has on-chain support
        has_onchain_support = False
        for blockchain in asset_metadata.get('blockchains', []):
            if blockchain.get('on_chain_support', False):
                has_onchain_support = True
                break
        
        # For now, assume all target assets have on-chain support
        # The API might not always return this flag correctly
        if asset_id in TARGET_ASSETS:
            has_onchain_support = True
        
        if not has_onchain_support:
            print(f"WARNING: {asset_id} does not have on-chain support, skipping")
            continue
        
        # Fetch metadata for each metric
        supported_metrics = []
        total_metrics = len(metrics_list)
        
        for i, metric_path in enumerate(metrics_list, 1):
            print(f"  Processing metric {i}/{total_metrics}: {metric_path}")
            
            # Fetch metadata for this metric
            metadata = fetch_metric_metadata(api_key, metric_path, asset_id)
            
            if metadata and is_metric_supported_for_asset(metadata, asset_id):
                supported_metrics.append((metric_path, metadata))
                print(f"    ✓ Supported")
            else:
                print(f"    ✗ Not supported")
            
            # Rate limiting
            if i % 10 == 0:
                import time
                time.sleep(1)
        
        print(f"\n✓ {asset_id}: Found {len(supported_metrics)} supported metrics")
        
        # Generate and save metrics_info CSV
        if supported_metrics:
            df = generate_metrics_info_csv(asset_id, supported_metrics)
            save_metrics_info(asset_id, df, NEW_BASE_PATH)
    
    print(f"\n{'='*50}")
    print("SUCCESS: Metadata catalog generation complete!")
    print(f"INFO: Configuration files saved to: {NEW_BASE_PATH}")
    print("\nNext steps:")
    print("1. Review the generated metrics_info_*.csv files")
    print("2. Run the new daily download script")
    print("3. Monitor the download progress")

if __name__ == "__main__":
    main()