"""
trading_config.py

Trading configuration settings for Bybit exchange.
Centralizes lot size, minimum order requirements, and other trading constraints.
"""

# Bybit Trading Constraints
BYBIT_MIN_LOT_SIZE = 0.001  # Minimum order size for BTCUSDT (BTC)
BYBIT_MIN_NOTIONAL = 100.0  # Minimum notional value in USDT

# Default lot sizes for different assets
DEFAULT_LOT_SIZES = {
    'BTC': 0.001,
    'ETH': 0.01,
    'SOL': 0.1,
    'TON': 1.0,
    'TRX': 10.0,
    'USDC': 10.0,
    'USDT': 10.0
}

# Position sizing defaults
DEFAULT_INITIAL_CAPITAL = 10000.0

def get_lot_size(asset: str) -> float:
    """
    Get the default lot size for a given asset.
    
    :param asset: Asset symbol (e.g., 'BTC', 'ETH')
    :return: Default lot size for the asset
    """
    return DEFAULT_LOT_SIZES.get(asset.upper(), BYBIT_MIN_LOT_SIZE)

def validate_order_quantity(quantity: float, asset: str = 'BTC') -> bool:
    """
    Validate if order quantity meets Bybit requirements.
    
    :param quantity: Order quantity
    :param asset: Asset symbol
    :return: True if valid, False otherwise
    """
    min_lot = get_lot_size(asset)
    return quantity >= min_lot

def get_min_notional(asset: str = 'BTC') -> float:
    """
    Get minimum notional value for an asset.
    
    :param asset: Asset symbol
    :return: Minimum notional value in USDT
    """
    return BYBIT_MIN_NOTIONAL
