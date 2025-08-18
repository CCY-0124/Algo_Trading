"""
Market Data handler for real-time data processing.
Basic implementation for live market data management.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class MarketDataPoint:
    """Market data point structure"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None

class MarketData:
    """
    Basic market data handler for real-time data processing.
    
    Features:
    - Real-time price updates
    - Data caching
    - Callback notifications
    - Basic data validation
    """
    
    def __init__(self):
        """Initialize market data handler"""
        self.price_cache: Dict[str, MarketDataPoint] = {}
        self.data_history: Dict[str, List[MarketDataPoint]] = {}
        self.callbacks: List[Callable] = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for market data"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def add_callback(self, callback: Callable):
        """
        Add callback function for price updates.
        
        :param callback: Function to call on price update
        """
        self.callbacks.append(callback)
        logging.info(f"Added market data callback: {callback.__name__}")
    
    def remove_callback(self, callback: Callable):
        """Remove callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logging.info(f"Removed market data callback: {callback.__name__}")
    
    def update_price(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        open_price: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        close: Optional[float] = None
    ):
        """
        Update price for symbol.
        
        :param symbol: Trading symbol
        :param price: Current price
        :param volume: Trading volume
        :param open_price: Open price
        :param high: High price
        :param low: Low price
        :param close: Close price
        """
        # Validate price
        if price <= 0:
            logging.warning(f"Invalid price for {symbol}: {price}")
            return
        
        # Create data point
        data_point = MarketDataPoint(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=datetime.now(),
            open=open_price,
            high=high,
            low=low,
            close=close
        )
        
        # Update cache
        self.price_cache[symbol] = data_point
        
        # Add to history
        if symbol not in self.data_history:
            self.data_history[symbol] = []
        
        self.data_history[symbol].append(data_point)
        
        # Keep only last 1000 data points per symbol
        if len(self.data_history[symbol]) > 1000:
            self.data_history[symbol] = self.data_history[symbol][-1000:]
        
        # Notify callbacks
        self._notify_callbacks(symbol, data_point)
        
        logging.debug(f"Updated price for {symbol}: {price}")
    
    def _notify_callbacks(self, symbol: str, data_point: MarketDataPoint):
        """Notify all callbacks of price update"""
        for callback in self.callbacks:
            try:
                callback(symbol, data_point)
            except Exception as e:
                logging.error(f"Callback error for {callback.__name__}: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        data_point = self.price_cache.get(symbol)
        return data_point.price if data_point else None
    
    def get_price_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get current price data for symbol"""
        return self.price_cache.get(symbol)
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[MarketDataPoint]:
        """Get price history for symbol"""
        history = self.data_history.get(symbol, [])
        return history[-limit:] if limit > 0 else history
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols with cached data"""
        return list(self.price_cache.keys())
    
    def get_price_change(self, symbol: str, period_minutes: int = 60) -> Optional[float]:
        """
        Calculate price change over period.
        
        :param symbol: Trading symbol
        :param period_minutes: Period in minutes
        :return: Price change percentage
        """
        history = self.data_history.get(symbol, [])
        if len(history) < 2:
            return None
        
        current_price = history[-1].price
        cutoff_time = datetime.now().timestamp() - (period_minutes * 60)
        
        # Find price at cutoff time
        start_price = None
        for data_point in reversed(history):
            if data_point.timestamp.timestamp() <= cutoff_time:
                start_price = data_point.price
                break
        
        if start_price is None:
            start_price = history[0].price
        
        if start_price == 0:
            return None
        
        return ((current_price - start_price) / start_price) * 100
    
    def get_volume_data(self, symbol: str, period_minutes: int = 60) -> float:
        """
        Calculate total volume over period.
        
        :param symbol: Trading symbol
        :param period_minutes: Period in minutes
        :return: Total volume
        """
        history = self.data_history.get(symbol, [])
        if not history:
            return 0.0
        
        cutoff_time = datetime.now().timestamp() - (period_minutes * 60)
        total_volume = 0.0
        
        for data_point in history:
            if data_point.timestamp.timestamp() >= cutoff_time:
                total_volume += data_point.volume
        
        return total_volume
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear price cache"""
        if symbol:
            if symbol in self.price_cache:
                del self.price_cache[symbol]
            if symbol in self.data_history:
                del self.data_history[symbol]
            logging.info(f"Cleared cache for {symbol}")
        else:
            self.price_cache.clear()
            self.data_history.clear()
            logging.info("Cleared all market data cache")
    
    def get_summary(self) -> Dict:
        """Get market data summary"""
        summary = {
            "cached_symbols": len(self.price_cache),
            "total_data_points": sum(len(history) for history in self.data_history.values()),
            "callbacks": len(self.callbacks),
            "last_update": datetime.now().isoformat()
        }
        
        # Add price summary for each symbol
        price_summary = {}
        for symbol, data_point in self.price_cache.items():
            price_summary[symbol] = {
                "price": data_point.price,
                "volume": data_point.volume,
                "timestamp": data_point.timestamp.isoformat()
            }
        
        summary["prices"] = price_summary
        return summary 