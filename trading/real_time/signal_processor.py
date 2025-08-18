"""
Signal Processor for handling trading signals.
Basic implementation for signal generation and processing.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    """Signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    strategy_name: str
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()

class SignalProcessor:
    """
    Basic signal processor for handling trading signals.
    
    Features:
    - Signal generation and validation
    - Signal filtering and aggregation
    - Callback notifications
    - Signal history tracking
    """
    
    def __init__(self):
        """Initialize signal processor"""
        self.signal_history: List[TradingSignal] = []
        self.callbacks: List[Callable] = []
        self.filters: List[Callable] = []
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for signal processor"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def add_callback(self, callback: Callable):
        """
        Add callback function for signal notifications.
        
        :param callback: Function to call on signal generation
        """
        self.callbacks.append(callback)
        logging.info(f"Added signal callback: {callback.__name__}")
    
    def remove_callback(self, callback: Callable):
        """Remove callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logging.info(f"Removed signal callback: {callback.__name__}")
    
    def add_filter(self, filter_func: Callable):
        """
        Add signal filter function.
        
        :param filter_func: Function to filter signals
        """
        self.filters.append(filter_func)
        logging.info(f"Added signal filter: {filter_func.__name__}")
    
    def remove_filter(self, filter_func: Callable):
        """Remove signal filter"""
        if filter_func in self.filters:
            self.filters.remove(filter_func)
            logging.info(f"Removed signal filter: {filter_func.__name__}")
    
    def generate_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: float,
        price: float,
        strategy_name: str,
        parameters: Dict = None
    ) -> TradingSignal:
        """
        Generate a new trading signal.
        
        :param symbol: Trading symbol
        :param signal_type: Signal type (buy/sell/hold)
        :param strength: Signal strength (0.0 to 1.0)
        :param price: Current price
        :param strategy_name: Strategy name
        :param parameters: Strategy parameters
        :return: Generated signal
        """
        # Validate signal parameters
        if not (0.0 <= strength <= 1.0):
            logging.warning(f"Invalid signal strength: {strength}, clamping to valid range")
            strength = max(0.0, min(1.0, strength))
        
        if price <= 0:
            logging.error(f"Invalid price for signal: {price}")
            return None
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            strategy_name=strategy_name,
            parameters=parameters or {}
        )
        
        # Apply filters
        if self._apply_filters(signal):
            # Add to history
            self.signal_history.append(signal)
            
            # Keep only last 1000 signals
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            # Notify callbacks
            self._notify_callbacks(signal)
            
            logging.info(f"Generated signal: {symbol} {signal_type.value} (strength: {strength:.2f})")
            return signal
        else:
            logging.info(f"Signal filtered out: {symbol} {signal_type.value}")
            return None
    
    def _apply_filters(self, signal: TradingSignal) -> bool:
        """Apply all filters to signal"""
        for filter_func in self.filters:
            try:
                if not filter_func(signal):
                    return False
            except Exception as e:
                logging.error(f"Filter error for {filter_func.__name__}: {e}")
                return False
        return True
    
    def _notify_callbacks(self, signal: TradingSignal):
        """Notify all callbacks of signal"""
        for callback in self.callbacks:
            try:
                callback(signal)
            except Exception as e:
                logging.error(f"Callback error for {callback.__name__}: {e}")
    
    def get_recent_signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[TradingSignal]:
        """
        Get recent signals.
        
        :param symbol: Filter by symbol (optional)
        :param limit: Number of signals to retrieve
        :return: List of recent signals
        """
        signals = self.signal_history
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        return signals[-limit:] if limit > 0 else signals
    
    def get_signals_by_type(self, signal_type: SignalType, symbol: Optional[str] = None) -> List[TradingSignal]:
        """
        Get signals by type.
        
        :param signal_type: Signal type to filter
        :param symbol: Filter by symbol (optional)
        :return: List of signals
        """
        signals = [s for s in self.signal_history if s.signal_type == signal_type]
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        return signals
    
    def get_signal_summary(self, symbol: Optional[str] = None) -> Dict:
        """
        Get signal summary.
        
        :param symbol: Filter by symbol (optional)
        :return: Signal summary
        """
        signals = self.signal_history
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        if not signals:
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
                "average_strength": 0.0,
                "last_signal": None
            }
        
        buy_count = len([s for s in signals if s.signal_type == SignalType.BUY])
        sell_count = len([s for s in signals if s.signal_type == SignalType.SELL])
        hold_count = len([s for s in signals if s.signal_type == SignalType.HOLD])
        
        total_strength = sum(s.strength for s in signals)
        average_strength = total_strength / len(signals)
        
        return {
            "total_signals": len(signals),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "hold_signals": hold_count,
            "average_strength": average_strength,
            "last_signal": signals[-1].timestamp.isoformat() if signals else None
        }
    
    def get_signals_by_strategy(self, strategy_name: str) -> List[TradingSignal]:
        """
        Get signals by strategy.
        
        :param strategy_name: Strategy name
        :return: List of signals
        """
        return [s for s in self.signal_history if s.strategy_name == strategy_name]
    
    def clear_history(self, symbol: Optional[str] = None):
        """Clear signal history"""
        if symbol:
            self.signal_history = [s for s in self.signal_history if s.symbol != symbol]
            logging.info(f"Cleared signal history for {symbol}")
        else:
            self.signal_history.clear()
            logging.info("Cleared all signal history")
    
    def get_strongest_signals(self, limit: int = 10) -> List[TradingSignal]:
        """
        Get strongest signals.
        
        :param limit: Number of signals to retrieve
        :return: List of strongest signals
        """
        sorted_signals = sorted(self.signal_history, key=lambda x: x.strength, reverse=True)
        return sorted_signals[:limit]
    
    def get_signals_in_timeframe(self, start_time: datetime, end_time: datetime) -> List[TradingSignal]:
        """
        Get signals within timeframe.
        
        :param start_time: Start time
        :param end_time: End time
        :return: List of signals in timeframe
        """
        return [
            s for s in self.signal_history
            if start_time <= s.timestamp <= end_time
        ] 