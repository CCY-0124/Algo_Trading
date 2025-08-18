"""
Trading Engine for coordinating real-time trading operations.
Updated for Bybit with real market data.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from threading import Thread, Event

from ..execution import OrderManager, PositionManager, RiskManager
from ..brokers import ExchangeFactory
from .market_data import MarketData
from .signal_processor import SignalProcessor, SignalType, TradingSignal
from config.trading_config import get_lot_size, validate_order_quantity, get_min_notional, DEFAULT_INITIAL_CAPITAL

class TradingEngine:
    """
    Main trading engine for coordinating real-time trading operations.
    
    Features:
    - Real-time market data processing
    - Signal generation and execution
    - Risk management integration
    - Order and position management
    - Automated trading coordination
    """
    
    def __init__(self, initial_capital: float = DEFAULT_INITIAL_CAPITAL, api_key: str = "", api_secret: str = "", testnet: bool = True):
        """Initialize trading engine"""
        self.initial_capital = initial_capital
        self.is_running = False
        self.stop_event = Event()
        
        # Bybit trading constraints
        self.min_lot_size = get_lot_size('BTC')  # Minimum order size for BTCUSDT
        self.min_notional = get_min_notional()  # Minimum notional value in USDT
        
        # Initialize components
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(initial_capital)
        self.exchange_factory = ExchangeFactory()
        
        # Initialize Bybit client
        self.bybit_client = self.exchange_factory.create_client(
            exchange="bybit",
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        self.market_data = MarketData()
        self.signal_processor = SignalProcessor()
        
        # Trading state
        self.active_strategies: Dict[str, Callable] = {}
        self.trading_pairs: List[str] = []
        
        # Callbacks
        self.execution_callbacks: List[Callable] = []
        
        self._setup_logging()
        self._setup_callbacks()
    
    def _setup_logging(self):
        """Setup logging for trading engine"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _setup_callbacks(self):
        """Setup internal callbacks"""
        # Market data callback
        self.market_data.add_callback(self._on_price_update)
        
        # Signal processor callback
        self.signal_processor.add_callback(self._on_signal_generated)
    
    def _on_price_update(self, symbol: str, data_point):
        """Handle price updates"""
        # Update unrealized P&L for positions
        position = self.position_manager.get_position(symbol)
        if position and position.quantity != 0:
            unrealized_pnl = self.position_manager.update_unrealized_pnl(symbol, data_point.price)
            
            # Update risk manager
            self.risk_manager.update_capital(unrealized_pnl - position.unrealized_pnl)
        
        # Run active strategies
        self._run_strategies(symbol, data_point.price)
    
    def _on_signal_generated(self, signal: TradingSignal):
        """Handle signal generation"""
        # Check risk limits before execution
        if self.risk_manager.should_stop_trading():
            logging.warning("Trading stopped due to risk limits")
            return
        
        # Execute signal
        self._execute_signal(signal)
        
        # Notify callbacks
        for callback in self.execution_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logging.error(f"Execution callback error: {e}")
    
    def _run_strategies(self, symbol: str, price: float):
        """Run active strategies for symbol"""
        for strategy_name, strategy_func in self.active_strategies.items():
            try:
                # Run strategy
                signal_type, strength = strategy_func(symbol, price)
                
                if signal_type != SignalType.HOLD:
                    # Generate signal
                    self.signal_processor.generate_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        price=price,
                        strategy_name=strategy_name
                    )
                    
            except Exception as e:
                logging.error(f"Strategy error for {strategy_name}: {e}")
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute trading signal"""
        try:
            # Get current position
            position = self.position_manager.get_position(signal.symbol)
            current_quantity = position.quantity if position else 0
            
            # Calculate order quantity (basic implementation)
            if signal.signal_type == SignalType.BUY and current_quantity == 0:
                # Buy signal - calculate position size
                available_capital = self.risk_manager.current_capital * 0.1  # 10% of capital
                order_quantity = available_capital / signal.price
                
                # Apply Bybit minimum constraints
                order_quantity = max(order_quantity, self.min_lot_size)  # Minimum lot size
                notional_value = order_quantity * signal.price
                if notional_value < self.min_notional:
                    # Adjust quantity to meet minimum notional requirement
                    order_quantity = self.min_notional / signal.price
                
                # Check risk limits
                risk_check = self.risk_manager.check_position_size(
                    signal.symbol, 
                    order_quantity * signal.price
                )
                
                if risk_check["allowed"]:
                    # Place buy order using Bybit
                    order_response = self.bybit_client.place_order(
                        symbol=signal.symbol,
                        side="BUY",
                        order_type="MARKET",
                        quantity=order_quantity
                    )
                    
                    if "error" not in order_response:
                        # Update order manager
                        order = self.order_manager.create_order(
                            symbol=signal.symbol,
                            side=self.order_manager.OrderSide.BUY,
                            order_type=self.order_manager.OrderType.MARKET,
                            quantity=order_quantity
                        )
                        
                        # Update order status
                        self.order_manager.update_order_status(
                            order_id=order.id,
                            status=self.order_manager.OrderStatus.FILLED,
                            filled_quantity=order_quantity,
                            average_price=signal.price
                        )
                        
                        # Update position
                        self.position_manager.update_position(
                            symbol=signal.symbol,
                            quantity_change=order_quantity,
                            price=signal.price,
                            side=self.order_manager.OrderSide.BUY
                        )
                        
                        # Update risk manager
                        self.risk_manager.update_capital(-order_quantity * signal.price)
                        
                        logging.info(f"Executed BUY signal for {signal.symbol}: {order_quantity}")
                    else:
                        logging.error(f"Failed to place order: {order_response['error']}")
                else:
                    logging.warning(f"Risk limit exceeded for {signal.symbol} BUY signal")
            
            elif signal.signal_type == SignalType.SELL and current_quantity > 0:
                # Sell signal - close position
                order_response = self.bybit_client.place_order(
                    symbol=signal.symbol,
                    side="SELL",
                    order_type="MARKET",
                    quantity=current_quantity
                )
                
                if "error" not in order_response:
                    # Update order manager
                    order = self.order_manager.create_order(
                        symbol=signal.symbol,
                        side=self.order_manager.OrderSide.SELL,
                        order_type=self.order_manager.OrderType.MARKET,
                        quantity=current_quantity
                    )
                    
                    # Update order status
                    self.order_manager.update_order_status(
                        order_id=order.id,
                        status=self.order_manager.OrderStatus.FILLED,
                        filled_quantity=current_quantity,
                        average_price=signal.price
                    )
                    
                    # Update position
                    self.position_manager.update_position(
                        symbol=signal.symbol,
                        quantity_change=current_quantity,
                        price=signal.price,
                        side=self.order_manager.OrderSide.SELL
                    )
                    
                    # Update risk manager
                    self.risk_manager.update_capital(current_quantity * signal.price)
                    
                    logging.info(f"Executed SELL signal for {signal.symbol}: {current_quantity}")
                else:
                    logging.error(f"Failed to place order: {order_response['error']}")
            
        except Exception as e:
            logging.error(f"Signal execution error: {e}")
    
    def add_strategy(self, strategy_name: str, strategy_func: Callable):
        """
        Add trading strategy.
        
        :param strategy_name: Strategy name
        :param strategy_func: Strategy function (symbol, price) -> (signal_type, strength)
        """
        self.active_strategies[strategy_name] = strategy_func
        logging.info(f"Added strategy: {strategy_name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove trading strategy"""
        if strategy_name in self.active_strategies:
            del self.active_strategies[strategy_name]
            logging.info(f"Removed strategy: {strategy_name}")
    
    def add_trading_pair(self, symbol: str):
        """Add trading pair"""
        if symbol not in self.trading_pairs:
            self.trading_pairs.append(symbol)
            logging.info(f"Added trading pair: {symbol}")
    
    def remove_trading_pair(self, symbol: str):
        """Remove trading pair"""
        if symbol in self.trading_pairs:
            self.trading_pairs.remove(symbol)
            logging.info(f"Removed trading pair: {symbol}")
    
    def add_execution_callback(self, callback: Callable):
        """Add execution callback"""
        self.execution_callbacks.append(callback)
        logging.info(f"Added execution callback: {callback.__name__}")
    
    def remove_execution_callback(self, callback: Callable):
        """Remove execution callback"""
        if callback in self.execution_callbacks:
            self.execution_callbacks.remove(callback)
            logging.info(f"Removed execution callback: {callback.__name__}")
    
    def start(self):
        """Start trading engine"""
        if self.is_running:
            logging.warning("Trading engine already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start market data thread
        self.market_data_thread = Thread(target=self._market_data_loop)
        self.market_data_thread.daemon = True
        self.market_data_thread.start()
        
        logging.info("Trading engine started")
    
    def stop(self):
        """Stop trading engine"""
        if not self.is_running:
            logging.warning("Trading engine not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        logging.info("Trading engine stopped")
    
    def _market_data_loop(self):
        """Market data processing loop with real Bybit data"""
        while not self.stop_event.is_set():
            try:
                # Get real market data from Bybit
                for symbol in self.trading_pairs:
                    # Get current price from Bybit
                    price_data = self.bybit_client.get_symbol_price(symbol)
                    
                    if "error" not in price_data:
                        # Update market data with real price
                        self.market_data.update_price(
                            symbol=symbol,
                            price=price_data["price"],
                            volume=price_data.get("volume", 0.0),
                            open_price=price_data.get("open", None),
                            high=price_data.get("high", None),
                            low=price_data.get("low", None),
                            close=price_data.get("close", None)
                        )
                    else:
                        logging.warning(f"Failed to get price for {symbol}: {price_data['error']}")
                
                time.sleep(5)  # Update every 5 seconds to respect rate limits
                
            except Exception as e:
                logging.error(f"Market data loop error: {e}")
                time.sleep(10)  # Wait before retry
    
    def get_status(self) -> Dict:
        """Get trading engine status"""
        return {
            "is_running": self.is_running,
            "initial_capital": self.initial_capital,
            "active_strategies": list(self.active_strategies.keys()),
            "trading_pairs": self.trading_pairs.copy(),
            "risk_summary": self.risk_manager.get_risk_summary(),
            "position_summary": self.position_manager.get_position_summary(),
            "signal_summary": self.signal_processor.get_signal_summary(),
            "market_data_summary": self.market_data.get_summary(),
            "last_update": datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        risk_summary = self.risk_manager.get_risk_summary()
        position_summary = self.position_manager.get_position_summary()
        
        total_pnl = self.position_manager.get_total_pnl()
        
        return {
            "capital": {
                "initial": self.initial_capital,
                "current": risk_summary["capital"]["current"],
                "change": risk_summary["capital"]["current"] - self.initial_capital,
                "change_percentage": ((risk_summary["capital"]["current"] - self.initial_capital) / self.initial_capital) * 100
            },
            "pnl": total_pnl,
            "risk_metrics": risk_summary["risk_metrics"],
            "positions": len(position_summary),
            "signals": self.signal_processor.get_signal_summary(),
            "last_update": datetime.now().isoformat()
        } 