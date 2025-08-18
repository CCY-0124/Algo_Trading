"""
Order Manager for handling trading orders.
Basic implementation for order creation, tracking, and management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from config.trading_config import validate_order_quantity

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class OrderManager:
    """
    Basic order manager for handling trading orders.
    
    Features:
    - Order creation and tracking
    - Order status updates
    - Order history management
    """
    
    def __init__(self):
        """Initialize order manager"""
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for order manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"order_{self.order_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None
    ) -> Order:
        """
        Create a new order.
        
        :param symbol: Trading symbol (e.g., 'BTCUSDT')
        :param side: Buy or sell
        :param order_type: Market, limit, etc.
        :param quantity: Order quantity
        :param price: Order price (required for limit orders)
        :return: Created order
        """
        # Validate order parameters
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError(f"Price is required for {order_type.value} orders")
        
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Validate minimum lot size for crypto derivatives
        if not validate_order_quantity(quantity, symbol.replace('USDT', '')):
            raise ValueError(f"Quantity {quantity} is below minimum lot size for {symbol}")
        
        # Create order
        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        
        # Store order
        self.orders[order_id] = order
        
        logging.info(f"Created order: {order_id} - {side.value} {quantity} {symbol}")
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: float = 0.0,
        average_price: Optional[float] = None
    ) -> bool:
        """
        Update order status.
        
        :param order_id: Order ID
        :param status: New status
        :param filled_quantity: Filled quantity
        :param average_price: Average fill price
        :return: True if updated successfully
        """
        order = self.orders.get(order_id)
        if not order:
            logging.error(f"Order {order_id} not found")
            return False
        
        # Update order
        order.status = status
        order.filled_quantity = filled_quantity
        
        if average_price is not None:
            order.average_price = average_price
        
        if status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            order.filled_at = datetime.now()
        
        logging.info(f"Updated order {order_id}: {status.value}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        order = self.orders.get(order_id)
        if not order:
            logging.error(f"Order {order_id} not found")
            return False
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logging.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logging.info(f"Cancelled order: {order_id}")
        return True
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders (pending or partially filled)"""
        active_orders = []
        for order in self.orders.values():
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                if symbol is None or order.symbol == symbol:
                    active_orders.append(order)
        return active_orders
    
    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history"""
        history = []
        for order in self.orders.values():
            if symbol is None or order.symbol == symbol:
                history.append(order)
        return sorted(history, key=lambda x: x.created_at, reverse=True)
    
    def get_total_volume(self, symbol: str, side: OrderSide) -> float:
        """Get total volume for symbol and side"""
        total_volume = 0.0
        for order in self.orders.values():
            if order.symbol == symbol and order.side == side:
                if order.status == OrderStatus.FILLED:
                    total_volume += order.filled_quantity
                elif order.status == OrderStatus.PARTIALLY_FILLED:
                    total_volume += order.filled_quantity
        return total_volume 