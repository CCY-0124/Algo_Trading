"""
Position Manager for tracking trading positions.
Basic implementation for position tracking and management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from .order_manager import OrderSide

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class PositionManager:
    """
    Basic position manager for tracking trading positions.
    
    Features:
    - Position tracking and updates
    - P&L calculation
    - Position history
    """
    
    def __init__(self):
        """Initialize position manager"""
        self.positions: Dict[str, Position] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for position manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def update_position(
        self,
        symbol: str,
        quantity_change: float,
        price: float,
        side: OrderSide
    ) -> Position:
        """
        Update position based on trade execution.
        
        :param symbol: Trading symbol
        :param quantity_change: Quantity change (positive for buy, negative for sell)
        :param price: Execution price
        :param side: Trade side
        :return: Updated position
        """
        # Adjust quantity based on side
        if side == OrderSide.SELL:
            quantity_change = -quantity_change
        
        current_position = self.positions.get(symbol)
        
        if current_position is None:
            # Create new position
            if quantity_change > 0:
                position = Position(
                    symbol=symbol,
                    quantity=quantity_change,
                    average_price=price
                )
                self.positions[symbol] = position
                logging.info(f"Created position: {symbol} - {quantity_change} @ {price}")
            else:
                logging.warning(f"Cannot create short position for {symbol}")
                return None
        else:
            # Update existing position
            old_quantity = current_position.quantity
            old_avg_price = current_position.average_price
            
            new_quantity = old_quantity + quantity_change
            
            if new_quantity == 0:
                # Position closed
                realized_pnl = (price - old_avg_price) * abs(quantity_change)
                current_position.realized_pnl += realized_pnl
                current_position.quantity = 0
                logging.info(f"Closed position: {symbol} - Realized P&L: {realized_pnl:.2f}")
            elif new_quantity > 0:
                # Long position
                if quantity_change > 0:
                    # Adding to long position
                    total_cost = (old_quantity * old_avg_price) + (quantity_change * price)
                    current_position.quantity = new_quantity
                    current_position.average_price = total_cost / new_quantity
                else:
                    # Reducing long position
                    realized_pnl = (price - old_avg_price) * abs(quantity_change)
                    current_position.realized_pnl += realized_pnl
                    current_position.quantity = new_quantity
                    logging.info(f"Reduced position: {symbol} - Realized P&L: {realized_pnl:.2f}")
            else:
                # Short position (not supported in basic implementation)
                logging.warning(f"Short positions not supported for {symbol}")
                return current_position
            
            current_position.last_updated = datetime.now()
        
        return self.positions.get(symbol)
    
    def update_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """
        Update unrealized P&L for a position.
        
        :param symbol: Trading symbol
        :param current_price: Current market price
        :return: Unrealized P&L
        """
        position = self.positions.get(symbol)
        if not position or position.quantity == 0:
            return 0.0
        
        unrealized_pnl = (current_price - position.average_price) * position.quantity
        position.unrealized_pnl = unrealized_pnl
        position.last_updated = datetime.now()
        
        return unrealized_pnl
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())
    
    def get_open_positions(self) -> List[Position]:
        """Get open positions (quantity != 0)"""
        return [pos for pos in self.positions.values() if pos.quantity != 0]
    
    def get_total_pnl(self) -> Dict[str, float]:
        """Get total P&L across all positions"""
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            "realized_pnl": total_realized,
            "unrealized_pnl": total_unrealized,
            "total_pnl": total_realized + total_unrealized
        }
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions"""
        summary = {}
        for symbol, position in self.positions.items():
            summary[symbol] = {
                "quantity": position.quantity,
                "average_price": position.average_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "last_updated": position.last_updated.isoformat()
            }
        return summary
    
    def close_position(self, symbol: str, price: float) -> float:
        """
        Close a position completely.
        
        :param symbol: Trading symbol
        :param price: Closing price
        :return: Realized P&L
        """
        position = self.positions.get(symbol)
        if not position or position.quantity == 0:
            return 0.0
        
        realized_pnl = (price - position.average_price) * position.quantity
        position.realized_pnl += realized_pnl
        position.quantity = 0
        position.unrealized_pnl = 0.0
        position.last_updated = datetime.now()
        
        logging.info(f"Closed position: {symbol} - Realized P&L: {realized_pnl:.2f}")
        return realized_pnl 