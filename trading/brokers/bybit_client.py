"""
Bybit Client using CCXT library.
Real exchange connectivity for Bybit trading.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import ccxt

class BybitClient:
    """
    Bybit client using CCXT library for exchange operations.
    
    Features:
    - Real market data retrieval
    - Order placement and management
    - Account information
    - Position management
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        """
        Initialize Bybit client using CCXT.
        
        :param api_key: API key
        :param api_secret: API secret
        :param testnet: Use testnet (default True for safety)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._setup_logging()
        self._initialize_exchange()
    
    def _setup_logging(self):
        """Setup logging for Bybit client"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange instance"""
        try:
            # Initialize Bybit exchange
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.testnet,  # Use testnet if specified
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Use spot trading
                }
            })
            
            logging.info(f"Initialized Bybit client (testnet: {self.testnet})")
            
        except Exception as e:
            logging.error(f"Failed to initialize Bybit exchange: {e}")
            raise
    
    def get_server_time(self) -> Dict:
        """Get server time"""
        try:
            timestamp = self.exchange.fetch_time()
            return {
                "serverTime": timestamp,
                "localTime": int(time.time() * 1000),
                "timeDiff": timestamp - int(time.time() * 1000)
            }
        except Exception as e:
            logging.error(f"Failed to get server time: {e}")
            return {"error": str(e)}
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        try:
            markets = self.exchange.load_markets()
            return {
                "markets": list(markets.keys()),
                "total_markets": len(markets),
                "exchange": self.exchange.id
            }
        except Exception as e:
            logging.error(f"Failed to get exchange info: {e}")
            return {"error": str(e)}
    
    def get_symbol_price(self, symbol: str) -> Dict:
        """
        Get current price for symbol.
        
        :param symbol: Trading symbol (e.g., 'BTC/USDT')
        :return: Price information
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "price": ticker['last'],
                "bid": ticker['bid'],
                "ask": ticker['ask'],
                "volume": ticker['baseVolume'],
                "timestamp": ticker['timestamp']
            }
        except Exception as e:
            logging.error(f"Failed to get price for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_klines(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List:
        """
        Get kline/candlestick data.
        
        :param symbol: Trading symbol
        :param timeframe: Time interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        :param limit: Number of klines to retrieve
        :return: Kline data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logging.error(f"Failed to get klines for {symbol}: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if not self.api_key:
                logging.warning("API key not provided, returning simulated account info")
                return {
                    "makerCommission": 0.1,
                    "takerCommission": 0.1,
                    "buyerCommission": 0.1,
                    "sellerCommission": 0.1,
                    "canTrade": True,
                    "canWithdraw": False,
                    "canDeposit": False,
                    "updateTime": int(time.time() * 1000),
                    "accountType": "SPOT",
                    "balances": [
                        {
                            "asset": "USDT",
                            "free": "10000.00000000",
                            "locked": "0.00000000"
                        },
                        {
                            "asset": "BTC",
                            "free": "0.00000000",
                            "locked": "0.00000000"
                        }
                    ]
                }
            
            balance = self.exchange.fetch_balance()
            
            # Format balance data
            balances = []
            for asset, info in balance['total'].items():
                if info > 0:  # Only show assets with balance
                    balances.append({
                        "asset": asset,
                        "free": str(balance['free'].get(asset, 0)),
                        "locked": str(balance['used'].get(asset, 0)),
                        "total": str(info)
                    })
            
            return {
                "makerCommission": 0.1,  # Bybit default
                "takerCommission": 0.1,  # Bybit default
                "buyerCommission": 0.1,
                "sellerCommission": 0.1,
                "canTrade": True,
                "canWithdraw": False,
                "canDeposit": False,
                "updateTime": int(time.time() * 1000),
                "accountType": "SPOT",
                "balances": balances
            }
            
        except Exception as e:
            logging.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict:
        """
        Place an order.
        
        :param symbol: Trading symbol
        :param side: BUY or SELL
        :param order_type: MARKET or LIMIT
        :param quantity: Order quantity
        :param price: Order price (required for LIMIT orders)
        :return: Order response
        """
        try:
            if not self.api_key:
                logging.warning("API key not provided, returning simulated order")
                
                # Simulate order placement
                order_id = f"sim_{int(time.time() * 1000)}"
                
                return {
                    "symbol": symbol,
                    "orderId": order_id,
                    "orderListId": -1,
                    "clientOrderId": order_id,
                    "transactTime": int(time.time() * 1000),
                    "price": str(price) if price else "0.00000000",
                    "origQty": str(quantity),
                    "executedQty": str(quantity) if order_type == "MARKET" else "0.00000000",
                    "cummulativeQuoteQty": str(quantity * (price or 50000)),
                    "status": "FILLED" if order_type == "MARKET" else "NEW",
                    "timeInForce": "GTC",
                    "type": order_type,
                    "side": side
                }
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'type': order_type.lower(),
                'side': side.lower(),
                'amount': quantity
            }
            
            if price and order_type.upper() == "LIMIT":
                order_params['price'] = price
            
            # Place order
            order = self.exchange.create_order(**order_params)
            
            # Format response
            return {
                "symbol": order['symbol'],
                "orderId": order['id'],
                "orderListId": -1,
                "clientOrderId": order.get('clientOrderId', ''),
                "transactTime": order['timestamp'],
                "price": str(order.get('price', 0)),
                "origQty": str(order['amount']),
                "executedQty": str(order['filled']),
                "cummulativeQuoteQty": str(order.get('cost', 0)),
                "status": order['status'].upper(),
                "timeInForce": "GTC",
                "type": order['type'].upper(),
                "side": order['side'].upper()
            }
            
        except Exception as e:
            logging.error(f"Failed to place order: {e}")
            return {"error": str(e)}
    
    def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """
        Get order status.
        
        :param symbol: Trading symbol
        :param order_id: Order ID
        :return: Order status
        """
        try:
            if not self.api_key:
                logging.warning("API key not provided, returning simulated order status")
                return {
                    "symbol": symbol,
                    "orderId": order_id,
                    "orderListId": -1,
                    "clientOrderId": order_id,
                    "price": "50000.00000000",
                    "origQty": "0.00100000",
                    "executedQty": "0.00100000",
                    "cummulativeQuoteQty": "50.00000000",
                    "status": "FILLED",
                    "timeInForce": "GTC",
                    "type": "MARKET",
                    "side": "BUY",
                    "stopPrice": "0.00000000",
                    "icebergQty": "0.00000000",
                    "time": int(time.time() * 1000),
                    "updateTime": int(time.time() * 1000),
                    "isWorking": False,
                    "origQuoteOrderQty": "0.00000000"
                }
            
            order = self.exchange.fetch_order(order_id, symbol)
            
            return {
                "symbol": order['symbol'],
                "orderId": order['id'],
                "orderListId": -1,
                "clientOrderId": order.get('clientOrderId', ''),
                "price": str(order.get('price', 0)),
                "origQty": str(order['amount']),
                "executedQty": str(order['filled']),
                "cummulativeQuoteQty": str(order.get('cost', 0)),
                "status": order['status'].upper(),
                "timeInForce": "GTC",
                "type": order['type'].upper(),
                "side": order['side'].upper(),
                "stopPrice": "0.00000000",
                "icebergQty": "0.00000000",
                "time": order['timestamp'],
                "updateTime": order['timestamp'],
                "isWorking": order['status'] in ['open', 'pending'],
                "origQuoteOrderQty": "0.00000000"
            }
            
        except Exception as e:
            logging.error(f"Failed to get order status: {e}")
            return {"error": str(e)}
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Cancel an order.
        
        :param symbol: Trading symbol
        :param order_id: Order ID
        :return: Cancellation response
        """
        try:
            if not self.api_key:
                logging.warning("API key not provided, returning simulated cancellation")
                return {
                    "symbol": symbol,
                    "orderId": order_id,
                    "orderListId": -1,
                    "clientOrderId": order_id,
                    "price": "50000.00000000",
                    "origQty": "0.00100000",
                    "executedQty": "0.00000000",
                    "cummulativeQuoteQty": "0.00000000",
                    "status": "CANCELED",
                    "timeInForce": "GTC",
                    "type": "LIMIT",
                    "side": "BUY"
                }
            
            result = self.exchange.cancel_order(order_id, symbol)
            
            return {
                "symbol": result['symbol'],
                "orderId": result['id'],
                "orderListId": -1,
                "clientOrderId": result.get('clientOrderId', ''),
                "price": str(result.get('price', 0)),
                "origQty": str(result['amount']),
                "executedQty": str(result['filled']),
                "cummulativeQuoteQty": str(result.get('cost', 0)),
                "status": result['status'].upper(),
                "timeInForce": "GTC",
                "type": result['type'].upper(),
                "side": result['side'].upper()
            }
            
        except Exception as e:
            logging.error(f"Failed to cancel order: {e}")
            return {"error": str(e)}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List:
        """
        Get open orders.
        
        :param symbol: Trading symbol (optional)
        :return: List of open orders
        """
        try:
            if not self.api_key:
                logging.warning("API key not provided, returning empty open orders")
                return []
            
            if symbol:
                orders = self.exchange.fetch_open_orders(symbol)
            else:
                orders = self.exchange.fetch_open_orders()
            
            # Format orders
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    "symbol": order['symbol'],
                    "orderId": order['id'],
                    "orderListId": -1,
                    "clientOrderId": order.get('clientOrderId', ''),
                    "price": str(order.get('price', 0)),
                    "origQty": str(order['amount']),
                    "executedQty": str(order['filled']),
                    "cummulativeQuoteQty": str(order.get('cost', 0)),
                    "status": order['status'].upper(),
                    "timeInForce": "GTC",
                    "type": order['type'].upper(),
                    "side": order['side'].upper(),
                    "stopPrice": "0.00000000",
                    "icebergQty": "0.00000000",
                    "time": order['timestamp'],
                    "updateTime": order['timestamp'],
                    "isWorking": order['status'] in ['open', 'pending'],
                    "origQuoteOrderQty": "0.00000000"
                })
            
            return formatted_orders
            
        except Exception as e:
            logging.error(f"Failed to get open orders: {e}")
            return []
    
    def get_trade_history(self, symbol: str, limit: int = 100) -> List:
        """
        Get trade history.
        
        :param symbol: Trading symbol
        :param limit: Number of trades to retrieve
        :return: Trade history
        """
        try:
            if not self.api_key:
                logging.warning("API key not provided, returning simulated trade history")
                return []
            
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            
            # Format trades
            formatted_trades = []
            for trade in trades:
                formatted_trades.append({
                    "symbol": trade['symbol'],
                    "id": trade['id'],
                    "orderId": trade['order'],
                    "orderListId": -1,
                    "price": str(trade['price']),
                    "qty": str(trade['amount']),
                    "quoteQty": str(trade['cost']),
                    "commission": str(trade.get('fee', {}).get('cost', 0)),
                    "commissionAsset": trade.get('fee', {}).get('currency', ''),
                    "time": trade['timestamp'],
                    "isBuyer": trade['side'] == 'buy',
                    "isMaker": trade.get('maker', False),
                    "isBestMatch": False
                })
            
            return formatted_trades
            
        except Exception as e:
            logging.error(f"Failed to get trade history: {e}")
            return []
    
    def get_positions(self) -> List:
        """Get current positions"""
        try:
            if not self.api_key:
                logging.warning("API key not provided, returning empty positions")
                return []
            
            # Note: This might need adjustment based on Bybit's API
            # For spot trading, positions are typically the balances
            balance = self.exchange.fetch_balance()
            
            positions = []
            for asset, info in balance['total'].items():
                if info > 0:  # Only show assets with balance
                    positions.append({
                        "symbol": f"{asset}/USDT",
                        "positionAmt": str(info),
                        "entryPrice": "0",  # Not available in spot
                        "markPrice": "0",    # Not available in spot
                        "unRealizedProfit": "0",
                        "liquidationPrice": "0",
                        "leverage": "1",
                        "marginType": "isolated",
                        "isolatedMargin": "0",
                        "isAutoAddMargin": "false",
                        "positionSide": "BOTH",
                        "notional": "0",
                        "isolatedWallet": "0",
                        "updateTime": int(time.time() * 1000)
                    })
            
            return positions
            
        except Exception as e:
            logging.error(f"Failed to get positions: {e}")
            return [] 