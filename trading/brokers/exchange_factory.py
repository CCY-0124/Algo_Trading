"""
Exchange Factory for managing different exchange connections.
Updated to use Bybit with CCXT.
"""

from typing import Dict, Optional
from .bybit_client import BybitClient

class ExchangeFactory:
    """
    Factory class for creating exchange clients.
    
    Features:
    - Unified interface for different exchanges
    - Configuration management
    - Client initialization
    - Bybit-focused implementation
    """
    
    def __init__(self):
        """Initialize exchange factory"""
        self.clients: Dict[str, any] = {}
        self.supported_exchanges = ["bybit"]
    
    def create_client(
        self,
        exchange: str = "bybit",
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True
    ):
        """
        Create exchange client.
        
        :param exchange: Exchange name (default: 'bybit')
        :param api_key: API key
        :param api_secret: API secret
        :param testnet: Use testnet
        :return: Exchange client
        """
        if exchange.lower() == "bybit":
            client = BybitClient(api_key, api_secret, testnet)
            self.clients[exchange] = client
            return client
        else:
            raise ValueError(f"Unsupported exchange: {exchange}. Only 'bybit' is supported.")
    
    def get_client(self, exchange: str = "bybit"):
        """Get existing client for exchange"""
        return self.clients.get(exchange)
    
    def list_supported_exchanges(self) -> list:
        """List supported exchanges"""
        return self.supported_exchanges.copy()
    
    def remove_client(self, exchange: str = "bybit"):
        """Remove client for exchange"""
        if exchange in self.clients:
            del self.clients[exchange]
    
    def get_all_clients(self) -> Dict[str, any]:
        """Get all active clients"""
        return self.clients.copy()
    
    def get_bybit_client(self):
        """Get Bybit client specifically"""
        return self.get_client("bybit") 