"""
core module

Core components for the Algo Trading system including backtesting engine and data loading.
"""

from .enhanced_engine import EnhancedBacktestEngine
from .dataloader import DataLoader

__all__ = ['EnhancedBacktestEngine', 'DataLoader']


