"""
config module

Configuration and secrets management for the Algo Trading system.
"""

from .secrets import get_api_key, get_api_keys, set_api_keys
from .paths import *

__all__ = ['get_api_key', 'get_api_keys', 'set_api_keys'] 