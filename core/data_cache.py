"""
data_cache.py

Lightweight session-level data cache for optimizing SSD data access.

Features:
- In-memory cache for current session
- LRU (Least Recently Used) eviction policy
- Maximum 50 factors cached
- No persistence (cleared on session end)
- Simple and low overhead
"""

import logging
from typing import Dict, Optional, Any, Callable
from collections import OrderedDict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LightweightDataCache:
    """
    Lightweight session-level data cache.
    
    Designed for scenarios where:
    - Same factor data may be read multiple times during optimization
    - Data is read from external SSD (slower)
    - Cache only needed for current session
    """
    
    def __init__(self, max_size: int = 50, load_function: Callable = None):
        """
        Initialize lightweight data cache.
        
        :param max_size: Maximum number of factors to cache (default: 50)
        :param load_function: Function to load data if not in cache
                              Should accept factor_id and return data
        """
        self.max_size = max_size
        self.load_function = load_function
        
        # Use OrderedDict for LRU implementation
        # OrderedDict maintains insertion order, useful for LRU
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self.session_start = datetime.now()
        
        logging.info(f"Lightweight Data Cache initialized")
        logging.info(f"  Max size: {max_size} factors")
        logging.info(f"  Load function: {'Provided' if load_function else 'Not provided'}")
    
    def get(self, factor_id: str, load_function: Callable = None) -> Optional[Any]:
        """
        Get factor data from cache or load if not present.
        
        :param factor_id: Factor identifier
        :param load_function: Optional load function (overrides instance function)
        :return: Factor data or None if not found and no load function
        """
        self.stats['total_requests'] += 1
        
        # Check cache
        if factor_id in self.cache:
            # Cache hit: move to end (most recently used)
            data = self.cache.pop(factor_id)
            self.cache[factor_id] = data
            self.stats['hits'] += 1
            logging.debug(f"Cache HIT: {factor_id}")
            return data
        
        # Cache miss
        self.stats['misses'] += 1
        logging.debug(f"Cache MISS: {factor_id}")
        
        # Try to load data
        loader = load_function or self.load_function
        if loader:
            try:
                data = loader(factor_id)
                if data is not None:
                    self.put(factor_id, data)
                return data
            except Exception as e:
                logging.error(f"Error loading data for {factor_id}: {e}")
                return None
        
        return None
    
    def put(self, factor_id: str, data: Any):
        """
        Put factor data into cache.
        
        :param factor_id: Factor identifier
        :param data: Factor data to cache
        """
        # If already in cache, remove it first (will be re-added at end)
        if factor_id in self.cache:
            self.cache.pop(factor_id)
        
        # If cache is full, evict least recently used (first item)
        if len(self.cache) >= self.max_size:
            evicted_id, _ = self.cache.popitem(last=False)  # Remove first (oldest)
            self.stats['evictions'] += 1
            logging.debug(f"Cache eviction: {evicted_id}")
        
        # Add to cache (at end, most recently used)
        self.cache[factor_id] = data
        logging.debug(f"Cached: {factor_id} (cache size: {len(self.cache)}/{self.max_size})")
    
    def remove(self, factor_id: str) -> bool:
        """
        Remove factor from cache.
        
        :param factor_id: Factor identifier
        :return: True if removed, False if not in cache
        """
        if factor_id in self.cache:
            self.cache.pop(factor_id)
            logging.debug(f"Removed from cache: {factor_id}")
            return True
        return False
    
    def clear(self):
        """
        Clear all cached data.
        """
        count = len(self.cache)
        self.cache.clear()
        logging.info(f"Cache cleared: {count} factors removed")
    
    def contains(self, factor_id: str) -> bool:
        """
        Check if factor is in cache.
        
        :param factor_id: Factor identifier
        :return: True if in cache, False otherwise
        """
        return factor_id in self.cache
    
    def size(self) -> int:
        """
        Get current cache size.
        
        :return: Number of factors in cache
        """
        return len(self.cache)
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        :return: Statistics dictionary
        """
        hit_rate = (
            self.stats['hits'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0.0
        )
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            'session_start': self.session_start.isoformat(),
            'session_duration_seconds': session_duration,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'total_requests': self.stats['total_requests'],
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate': hit_rate,
            'miss_rate': 1.0 - hit_rate
        }
    
    def print_stats(self):
        """
        Print cache statistics to console.
        """
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("DATA CACHE STATISTICS")
        print("=" * 60)
        print(f"Session Duration: {stats['session_duration_seconds']:.2f}s")
        print(f"Cache Size: {stats['cache_size']}/{stats['max_size']}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Hits: {stats['hits']} ({stats['hit_rate']:.2%})")
        print(f"Misses: {stats['misses']} ({stats['miss_rate']:.2%})")
        print(f"Evictions: {stats['evictions']}")
        print("=" * 60)
    
