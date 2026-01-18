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
    
    def reset_stats(self):
        """
        Reset statistics (keep cache data).
        """
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self.session_start = datetime.now()
        logging.info("Cache statistics reset")
    
    def get_cached_factors(self) -> list:
        """
        Get list of currently cached factor IDs.
        
        :return: List of factor IDs (in LRU order, oldest first)
        """
        return list(self.cache.keys())
    
    def preload(self, factor_ids: list, load_function: Callable = None):
        """
        Preload multiple factors into cache.
        
        :param factor_ids: List of factor IDs to preload
        :param load_function: Optional load function
        """
        logging.info(f"Preloading {len(factor_ids)} factors into cache...")
        
        loader = load_function or self.load_function
        if not loader:
            logging.warning("No load function provided, cannot preload")
            return
        
        loaded = 0
        for factor_id in factor_ids:
            if factor_id not in self.cache:
                try:
                    data = loader(factor_id)
                    if data is not None:
                        self.put(factor_id, data)
                        loaded += 1
                except Exception as e:
                    logging.error(f"Error preloading {factor_id}: {e}")
        
        logging.info(f"Preloaded {loaded}/{len(factor_ids)} factors")


# Global cache instance for easy access
_global_cache = None

def get_cache(max_size: int = 50, load_function: Callable = None) -> LightweightDataCache:
    """
    Get global cache instance.
    
    :param max_size: Maximum cache size (only used if creating new instance)
    :param load_function: Load function (only used if creating new instance)
    :return: Global LightweightDataCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = LightweightDataCache(max_size=max_size, load_function=load_function)
    return _global_cache


if __name__ == "__main__":
    # Test the data cache
    import time
    
    print("Testing Lightweight Data Cache...")
    
    # Create test load function
    def test_load_function(factor_id):
        """Simulate loading data (with delay)"""
        time.sleep(0.1)  # Simulate slow SSD read
        return f"Data for {factor_id}"
    
    # Create cache
    cache = LightweightDataCache(max_size=5, load_function=test_load_function)
    
    # Test: Load and cache
    print("\n1. Testing cache put/get...")
    start = time.time()
    data1 = cache.get("factor_1")
    time1 = time.time() - start
    print(f"   First load (from SSD): {time1:.3f}s")
    
    start = time.time()
    data2 = cache.get("factor_1")
    time2 = time.time() - start
    print(f"   Second load (from cache): {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.1f}x faster")
    
    # Test: Cache eviction
    print("\n2. Testing cache eviction...")
    for i in range(7):
        cache.put(f"factor_{i}", f"data_{i}")
    
    print(f"   Cache size: {cache.size()}/{cache.max_size}")
    print(f"   Cached factors: {cache.get_cached_factors()}")
    
    # Test: Access pattern (LRU)
    print("\n3. Testing LRU behavior...")
    cache.get("factor_2")  # Access factor_2, should move to end
    print(f"   After accessing factor_2: {cache.get_cached_factors()}")
    
    # Test: Statistics
    print("\n4. Testing statistics...")
    cache.print_stats()
    
    # Test: Preload
    print("\n5. Testing preload...")
    cache.clear()
    cache.preload(["factor_A", "factor_B", "factor_C"])
    print(f"   Preloaded factors: {cache.get_cached_factors()}")
    
    print("\n✅ All cache tests completed!")
