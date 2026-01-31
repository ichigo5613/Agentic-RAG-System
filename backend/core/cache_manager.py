# backend/core/cache_manager.py
from typing import Any, Optional, List
import time
from collections import OrderedDict

from backend.config import config
from backend.utils.logger import logger

class CacheManager:
    """LRU cache manager with TTL"""
    
    def __init__(self, max_size: int = None, ttl: int = None):
        self.max_size = max_size or config.MAX_CACHE_SIZE
        self.ttl = ttl or config.CACHE_TTL
        self.cache = OrderedDict()
        logger.info(f"Initialized CacheManager (max_size={self.max_size}, ttl={self.ttl}s)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Check TTL
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            logger.debug(f"Cache expired for key: {key[:50]}...")
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return value
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache full, removed oldest key: {oldest_key[:50]}...")
        
        # Add new item
        self.cache[key] = (value, time.time())
        logger.debug(f"Cached value for key: {key[:50]}...")
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
    
    def stats(self) -> dict:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "oldest": next(iter(self.cache)) if self.cache else None,
            "newest": next(reversed(self.cache)) if self.cache else None
        }