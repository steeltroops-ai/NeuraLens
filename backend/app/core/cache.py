"""
MediLens Response Caching

Provides in-memory caching for expensive analysis operations.
Used to cache results of identical inputs (e.g., same image hash).
"""

import hashlib
import time
import logging
from typing import TypeVar, Generic, Optional, Dict, Any, Callable
from dataclasses import dataclass
from functools import wraps
import asyncio

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with metadata."""
    value: T
    timestamp: float
    ttl: float
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl
    
    def access(self) -> T:
        self.hits += 1
        return self.value


class TTLCache(Generic[T]):
    """
    Simple TTL-based in-memory cache.
    
    Thread-safe for async use. Automatically evicts expired entries.
    """
    
    def __init__(
        self,
        max_size: int | None = None,
        default_ttl: float | None = None
    ):
        self.max_size = max_size or settings.CACHE_MAX_SIZE
        self.default_ttl = default_ttl or settings.CACHE_TTL
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache if exists and not expired."""
        entry = self._cache.get(key)
        
        if entry is None:
            return None
        
        if entry.is_expired:
            async with self._lock:
                self._cache.pop(key, None)
            return None
        
        return entry.access()
    
    async def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """Set value in cache with TTL."""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                await self._evict_oldest()
            
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            return self._cache.pop(key, None) is not None
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    async def _evict_oldest(self) -> None:
        """Evict the oldest entry (LRU-like)."""
        if not self._cache:
            return
        
        # Remove expired entries first
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            self._cache.pop(key, None)
        
        # If still at capacity, remove oldest
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            self._cache.pop(oldest_key, None)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "entries": [
                {"key": k, "hits": v.hits, "age_s": time.time() - v.timestamp}
                for k, v in self._cache.items()
            ]
        }


# Singleton cache instance for analysis results
_analysis_cache: TTLCache[Dict[str, Any]] | None = None


def get_analysis_cache() -> TTLCache[Dict[str, Any]]:
    """Get the global analysis cache singleton."""
    global _analysis_cache
    if _analysis_cache is None:
        _analysis_cache = TTLCache(
            max_size=settings.CACHE_MAX_SIZE,
            default_ttl=settings.CACHE_TTL
        )
    return _analysis_cache


def compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content for cache key."""
    return hashlib.sha256(content).hexdigest()


def cached_analysis(ttl: float | None = None):
    """
    Decorator to cache analysis results by content hash.
    
    Usage:
        @cached_analysis(ttl=3600)
        async def analyze(content: bytes, **kwargs) -> Dict:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(content: bytes, *args, **kwargs) -> Any:
            cache = get_analysis_cache()
            cache_key = compute_content_hash(content)
            
            # Check cache
            cached = await cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key[:16]}...")
                return cached
            
            # Execute function
            result = await func(content, *args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {cache_key[:16]}...")
            
            return result
        
        return wrapper
    return decorator
