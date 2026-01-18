"""
Voice Pipeline - Audio Cache
In-memory caching for generated audio to reduce API calls
"""

import hashlib
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    audio_bytes: bytes
    provider: str
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    @property
    def size(self) -> int:
        return len(self.audio_bytes)
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class AudioCache:
    """
    LRU cache for generated audio to reduce API calls
    
    Features:
    - Size-limited cache (default 100MB)
    - LRU eviction
    - TTL expiration
    - Hit rate tracking
    """
    
    DEFAULT_MAX_SIZE_MB = 100
    DEFAULT_TTL_SECONDS = 3600  # 1 hour
    
    def __init__(
        self,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        ttl_seconds: int = DEFAULT_TTL_SECONDS
    ):
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _generate_key(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0
    ) -> str:
        """Generate unique cache key from parameters"""
        content = f"{text}:{voice_id}:{speed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0
    ) -> Optional[Tuple[bytes, str]]:
        """
        Retrieve cached audio if available
        
        Args:
            text: Original text
            voice_id: Voice identifier
            speed: Speech speed
            
        Returns:
            Tuple of (audio_bytes, provider) or None if not cached
        """
        key = self._generate_key(text, voice_id, speed)
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if entry.age_seconds > self.ttl:
            self._remove(key)
            self.misses += 1
            return None
        
        # Update access info (LRU)
        entry.access_count += 1
        entry.last_accessed = time.time()
        self.cache.move_to_end(key)
        
        self.hits += 1
        logger.debug(f"Cache hit for key {key[:8]}...")
        
        return (entry.audio_bytes, entry.provider)
    
    def set(
        self,
        text: str,
        voice_id: str,
        speed: float,
        audio_bytes: bytes,
        provider: str
    ) -> bool:
        """
        Cache audio bytes
        
        Args:
            text: Original text
            voice_id: Voice identifier
            speed: Speech speed
            audio_bytes: Generated audio data
            provider: TTS provider used
            
        Returns:
            True if cached successfully
        """
        key = self._generate_key(text, voice_id, speed)
        audio_size = len(audio_bytes)
        
        # Don't cache if single item exceeds limit
        if audio_size > self.max_size:
            logger.warning(f"Audio too large to cache: {audio_size} bytes")
            return False
        
        # Evict entries until we have space
        while self.current_size + audio_size > self.max_size:
            self._evict_oldest()
        
        # Remove existing entry if present
        if key in self.cache:
            self._remove(key)
        
        # Add new entry
        entry = CacheEntry(
            audio_bytes=audio_bytes,
            provider=provider,
            created_at=time.time()
        )
        
        self.cache[key] = entry
        self.current_size += audio_size
        
        logger.debug(f"Cached audio: {audio_size} bytes, key {key[:8]}...")
        return True
    
    def _remove(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            self.current_size -= self.cache[key].size
            del self.cache[key]
    
    def _evict_oldest(self):
        """Evict the least recently used entry"""
        if not self.cache:
            return
        
        # Get oldest (first) item
        oldest_key = next(iter(self.cache))
        logger.debug(f"Evicting cache entry: {oldest_key[:8]}...")
        self._remove(oldest_key)
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.current_size = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "entries": len(self.cache),
            "size_bytes": self.current_size,
            "size_mb": round(self.current_size / (1024 * 1024), 2),
            "max_size_mb": self.max_size // (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 1),
            "ttl_seconds": self.ttl
        }


class UsageTracker:
    """
    Track TTS API usage for cost management
    
    ElevenLabs pricing:
    - Free tier: 10,000 characters/month
    - After: ~$0.30 per 1,000 characters
    """
    
    ELEVENLABS_FREE_CHARS = 10000  # per month
    ELEVENLABS_COST_PER_1K = 0.30  # USD after free tier
    
    def __init__(self):
        self.monthly_chars: Dict[str, int] = {}  # provider -> chars
        self.monthly_reset = datetime.now().replace(day=1, hour=0, minute=0, second=0)
    
    def track_usage(self, char_count: int, provider: str):
        """Track character usage"""
        # Reset monthly counters if new month
        now = datetime.now()
        if now.month != self.monthly_reset.month or now.year != self.monthly_reset.year:
            self.monthly_chars = {}
            self.monthly_reset = now.replace(day=1, hour=0, minute=0, second=0)
        
        # Track usage
        if provider not in self.monthly_chars:
            self.monthly_chars[provider] = 0
        self.monthly_chars[provider] += char_count
        
        # Log warning if approaching limits
        if provider == "elevenlabs":
            used = self.monthly_chars[provider]
            if used > self.ELEVENLABS_FREE_CHARS * 0.8:
                logger.warning(
                    f"Approaching ElevenLabs free tier limit: "
                    f"{used}/{self.ELEVENLABS_FREE_CHARS} chars"
                )
    
    def should_use_fallback(self) -> bool:
        """Determine if should use gTTS to save costs"""
        elevenlabs_usage = self.monthly_chars.get("elevenlabs", 0)
        return elevenlabs_usage > self.ELEVENLABS_FREE_CHARS * 0.9
    
    def get_usage(self) -> Dict:
        """Get current usage stats"""
        elevenlabs_chars = self.monthly_chars.get("elevenlabs", 0)
        gtts_chars = self.monthly_chars.get("gtts", 0)
        
        return {
            "elevenlabs": {
                "chars_used": elevenlabs_chars,
                "free_tier_limit": self.ELEVENLABS_FREE_CHARS,
                "free_tier_remaining": max(0, self.ELEVENLABS_FREE_CHARS - elevenlabs_chars),
                "estimated_cost_usd": max(0, (elevenlabs_chars - self.ELEVENLABS_FREE_CHARS) / 1000 * self.ELEVENLABS_COST_PER_1K)
            },
            "gtts": {
                "chars_used": gtts_chars,
                "cost": 0  # Free
            },
            "reset_date": self.monthly_reset.isoformat()
        }


# Global instances
audio_cache = AudioCache()
usage_tracker = UsageTracker()
