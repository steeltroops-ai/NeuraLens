"""
Performance Optimization Module for Retinal Analysis Pipeline

Implements:
- Redis caching for results (Requirements 8.8, 10.4)
- ONNX Runtime for faster inference (Requirements 4.4, 10.2)
- Request queueing under load (Requirement 10.6)
- Horizontal scaling triggers (Requirement 10.7)
- Batch processing support

Author: NeuraLens Team
"""

import logging
import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import hashlib
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Configuration
# ============================================================================

class CacheBackend(str, Enum):
    """Available cache backends"""
    MEMORY = "memory"
    REDIS = "redis"


@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: CacheBackend = CacheBackend.MEMORY
    redis_url: str = "redis://localhost:6379"
    default_ttl_seconds: int = 3600  # 1 hour
    max_memory_items: int = 1000
    enable_compression: bool = True


# ============================================================================
# In-Memory Cache (Fallback)
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 3600
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds


class MemoryCache:
    """
    Thread-safe in-memory cache with LRU eviction.
    
    Used as fallback when Redis is unavailable.
    """
    
    def __init__(self, max_items: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_items = max_items
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set value in cache with TTL"""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_items and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(value=value, ttl_seconds=ttl_seconds)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._cache:
            return
        
        # Find LRU item
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[lru_key]
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            return {
                "total_items": len(self._cache),
                "expired_items": expired,
                "max_items": self._max_items,
                "hit_rate": self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if not self._cache:
            return 0.0
        total_access = sum(e.access_count for e in self._cache.values())
        return min(total_access / (len(self._cache) + 1), 1.0)


# ============================================================================
# Unified Cache Service
# ============================================================================

class CacheService:
    """
    Unified caching service with Redis and memory fallback.
    
    Requirements 8.8, 10.4: Cache frequently accessed data
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._memory_cache = MemoryCache(self.config.max_memory_items)
        self._redis_client = None
        self._redis_available = False
        
        if self.config.backend == CacheBackend.REDIS:
            self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            self._redis_client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            self._redis_available = True
            logger.info("Redis cache initialized successfully")
        except ImportError:
            logger.warning("Redis library not installed, falling back to memory cache")
            self._redis_available = False
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to memory cache")
            self._redis_available = False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.
        
        Requirement 8.8: Fast retrieval of cached results
        """
        cache_key = self._make_key(key)
        
        if self._redis_available:
            try:
                value = await self._redis_client.get(cache_key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fallback to memory cache
        return self._memory_cache.get(cache_key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set cached value with TTL"""
        cache_key = self._make_key(key)
        ttl = ttl_seconds or self.config.default_ttl_seconds
        
        if self._redis_available:
            try:
                await self._redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Also set in memory cache for fast access
        self._memory_cache.set(cache_key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        cache_key = self._make_key(key)
        deleted = False
        
        if self._redis_available:
            try:
                await self._redis_client.delete(cache_key)
                deleted = True
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        return self._memory_cache.delete(cache_key) or deleted
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        ttl_seconds: Optional[int] = None
    ) -> Any:
        """Get from cache or compute and cache if missing"""
        value = await self.get(key)
        if value is not None:
            return value
        
        value = await factory()
        await self.set(key, value, ttl_seconds)
        return value
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key"""
        return f"neuralens:retinal:{key}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "backend": self.config.backend.value,
            "memory": self._memory_cache.stats()
        }
        
        if self._redis_available:
            try:
                info = await self._redis_client.info("memory")
                stats["redis"] = {
                    "connected": True,
                    "used_memory": info.get("used_memory_human", "unknown")
                }
            except Exception:
                stats["redis"] = {"connected": False}
        
        return stats


# ============================================================================
# Request Queue for Load Management
# ============================================================================

@dataclass
class QueuedRequest:
    """Queued analysis request"""
    request_id: str
    patient_id: str
    priority: int = 0
    queued_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


class RequestQueue:
    """
    Request queue for managing analysis workload.
    
    Requirement 10.6: Queue requests during high load
    """
    
    SCALE_THRESHOLD = 50  # Requirement 10.7: Scale at 50 items
    
    def __init__(self, max_concurrent: int = 4):
        self._queue: deque[QueuedRequest] = deque()
        self._processing: Dict[str, QueuedRequest] = {}
        self._completed: Dict[str, QueuedRequest] = {}
        self._max_concurrent = max_concurrent
        self._lock = asyncio.Lock()
        self._scale_callbacks: List[Callable[[int], Awaitable[None]]] = []
    
    async def enqueue(
        self,
        request_id: str,
        patient_id: str,
        priority: int = 0
    ) -> int:
        """
        Add request to queue.
        
        Returns position in queue.
        """
        async with self._lock:
            request = QueuedRequest(
                request_id=request_id,
                patient_id=patient_id,
                priority=priority
            )
            
            # Insert by priority
            inserted = False
            for i, r in enumerate(self._queue):
                if priority > r.priority:
                    self._queue.insert(i, request)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(request)
            
            position = list(self._queue).index(request) + 1
            
            # Check scale threshold
            queue_size = len(self._queue)
            if queue_size >= self.SCALE_THRESHOLD:
                await self._trigger_scaling(queue_size)
            
            return position
    
    async def dequeue(self) -> Optional[QueuedRequest]:
        """Get next request from queue if capacity available"""
        async with self._lock:
            if len(self._processing) >= self._max_concurrent:
                return None
            
            if not self._queue:
                return None
            
            request = self._queue.popleft()
            request.status = "processing"
            request.started_at = datetime.utcnow()
            self._processing[request.request_id] = request
            
            return request
    
    async def complete(
        self,
        request_id: str,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """Mark request as completed"""
        async with self._lock:
            if request_id in self._processing:
                request = self._processing.pop(request_id)
                request.status = "completed" if not error else "failed"
                request.completed_at = datetime.utcnow()
                request.result = result
                request.error = error
                self._completed[request_id] = request
    
    async def get_position(self, request_id: str) -> Optional[int]:
        """
        Get current position in queue.
        
        Requirement 10.6: Display queue position
        """
        async with self._lock:
            for i, r in enumerate(self._queue):
                if r.request_id == request_id:
                    return i + 1
            
            if request_id in self._processing:
                return 0  # Currently processing
            
            return None  # Not found or completed
    
    async def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a request"""
        async with self._lock:
            # Check queue
            for i, r in enumerate(self._queue):
                if r.request_id == request_id:
                    return {
                        "status": "queued",
                        "position": i + 1,
                        "estimated_wait": self._estimate_wait(i + 1)
                    }
            
            # Check processing
            if request_id in self._processing:
                r = self._processing[request_id]
                return {
                    "status": "processing",
                    "started_at": r.started_at.isoformat() if r.started_at else None
                }
            
            # Check completed
            if request_id in self._completed:
                r = self._completed[request_id]
                return {
                    "status": r.status,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "processing_time_ms": self._calc_processing_time(r)
                }
            
            return None
    
    def _estimate_wait(self, position: int) -> int:
        """Estimate wait time in seconds based on position"""
        avg_processing_time = 5  # seconds
        return position * avg_processing_time // self._max_concurrent
    
    def _calc_processing_time(self, request: QueuedRequest) -> Optional[int]:
        """Calculate processing time in milliseconds"""
        if request.started_at and request.completed_at:
            delta = request.completed_at - request.started_at
            return int(delta.total_seconds() * 1000)
        return None
    
    async def _trigger_scaling(self, queue_size: int) -> None:
        """
        Trigger horizontal scaling.
        
        Requirement 10.7: Auto-scale at 50 queue items
        """
        logger.warning(f"Queue size {queue_size} exceeds threshold, triggering scaling")
        
        for callback in self._scale_callbacks:
            try:
                await callback(queue_size)
            except Exception as e:
                logger.error(f"Scale callback failed: {e}")
    
    def on_scale_needed(
        self,
        callback: Callable[[int], Awaitable[None]]
    ) -> None:
        """Register callback for scaling events"""
        self._scale_callbacks.append(callback)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        async with self._lock:
            return {
                "queued": len(self._queue),
                "processing": len(self._processing),
                "completed_recent": len(self._completed),
                "max_concurrent": self._max_concurrent,
                "scale_threshold": self.SCALE_THRESHOLD,
                "needs_scaling": len(self._queue) >= self.SCALE_THRESHOLD
            }


# ============================================================================
# ONNX Runtime Optimizer
# ============================================================================

class ONNXOptimizer:
    """
    ONNX Runtime wrapper for accelerated inference.
    
    Requirements 4.4, 10.2: Fast ML inference with GPU support
    """
    
    TARGET_INFERENCE_TIME_MS = 500  # Requirement 4.4
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        self._session = None
        self._providers = self._select_providers(use_gpu)
        self._model_path = model_path
        
        if model_path:
            self._load_model(model_path)
    
    def _select_providers(self, use_gpu: bool) -> List[str]:
        """Select ONNX execution providers"""
        if use_gpu:
            return [
                'CUDAExecutionProvider',
                'TensorrtExecutionProvider',
                'CPUExecutionProvider'
            ]
        return ['CPUExecutionProvider']
    
    def _load_model(self, model_path: str) -> None:
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4
            
            self._session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=self._providers
            )
            
            logger.info(f"ONNX model loaded: {model_path}")
            logger.info(f"Providers: {self._session.get_providers()}")
            
        except ImportError:
            logger.warning("ONNX Runtime not installed")
            self._session = None
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self._session = None
    
    def is_available(self) -> bool:
        """Check if ONNX inference is available"""
        return self._session is not None
    
    def infer(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run inference with performance tracking.
        
        Requirement 4.4: < 500ms inference time
        """
        if not self._session:
            return None
        
        start = time.perf_counter()
        
        try:
            outputs = self._session.run(None, inputs)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if elapsed_ms > self.TARGET_INFERENCE_TIME_MS:
                logger.warning(
                    f"Inference time {elapsed_ms:.1f}ms exceeds target "
                    f"{self.TARGET_INFERENCE_TIME_MS}ms"
                )
            
            return {
                "outputs": outputs,
                "inference_time_ms": elapsed_ms
            }
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return None
    
    def get_input_names(self) -> List[str]:
        """Get model input names"""
        if not self._session:
            return []
        return [i.name for i in self._session.get_inputs()]
    
    def get_output_names(self) -> List[str]:
        """Get model output names"""
        if not self._session:
            return []
        return [o.name for o in self._session.get_outputs()]


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor:
    """
    Batch processing for bulk operations.
    
    Requirement 10.2: Efficient processing of multiple images
    """
    
    def __init__(self, batch_size: int = 4, max_wait_seconds: float = 1.0):
        self._batch_size = batch_size
        self._max_wait = max_wait_seconds
        self._pending: List[Any] = []
        self._lock = asyncio.Lock()
    
    async def add(self, item: Any) -> None:
        """Add item to batch"""
        async with self._lock:
            self._pending.append(item)
    
    async def process_batch(
        self,
        processor: Callable[[List[Any]], Awaitable[List[Any]]]
    ) -> List[Any]:
        """Process accumulated batch"""
        async with self._lock:
            if not self._pending:
                return []
            
            batch = self._pending[:self._batch_size]
            self._pending = self._pending[self._batch_size:]
        
        return await processor(batch)
    
    def pending_count(self) -> int:
        """Get number of pending items"""
        return len(self._pending)


# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self._timings: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def record(self, duration_ms: float) -> None:
        """Record a timing measurement"""
        with self._lock:
            self._timings.append(duration_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        with self._lock:
            if not self._timings:
                return {"count": 0}
            
            timings = list(self._timings)
            return {
                "count": len(timings),
                "avg_ms": sum(timings) / len(timings),
                "min_ms": min(timings),
                "max_ms": max(timings),
                "p50_ms": sorted(timings)[len(timings) // 2],
                "p95_ms": sorted(timings)[int(len(timings) * 0.95)]
            }


# ============================================================================
# Singleton Instances
# ============================================================================

cache_service = CacheService()
request_queue = RequestQueue()
performance_monitor = PerformanceMonitor()
