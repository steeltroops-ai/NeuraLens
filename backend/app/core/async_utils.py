"""
MediLens Async Utilities

Provides utilities for running CPU-bound tasks in async context:
- ThreadPoolExecutor for CPU-bound operations
- ProcessPoolExecutor for heavy computation (optional)
- Async wrappers for synchronous functions
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, TypeVar, ParamSpec, Coroutine, Any
from functools import wraps
import logging

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

# Global thread pool for CPU-bound tasks
_thread_executor: ThreadPoolExecutor | None = None


def get_thread_executor() -> ThreadPoolExecutor:
    """Get the global thread pool executor singleton."""
    global _thread_executor
    if _thread_executor is None:
        _thread_executor = ThreadPoolExecutor(
            max_workers=settings.WORKER_THREADS,
            thread_name_prefix="medilens-worker"
        )
    return _thread_executor


async def run_in_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """
    Run a synchronous function in a thread pool.
    
    Use this for CPU-bound operations that would block the async event loop:
    - Image processing (PIL, OpenCV)
    - Audio processing (librosa, parselmouth)
    - Heavy computation
    
    Usage:
        result = await run_in_thread(process_image, image_bytes)
    """
    loop = asyncio.get_running_loop()
    executor = get_thread_executor()
    
    # Wrap with kwargs support
    if kwargs:
        return await loop.run_in_executor(
            executor,
            lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(executor, func, *args)


def async_wrap(func: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
    """
    Decorator to wrap a synchronous function for async execution.
    
    Usage:
        @async_wrap
        def heavy_computation(data: bytes) -> dict:
            # Synchronous code
            return result
        
        # Now can be awaited
        result = await heavy_computation(data)
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await run_in_thread(func, *args, **kwargs)
    return wrapper


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    error_message: str = "Operation timed out"
) -> T:
    """
    Run a coroutine with a timeout.
    
    Raises TimeoutError with custom message if timeout is exceeded.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(error_message)


def shutdown_executors():
    """Shutdown all thread pool executors gracefully."""
    global _thread_executor
    if _thread_executor is not None:
        _thread_executor.shutdown(wait=True)
        _thread_executor = None
        logger.info("Thread pool executor shutdown complete")
