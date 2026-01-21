"""
MediLens Request Middleware

Provides performance and security middleware:
- Request timeout enforcement
- Request logging with timing
- Rate limiting (optional)
"""

import asyncio
import time
import logging
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import settings

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Enforce request timeout for long-running requests.
    
    Prevents requests from running indefinitely and blocking workers.
    """
    
    def __init__(self, app, timeout: float | None = None):
        super().__init__(app)
        self.timeout = timeout or settings.REQUEST_TIMEOUT
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout: {request.method} {request.url.path} "
                f"exceeded {self.timeout}s"
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Gateway Timeout",
                    "detail": f"Request exceeded {self.timeout}s timeout",
                    "path": str(request.url.path)
                }
            )


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Log request timing for performance monitoring.
    
    Adds X-Process-Time header to responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = (time.perf_counter() - start_time) * 1000  # ms
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        # Log slow requests
        if process_time > 1000:  # > 1 second
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.2f}ms"
            )
        elif settings.DEBUG:
            logger.debug(
                f"{request.method} {request.url.path} - {process_time:.2f}ms"
            )
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Limit request body size to prevent abuse.
    
    Default: 50MB for file uploads
    """
    
    def __init__(self, app, max_size: int = 50 * 1024 * 1024):
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable):
        content_length = request.headers.get("content-length")
        
        if content_length and int(content_length) > self.max_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Payload Too Large",
                    "detail": f"Request body exceeds {self.max_size // (1024*1024)}MB limit",
                    "max_size_bytes": self.max_size
                }
            )
        
        return await call_next(request)
