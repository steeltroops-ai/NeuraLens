"""
MediLens Backend - FastAPI Application
AI-Powered Medical Diagnostics Platform

Performance optimizations:
- Request timeout middleware
- Request timing headers
- Connection pooling (in database.py)
- Background task execution
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_db
from app.routers import router
from app.middleware import TimeoutMiddleware, RequestTimingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info(f"[START] {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"[ENV] {settings.ENVIRONMENT} (DEBUG={settings.DEBUG})")
    
    # Initialize database
    await init_db()
    
    yield
    
    # Cleanup
    logger.info("[STOP] Shutting down...")
    
    # Shutdown thread executors
    try:
        from app.core.async_utils import shutdown_executors
        shutdown_executors()
    except ImportError:
        pass
    
    logger.info("[STOP] Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-Powered Multi-Modal Medical Diagnostics",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,  # Disable docs in production
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Performance Middleware (order matters - first added = outermost)
# 1. Request timing (measures total time including middleware)
app.add_middleware(RequestTimingMiddleware)

# 2. Request timeout enforcement
if not settings.DEBUG:  # Only enforce timeout in production
    app.add_middleware(TimeoutMiddleware, timeout=settings.REQUEST_TIMEOUT)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs" if settings.DEBUG else "disabled",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    response = {
        "status": "ok",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }
    
    # Include cache stats in debug mode
    if settings.DEBUG:
        try:
            from app.core.cache import get_analysis_cache
            cache = get_analysis_cache()
            response["cache"] = cache.stats
        except ImportError:
            pass
    
    return response


# Include all pipeline routes
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug" if settings.DEBUG else "info"
    )

