"""
NeuroLens-X FastAPI Backend
Hackathon-optimized multi-modal neurological risk assessment API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.response import (
    api_exception_handler,
    health_check_response,
    api_status_response,
    ResponseBuilder
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("[START] Starting NeuroLens-X Backend...")
    print("[DB] Connecting to Neon PostgreSQL database...")
    await init_db()
    print("[OK] Database initialized")
    print("[ML] ML models loading...")
    # TODO: Load ML models here
    print("[OK] ML models ready")
    yield
    # Shutdown
    print("[STOP] Shutting down NeuroLens-X Backend...")
    await close_db()
    print("[OK] Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="NeuroLens-X API",
    description="Multi-modal neurological risk assessment platform",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Health check endpoint
@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint for monitoring"""
    from app.core.database import DatabaseManager

    # Check database health
    db_health = DatabaseManager.health_check()

    components = {
        "database": db_health["status"],
        "ml_models": "healthy",  # TODO: Add actual ML model health checks
        "api": "healthy"
    }

    overall_status = "healthy" if all(
        status == "healthy" for status in components.values()
    ) else "unhealthy"

    return health_check_response(
        status=overall_status,
        service="NeuraLens-X API",
        environment=settings.ENVIRONMENT,
        components=components
    )

# API status endpoint
@app.get("/api/v1/status")
async def api_status():
    """API status endpoint"""
    return api_status_response()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NeuroLens-X API",
        "description": "Multi-modal neurological risk assessment platform",
        "version": "1.0.0",
        "docs": "/docs" if settings.ENVIRONMENT == "development" else "Contact admin for API documentation",
        "health": "/health"
    }


# Add exception handlers
app.add_exception_handler(HTTPException, api_exception_handler)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for production error handling"""
    if settings.ENVIRONMENT == "development":
        # In development, show detailed errors
        raise exc
    else:
        # In production, return generic error
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": getattr(request.state, "request_id", None)
            }
        )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )
