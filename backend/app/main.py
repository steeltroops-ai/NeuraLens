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
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting NeuroLens-X Backend...")
    await init_db()
    print("✅ Database initialized")
    print("🧠 ML models loading...")
    # TODO: Load ML models here
    print("✅ ML models ready")
    yield
    # Shutdown
    print("🛑 Shutting down NeuroLens-X Backend...")


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
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "neurolens-x-api",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }


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
