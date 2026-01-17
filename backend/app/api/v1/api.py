"""
API Router for MediLens v1
Multi-modal diagnostic assessment endpoints
"""

from fastapi import APIRouter
from app.api.v1.endpoints import speech

# Create main API router
api_router = APIRouter()

# Include speech analysis endpoint
api_router.include_router(
    speech.router,
    prefix="/speech",
    tags=["Speech Analysis"]
)

# API status endpoint
@api_router.get("/status")
async def api_status():
    """Get API status and available endpoints"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "speech": "/api/v1/speech"
        },
        "features": {
            "real_time_processing": True,
            "clinical_validation": True
        }
    }
