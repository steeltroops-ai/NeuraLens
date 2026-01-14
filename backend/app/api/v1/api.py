"""
API Router for NeuroLens-X v1
Hackathon-optimized multi-modal assessment endpoints
"""

from fastapi import APIRouter

from app.pipelines import speech, retinal, motor, cognitive, nri, validation


# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    speech.router,
    prefix="/speech",
    tags=["Speech Analysis"]
)

api_router.include_router(
    retinal.router,
    prefix="/retinal",
    tags=["Retinal Analysis"]
)

api_router.include_router(
    motor.router,
    prefix="/motor",
    tags=["Motor Assessment"]
)

api_router.include_router(
    cognitive.router,
    prefix="/cognitive",
    tags=["Cognitive Evaluation"]
)

api_router.include_router(
    nri.router,
    prefix="/nri",
    tags=["NRI Fusion"]
)

api_router.include_router(
    validation.router,
    prefix="/validation",
    tags=["Clinical Validation"]
)


# API status endpoint
@api_router.get("/status")
async def api_status():
    """Get API status and available endpoints"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "speech": "/api/v1/speech",
            "retinal": "/api/v1/retinal", 
            "motor": "/api/v1/motor",
            "cognitive": "/api/v1/cognitive",
            "nri": "/api/v1/nri",
            "validation": "/api/v1/validation"
        },
        "features": {
            "multi_modal_fusion": True,
            "real_time_processing": True,
            "clinical_validation": True,
            "uncertainty_quantification": True
        }
    }
