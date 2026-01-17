"""
Radiology X-Ray Router - FastAPI endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import time

from .analyzer import XRayAnalyzer

router = APIRouter(prefix="/radiology", tags=["Radiology"])

# Initialize analyzer (singleton)
analyzer = XRayAnalyzer()


class RadiologyResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    processing_time_ms: int


@router.post("/analyze", response_model=RadiologyResponse)
async def analyze_xray(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)")
):
    """
    Analyze chest X-ray for pulmonary and cardiac conditions
    
    Detects 14+ conditions including:
    - Pneumonia
    - Cardiomegaly
    - Pleural Effusion
    - Consolidation
    - Atelectasis
    - Nodules/Masses
    - Pneumothorax
    
    Returns findings with confidence scores and heatmap overlay
    """
    start_time = time.time()
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image (JPEG, PNG)")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Validate size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(400, "File too large. Maximum 10MB.")
        
        # Run analysis
        result = analyzer.analyze(image_bytes)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return RadiologyResponse(
            success=True,
            data=result.to_dict(),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.get("/conditions")
async def list_conditions():
    """List all detectable conditions"""
    return {
        "conditions": [
            {"name": "Pneumonia", "category": "Infection", "urgency": "High"},
            {"name": "Cardiomegaly", "category": "Cardiac", "urgency": "Medium"},
            {"name": "Effusion", "category": "Fluid", "urgency": "Medium"},
            {"name": "Consolidation", "category": "Opacity", "urgency": "High"},
            {"name": "Atelectasis", "category": "Collapse", "urgency": "Medium"},
            {"name": "Nodule", "category": "Mass", "urgency": "Medium"},
            {"name": "Mass", "category": "Mass", "urgency": "High"},
            {"name": "Pneumothorax", "category": "Collapse", "urgency": "Critical"},
            {"name": "Emphysema", "category": "COPD", "urgency": "Low"},
            {"name": "Fibrosis", "category": "Scarring", "urgency": "Low"},
        ],
        "total": 14
    }


@router.get("/health")
async def health_check():
    """Health check for radiology module"""
    try:
        import torchxrayvision
        xrv_status = "available"
    except ImportError:
        xrv_status = "not installed"
    
    return {
        "status": "healthy",
        "module": "radiology",
        "model": "TorchXRayVision DenseNet121" if xrv_status == "available" else "Simulation",
        "dependencies": {
            "torchxrayvision": xrv_status
        }
    }


@router.get("/info")
async def module_info():
    """Get information about radiology module"""
    return {
        "name": "CXR-Insight AI",
        "description": "AI-powered chest X-ray analysis",
        "model": "DenseNet121 trained on 8 datasets",
        "datasets": [
            "NIH ChestX-ray14",
            "CheXpert",
            "MIMIC-CXR",
            "PadChest"
        ],
        "supported_formats": ["JPEG", "PNG", "DICOM (coming soon)"],
        "max_file_size": "10MB"
    }
