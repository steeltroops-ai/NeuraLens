"""
Cardiology ECG Router - FastAPI endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import time
import numpy as np

from .analyzer import ECGAnalyzer, parse_ecg_file, generate_demo_ecg

router = APIRouter(prefix="/cardiology", tags=["Cardiology"])


class ECGResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    processing_time_ms: int


@router.post("/analyze", response_model=ECGResponse)
async def analyze_ecg(
    file: UploadFile = File(...),
    sample_rate: int = Query(500, description="ECG sample rate in Hz")
):
    """
    Analyze uploaded ECG file for cardiac abnormalities
    
    Supports: CSV, TXT files with ECG signal data
    Returns: Heart rate, rhythm classification, risk level, recommendations
    """
    start_time = time.time()
    
    try:
        # Read file
        content = await file.read()
        
        # Parse ECG data
        ecg_signal = parse_ecg_file(content, file.filename or "data.csv", sample_rate)
        
        # Analyze
        analyzer = ECGAnalyzer(sample_rate=sample_rate)
        result = analyzer.analyze(ecg_signal)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ECGResponse(
            success=True,
            data=result.to_dict(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/demo", response_model=ECGResponse)
async def demo_analysis(
    heart_rate: int = Query(72, ge=40, le=200, description="Simulated heart rate"),
    duration: float = Query(10, ge=5, le=60, description="Signal duration in seconds")
):
    """
    Generate demo ECG analysis with synthetic signal
    
    Use this for testing without real ECG data
    """
    start_time = time.time()
    
    try:
        # Generate synthetic ECG
        ecg_signal = generate_demo_ecg(
            sample_rate=500,
            duration=duration,
            heart_rate=heart_rate
        )
        
        # Analyze
        analyzer = ECGAnalyzer(sample_rate=500)
        result = analyzer.analyze(ecg_signal)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ECGResponse(
            success=True,
            data=result.to_dict(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for cardiology module"""
    try:
        import heartpy
        heartpy_status = "available"
    except ImportError:
        heartpy_status = "not installed"
    
    try:
        import neurokit2
        neurokit_status = "available"
    except ImportError:
        neurokit_status = "not installed"
    
    return {
        "status": "healthy",
        "module": "cardiology",
        "dependencies": {
            "heartpy": heartpy_status,
            "neurokit2": neurokit_status
        }
    }


@router.get("/info")
async def module_info():
    """Get information about cardiology module capabilities"""
    return {
        "name": "CardioPredict AI",
        "description": "AI-powered ECG analysis for cardiac conditions",
        "supported_conditions": [
            "Normal Sinus Rhythm",
            "Sinus Bradycardia",
            "Sinus Tachycardia",
            "Atrial Fibrillation (screening)",
            "Heart Rate Variability analysis"
        ],
        "input_formats": ["CSV", "TXT"],
        "sample_rate_range": "100-1000 Hz",
        "libraries_used": ["HeartPy", "NeuroKit2"]
    }
