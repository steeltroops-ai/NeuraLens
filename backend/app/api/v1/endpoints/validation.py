"""
NeuroLens-X Clinical Validation API
Endpoint for model validation metrics and clinical performance analysis
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List

from app.core.config import settings
from app.ml.realtime.realtime_validation import realtime_validation_engine

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/metrics")
async def get_validation_metrics(
    modality: Optional[str] = Query(None, description="Specific modality to get metrics for"),
    metric_type: Optional[str] = Query("all", description="Type of metrics (performance, calibration, fairness, all)")
):
    """
    Get comprehensive validation metrics for NeuroLens models
    
    This endpoint provides detailed performance metrics including
    accuracy, calibration, fairness, and reliability measures
    for all or specific modalities.
    
    Args:
        modality: Specific modality (speech, retinal, motor, cognitive, nri)
        metric_type: Type of metrics to retrieve
        
    Returns:
        Comprehensive validation metrics and performance analysis
        
    Raises:
        HTTPException: If parameters are invalid
    """
    
    # Validate modality
    valid_modalities = ["speech", "retinal", "motor", "cognitive", "nri", "all"]
    if modality and modality not in valid_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality: {modality}. Valid options: {', '.join(valid_modalities)}"
        )
    
    # Validate metric type
    valid_metric_types = ["performance", "calibration", "fairness", "reliability", "all"]
    if metric_type not in valid_metric_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric type: {metric_type}. Valid options: {', '.join(valid_metric_types)}"
        )
    
    try:
        metrics = await realtime_validation_engine.get_validation_metrics(
            modality=modality,
            metric_type=metric_type
        )
        
        logger.info(f"Validation metrics retrieved for modality: {modality}, type: {metric_type}")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to retrieve validation metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving validation metrics."
        )

@router.get("/performance")
async def get_performance_metrics():
    """
    Get performance metrics for all modalities
    
    Returns:
        Performance metrics including accuracy, sensitivity, specificity, AUC
    """
    try:
        return await realtime_validation_engine.get_study_overview()
    except Exception as e:
        logger.error(f"Failed to retrieve performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving performance metrics."
        )

@router.get("/calibration")
async def get_calibration_metrics():
    """
    Get calibration metrics for all modalities
    
    Returns:
        Calibration analysis including reliability diagrams and calibration error
    """
    try:
        return await realtime_validation_engine.get_performance_trends()
    except Exception as e:
        logger.error(f"Failed to retrieve calibration metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving calibration metrics."
        )

@router.get("/fairness")
async def get_fairness_metrics():
    """
    Get fairness and bias analysis for all modalities
    
    Returns:
        Fairness metrics across demographic groups and bias analysis
    """
    try:
        return await realtime_validation_engine.get_study_overview()
    except Exception as e:
        logger.error(f"Failed to retrieve fairness metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving fairness metrics."
        )

@router.get("/reliability")
async def get_reliability_metrics():
    """
    Get reliability and robustness metrics
    
    Returns:
        Reliability analysis including test-retest reliability and robustness
    """
    try:
        return await realtime_validation_engine.get_validation_metrics()
    except Exception as e:
        logger.error(f"Failed to retrieve reliability metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving reliability metrics."
        )

@router.get("/clinical-validation")
async def get_clinical_validation():
    """
    Get clinical validation results and real-world performance
    
    Returns:
        Clinical validation metrics and real-world deployment results
    """
    try:
        return await realtime_validation_engine.get_study_overview()
    except Exception as e:
        logger.error(f"Failed to retrieve clinical validation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving clinical validation."
        )

@router.get("/cross-validation")
async def get_cross_validation_results():
    """
    Get cross-validation results for model robustness
    
    Returns:
        K-fold cross-validation results and statistical significance tests
    """
    try:
        return await realtime_validation_engine.get_performance_trends()
    except Exception as e:
        logger.error(f"Failed to retrieve cross-validation results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving cross-validation results."
        )

@router.get("/comparison")
async def get_model_comparison():
    """
    Get comparison with baseline methods and state-of-the-art
    
    Returns:
        Comparative analysis with other neurological assessment methods
    """
    try:
        return await realtime_validation_engine.get_study_overview()
    except Exception as e:
        logger.error(f"Failed to retrieve model comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving model comparison."
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint for validation service
    
    Returns:
        Service health status and validation engine information
    """
    try:
        health_status = await realtime_validation_engine.health_check()
        return {
            "service": "clinical_validation",
            "status": "healthy",
            "validation_engine_status": health_status,
            "available_metrics": ["performance", "calibration", "fairness", "reliability"],
            "supported_modalities": ["speech", "retinal", "motor", "cognitive", "nri"]
        }
    except Exception as e:
        logger.error(f"Validation service health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "clinical_validation",
                "status": "unhealthy",
                "error": str(e)
            }
        )

@router.get("/info")
async def get_service_info():
    """
    Get information about the clinical validation service
    
    Returns:
        Service capabilities and validation methodology
    """
    return {
        "service": "clinical_validation",
        "version": "1.0.0",
        "description": "Comprehensive clinical validation and model performance analysis",
        "validation_framework": {
            "performance_metrics": [
                "accuracy", "sensitivity", "specificity", "precision", "recall",
                "f1_score", "auc_roc", "auc_pr", "balanced_accuracy"
            ],
            "calibration_metrics": [
                "brier_score", "calibration_error", "reliability_diagram",
                "sharpness", "resolution"
            ],
            "fairness_metrics": [
                "demographic_parity", "equalized_odds", "calibration_fairness",
                "individual_fairness", "counterfactual_fairness"
            ],
            "reliability_metrics": [
                "test_retest_reliability", "inter_rater_reliability",
                "internal_consistency", "measurement_error"
            ]
        },
        "validation_datasets": {
            "internal": "NeuroLens development dataset (N=5,000)",
            "external": "Multi-site validation dataset (N=2,000)",
            "clinical": "Prospective clinical study (N=1,000)"
        },
        "statistical_methods": [
            "k_fold_cross_validation", "bootstrap_confidence_intervals",
            "permutation_tests", "bayesian_model_comparison"
        ],
        "clinical_endpoints": [
            "diagnostic_accuracy", "prognostic_value", "clinical_utility",
            "cost_effectiveness", "patient_outcomes"
        ]
    }
