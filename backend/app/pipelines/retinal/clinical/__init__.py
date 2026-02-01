"""
Retinal Pipeline - Clinical Module

Contains clinical assessment components (matching speech/clinical/ structure):
- risk_scorer: Risk assessment with uncertainty
- graders: DR, Glaucoma, AMD grading  
- uncertainty: Calibrated uncertainty estimation

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

# Risk scoring (matching speech/clinical/risk_scorer.py)
from .risk_scorer import (
    RetinalRiskScorer,
    RetinalRiskResult,
    RiskLevel,
    ConditionRisk,
    BiomarkerDeviation,
    retinal_risk_scorer,
)

# Disease graders (matching speech/clinical/condition_classifier.py)
from .graders import (
    DiabeticRetinopathyGrader,
    GlaucomaRiskGrader,
    AMDGrader,
    GradingResult,
    dr_grader as diabetic_retinopathy_grader,
    glaucoma_grader,
    amd_grader,
)

# Uncertainty estimation (matching speech/clinical/uncertainty.py)
from .uncertainty import (
    UncertaintyEstimator,
    UncertaintyEstimate,
    uncertainty_estimator,
)

# Clinical assessment (local module)
from .clinical_assessment import (
    DRGrader,
    DMEAssessor, 
    RiskCalculator,
    ClinicalFindingsGenerator,
    DifferentialGenerator,
    RecommendationGenerator,
    ClinicalSummaryGenerator,
    dr_grader,
    dme_assessor,
    risk_calculator,
    findings_generator,
    differential_generator,
    recommendation_generator,
    summary_generator,
)

# v5.0 Bayesian uncertainty (research-grade)
from .bayesian_uncertainty import (
    BayesianUncertaintyEstimator,
    BayesianUncertaintyResult,
    MCDropoutEstimator,
    ConformalPredictor,
    TemperatureScaler,
    CalibrationMetrics,
    bayesian_estimator,
    calibration_metrics,
)

# v5.0 Clinical safety gates
from .safety_gates import (
    ClinicalSafetyChecker,
    SafetyCheckResult,
    SafetyGateResult,
    GateStatus,
    ClinicalAction,
    ReferralUrgency,
    SafetyThresholds,
    QualityGate,
    UncertaintyGate,
    ReferralGate,
    ConsistencyGate,
    CriticalBiomarkerGate,
    clinical_safety_checker,
)

__all__ = [
    # Risk scoring
    "RetinalRiskScorer",
    "RetinalRiskResult",
    "RiskLevel",
    "ConditionRisk",
    "BiomarkerDeviation",
    "retinal_risk_scorer",
    
    # Graders
    "DiabeticRetinopathyGrader",
    "GlaucomaRiskGrader",
    "AMDGrader",
    "GradingResult",
    "glaucoma_grader",
    "amd_grader",
    
    # Uncertainty
    "UncertaintyEstimator",
    "UncertaintyEstimate",
    "uncertainty_estimator",
    
    # Clinical Assessment
    "DRGrader",
    "DMEAssessor",
    "RiskCalculator",
    "ClinicalFindingsGenerator",
    "DifferentialGenerator",
    "RecommendationGenerator",
    "ClinicalSummaryGenerator",
    "dr_grader",
    "dme_assessor",
    "risk_calculator",
    "findings_generator",
    "differential_generator",
    "recommendation_generator",
    "summary_generator",
    
    # v5.0 Bayesian Uncertainty
    "BayesianUncertaintyEstimator",
    "BayesianUncertaintyResult",
    "MCDropoutEstimator",
    "ConformalPredictor",
    "TemperatureScaler",
    "CalibrationMetrics",
    "bayesian_estimator",
    "calibration_metrics",
    
    # v5.0 Safety Gates
    "ClinicalSafetyChecker",
    "SafetyCheckResult",
    "SafetyGateResult",
    "GateStatus",
    "ClinicalAction",
    "ReferralUrgency",
    "SafetyThresholds",
    "QualityGate",
    "UncertaintyGate",
    "ReferralGate",
    "ConsistencyGate",
    "CriticalBiomarkerGate",
    "clinical_safety_checker",
    
    # v5.1 Enhanced Uncertainty (research-grade)
    "EnhancedUncertaintyEstimator",
    "UncertaintyEstimateV2",
    "SafetyGateResultV2",
    "SafetyLevel",
    "DRConformalPredictor",
    "enhanced_uncertainty_estimator",
]

# v5.1 Enhanced uncertainty (research-grade upgrade)
try:
    from .enhanced_uncertainty import (
        EnhancedUncertaintyEstimator,
        UncertaintyEstimate as UncertaintyEstimateV2,
        SafetyGateResult as SafetyGateResultV2,
        SafetyLevel,
        MCDropoutEstimator as MCDropoutEstimatorV2,
        ConformalPredictor as ConformalPredictorV2,
        DRConformalPredictor,
        TemperatureScaler as TemperatureScalerV2,
        ClinicalSafetyGates,
        enhanced_uncertainty_estimator,
    )
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Enhanced uncertainty module not available: {e}")
