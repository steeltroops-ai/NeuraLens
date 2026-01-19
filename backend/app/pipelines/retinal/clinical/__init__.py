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
]
