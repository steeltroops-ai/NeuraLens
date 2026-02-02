"""
Clinical Risk Assessment Module v4.0
Research-grade risk scoring with uncertainty quantification.

Components:
- ClinicalRiskScorer: Base risk calculation with normative data
- UncertaintyEstimator: Monte Carlo uncertainty quantification
- ConditionClassifier: Multi-condition probability estimation
- RiskExplainer: Feature-based explanations (LIME/SHAP-style)
"""

from .risk_scorer import (
    ClinicalRiskScorer,
    RiskResult,
    RiskAssessmentResult,
    ConditionRiskResult,
    BiomarkerDeviation,
    RiskLevel,
    calculate_speech_risk,
    get_biomarker_status,
    get_risk_category
)
from .uncertainty import UncertaintyEstimator, UncertaintyResult
from .explainer import RiskExplainer, RiskExplanation
from .normative import NormativeDataManager, NormativeReference

__all__ = [
    # Core scoring
    "ClinicalRiskScorer",
    "RiskResult",
    "RiskAssessmentResult",
    "ConditionRiskResult",
    "BiomarkerDeviation",
    "RiskLevel",
    
    # Utility functions
    "calculate_speech_risk",
    "get_biomarker_status",
    "get_risk_category",
    
    # Uncertainty
    "UncertaintyEstimator",
    "UncertaintyResult",
    
    # Explainability
    "RiskExplainer",
    "RiskExplanation",
    
    # Normative data
    "NormativeDataManager",
    "NormativeReference",
]
