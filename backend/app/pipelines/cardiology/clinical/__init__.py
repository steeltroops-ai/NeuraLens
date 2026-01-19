"""
Cardiology Pipeline - Clinical Module
Risk scoring, grading, and clinical recommendations.
"""

from .risk_scorer import (
    CardiacRiskScorer,
    RiskAssessment,
    compute_risk_score,
)

from .recommendations import (
    RecommendationGenerator,
    generate_recommendations,
)

__all__ = [
    # Risk Scoring
    "CardiacRiskScorer",
    "RiskAssessment",
    "compute_risk_score",
    
    # Recommendations
    "RecommendationGenerator",
    "generate_recommendations",
]
