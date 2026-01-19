"""
Retinal Pipeline - Explanation Module

AI explanation rules for retinal analysis results.
"""

from .rules import (
    UrgencyLevel,
    RiskLevel,
    BiomarkerExplanation,
    BIOMARKER_EXPLANATIONS,
    DR_GRADE_EXPLANATIONS,
    RISK_LEVEL_MESSAGES,
    CONDITION_EXPLANATIONS,
    QUALITY_WARNINGS,
    MANDATORY_DISCLAIMER,
    RetinalExplanationGenerator,
    generate_retinal_explanation,
)

# Backward compatibility alias
RETINAL_EXPLANATION_RULES = {
    "biomarkers": BIOMARKER_EXPLANATIONS,
    "dr_grades": DR_GRADE_EXPLANATIONS,
    "risk_levels": RISK_LEVEL_MESSAGES,
    "conditions": CONDITION_EXPLANATIONS,
    "quality_warnings": QUALITY_WARNINGS,
}

__all__ = [
    'UrgencyLevel',
    'RiskLevel',
    'BiomarkerExplanation',
    'BIOMARKER_EXPLANATIONS',
    'DR_GRADE_EXPLANATIONS',
    'RISK_LEVEL_MESSAGES',
    'CONDITION_EXPLANATIONS',
    'QUALITY_WARNINGS',
    'MANDATORY_DISCLAIMER',
    'RetinalExplanationGenerator',
    'generate_retinal_explanation',
    'RETINAL_EXPLANATION_RULES',
]
