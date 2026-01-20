"""
Retinal Explanation Rules (Legacy Compatibility)

This file provides backward compatibility for the explanation system.
The main implementation is in explanation/rules.py following the new architecture.
"""

# Re-export from the new architecture location
from .explanation.rules import (
    BIOMARKER_EXPLANATIONS,
    RISK_LEVEL_MESSAGES,
    CONDITION_EXPLANATIONS,
    MANDATORY_DISCLAIMER,
    QUALITY_WARNINGS,
    DR_GRADE_EXPLANATIONS,
    generate_retinal_explanation,
    RetinalExplanationGenerator,
)

__all__ = [
    "BIOMARKER_EXPLANATIONS",
    "RISK_LEVEL_MESSAGES",
    "CONDITION_EXPLANATIONS",
    "MANDATORY_DISCLAIMER",
    "QUALITY_WARNINGS",
    "DR_GRADE_EXPLANATIONS",
    "generate_retinal_explanation",
    "RetinalExplanationGenerator",
]
