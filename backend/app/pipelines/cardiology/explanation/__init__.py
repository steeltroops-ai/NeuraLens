"""
Cardiology Pipeline - AI Explanation Module

Rules and templates for generating human-readable explanations
of cardiology analysis results.
"""

from ..explanation_rules import (
    CARDIOLOGY_EXPLANATION_RULES,
    generate_cardiology_explanation,
)

__all__ = [
    'CARDIOLOGY_EXPLANATION_RULES',
    'generate_cardiology_explanation',
]
