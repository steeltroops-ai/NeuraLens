"""
Cognitive Pipeline Analysis Module - Research Grade

Sub-modules:
- statistics: Robust statistical analysis (RT, SDT, fatigue)
- normative: Age-adjusted normative comparisons
- analyzer: Real-time cognitive analysis

References:
- Ratcliff (1993): RT analysis
- Stanislaw & Todorov (1999): Signal detection
- Tombaugh (2004): Normative data
"""

from .statistics import (
    calculate_rt_statistics,
    calculate_signal_detection_metrics,
    calculate_fatigue_index,
    calculate_consistency_score,
    calculate_learning_slope,
    RTStatistics,
    SignalDetectionMetrics,
    OutlierMethod
)

from .normative import (
    compare_reaction_time,
    compare_trail_making_a,
    compare_trail_making_b,
    compare_stroop_effect,
    compare_nback_dprime,
    compare_gonogo_commission,
    compare_digit_symbol,
    compare_domain_composite,
    calculate_global_cognitive_index,
    get_age_group,
    get_education_level,
    NormativeComparison,
    PerformanceCategory
)

__all__ = [
    # Statistics
    "calculate_rt_statistics",
    "calculate_signal_detection_metrics",
    "calculate_fatigue_index",
    "calculate_consistency_score",
    "calculate_learning_slope",
    "RTStatistics",
    "SignalDetectionMetrics",
    "OutlierMethod",
    # Normative
    "compare_reaction_time",
    "compare_trail_making_a",
    "compare_trail_making_b",
    "compare_stroop_effect",
    "compare_nback_dprime",
    "compare_gonogo_commission",
    "compare_digit_symbol",
    "compare_domain_composite",
    "calculate_global_cognitive_index",
    "get_age_group",
    "get_education_level",
    "NormativeComparison",
    "PerformanceCategory"
]
