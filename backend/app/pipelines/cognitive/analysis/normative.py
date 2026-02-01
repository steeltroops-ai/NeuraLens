"""
Normative Data Module - Research Grade
Age and education-adjusted normative comparisons for cognitive tests.

References:
- Tombaugh (2004): Trail Making Test normative data
- Woods et al. (2015): Age norms for simple reaction time
- Scarpina & Tagini (2017): Stroop test meta-analysis
- Kirchner (1958), Jaeggi (2010): N-back performance norms
- NACC UDS: National Alzheimer's Coordinating Center norms
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class PerformanceCategory(str, Enum):
    """Clinical performance categories based on percentile."""
    SUPERIOR = "superior"           # >= 95th percentile
    HIGH_AVERAGE = "high_average"   # 75-94th percentile
    AVERAGE = "average"             # 25-74th percentile
    LOW_AVERAGE = "low_average"     # 9-24th percentile
    BORDERLINE = "borderline"       # 2-8th percentile
    IMPAIRED = "impaired"           # < 2nd percentile


@dataclass
class NormativeComparison:
    """Result of normative comparison."""
    raw_score: float
    z_score: float
    percentile: int
    category: PerformanceCategory
    age_group: str
    education_adjusted: bool
    normative_mean: float
    normative_sd: float
    interpretation: str


# =============================================================================
# NORMATIVE DATA TABLES
# Based on published research and meta-analyses
# =============================================================================

# Simple Reaction Time (ms) - Lower is better
# Source: Woods et al. (2015), Der & Deary (2006)
RT_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # age_group: {education_level: (mean_ms, sd_ms)}
    "18-29": {"all": (250.0, 40.0), "high": (240.0, 35.0), "low": (260.0, 45.0)},
    "30-39": {"all": (265.0, 45.0), "high": (255.0, 40.0), "low": (275.0, 50.0)},
    "40-49": {"all": (280.0, 50.0), "high": (270.0, 45.0), "low": (290.0, 55.0)},
    "50-59": {"all": (300.0, 55.0), "high": (285.0, 50.0), "low": (315.0, 60.0)},
    "60-69": {"all": (325.0, 65.0), "high": (305.0, 55.0), "low": (345.0, 70.0)},
    "70-79": {"all": (360.0, 75.0), "high": (335.0, 65.0), "low": (385.0, 85.0)},
    "80+":   {"all": (400.0, 90.0), "high": (370.0, 80.0), "low": (430.0, 100.0)},
}

# Trail Making Test A (seconds) - Lower is better
# Source: Tombaugh (2004), Giovagnoli et al. (1996)
TMT_A_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "18-29": {"all": (22.0, 6.0), "high": (20.0, 5.0), "low": (25.0, 7.0)},
    "30-39": {"all": (24.0, 7.0), "high": (22.0, 6.0), "low": (27.0, 8.0)},
    "40-49": {"all": (28.0, 8.0), "high": (25.0, 7.0), "low": (32.0, 9.0)},
    "50-59": {"all": (32.0, 9.0), "high": (28.0, 8.0), "low": (36.0, 10.0)},
    "60-69": {"all": (39.0, 12.0), "high": (34.0, 10.0), "low": (45.0, 14.0)},
    "70-79": {"all": (48.0, 15.0), "high": (42.0, 13.0), "low": (55.0, 18.0)},
    "80+":   {"all": (60.0, 20.0), "high": (52.0, 17.0), "low": (70.0, 25.0)},
}

# Trail Making Test B (seconds) - Lower is better
# Source: Tombaugh (2004)
TMT_B_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "18-29": {"all": (48.0, 12.0), "high": (42.0, 10.0), "low": (55.0, 15.0)},
    "30-39": {"all": (52.0, 14.0), "high": (46.0, 12.0), "low": (60.0, 16.0)},
    "40-49": {"all": (60.0, 16.0), "high": (52.0, 14.0), "low": (70.0, 20.0)},
    "50-59": {"all": (72.0, 20.0), "high": (62.0, 17.0), "low": (85.0, 25.0)},
    "60-69": {"all": (90.0, 28.0), "high": (75.0, 22.0), "low": (110.0, 35.0)},
    "70-79": {"all": (115.0, 38.0), "high": (95.0, 30.0), "low": (140.0, 48.0)},
    "80+":   {"all": (150.0, 55.0), "high": (125.0, 45.0), "low": (180.0, 70.0)},
}

# Stroop Effect (ms) - Lower is better
# Source: Scarpina & Tagini (2017) meta-analysis
STROOP_EFFECT_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "18-29": {"all": (80.0, 30.0), "high": (70.0, 25.0), "low": (95.0, 35.0)},
    "30-39": {"all": (90.0, 35.0), "high": (80.0, 30.0), "low": (105.0, 40.0)},
    "40-49": {"all": (100.0, 40.0), "high": (85.0, 32.0), "low": (120.0, 48.0)},
    "50-59": {"all": (115.0, 45.0), "high": (95.0, 38.0), "low": (140.0, 55.0)},
    "60-69": {"all": (135.0, 55.0), "high": (110.0, 45.0), "low": (165.0, 68.0)},
    "70-79": {"all": (160.0, 70.0), "high": (130.0, 55.0), "low": (200.0, 90.0)},
    "80+":   {"all": (200.0, 90.0), "high": (160.0, 70.0), "low": (250.0, 115.0)},
}

# N-Back d-prime - Higher is better
# Source: Jaeggi et al. (2010), Harvey (2019)
NBACK_DPRIME_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "18-29": {"all": (2.8, 0.6), "high": (3.0, 0.5), "low": (2.5, 0.7)},
    "30-39": {"all": (2.6, 0.6), "high": (2.8, 0.5), "low": (2.4, 0.7)},
    "40-49": {"all": (2.4, 0.7), "high": (2.6, 0.6), "low": (2.2, 0.8)},
    "50-59": {"all": (2.2, 0.7), "high": (2.4, 0.6), "low": (2.0, 0.8)},
    "60-69": {"all": (1.9, 0.8), "high": (2.1, 0.7), "low": (1.7, 0.9)},
    "70-79": {"all": (1.6, 0.8), "high": (1.8, 0.7), "low": (1.4, 0.9)},
    "80+":   {"all": (1.3, 0.9), "high": (1.5, 0.8), "low": (1.1, 1.0)},
}

# Go/No-Go Commission Errors (proportion) - Lower is better
# Source: Bezdjian et al. (2009), adapted from Continuous Performance Tests
GO_NOGO_COMMISSION_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "18-29": {"all": (0.15, 0.08), "high": (0.12, 0.06), "low": (0.18, 0.10)},
    "30-39": {"all": (0.14, 0.08), "high": (0.11, 0.06), "low": (0.17, 0.10)},
    "40-49": {"all": (0.13, 0.08), "high": (0.10, 0.06), "low": (0.16, 0.10)},
    "50-59": {"all": (0.15, 0.09), "high": (0.12, 0.07), "low": (0.18, 0.11)},
    "60-69": {"all": (0.18, 0.10), "high": (0.14, 0.08), "low": (0.22, 0.12)},
    "70-79": {"all": (0.22, 0.12), "high": (0.17, 0.09), "low": (0.27, 0.14)},
    "80+":   {"all": (0.28, 0.14), "high": (0.22, 0.11), "low": (0.34, 0.17)},
}

# Digit Symbol (items in 90s) - Higher is better
# Source: Wechsler (2008), Lezak et al. (2012)
DIGIT_SYMBOL_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "18-29": {"all": (75.0, 12.0), "high": (80.0, 10.0), "low": (68.0, 14.0)},
    "30-39": {"all": (70.0, 13.0), "high": (76.0, 11.0), "low": (64.0, 15.0)},
    "40-49": {"all": (65.0, 14.0), "high": (72.0, 12.0), "low": (58.0, 16.0)},
    "50-59": {"all": (58.0, 14.0), "high": (65.0, 12.0), "low": (51.0, 16.0)},
    "60-69": {"all": (50.0, 15.0), "high": (57.0, 13.0), "low": (43.0, 17.0)},
    "70-79": {"all": (42.0, 14.0), "high": (48.0, 12.0), "low": (35.0, 16.0)},
    "80+":   {"all": (34.0, 13.0), "high": (40.0, 11.0), "low": (28.0, 15.0)},
}

# Domain composite scores (0-1 scale) - Higher is better
# Synthesized from literature for multi-domain comparison
DOMAIN_COMPOSITE_NORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "memory": {
        "18-29": (0.88, 0.08), "30-39": (0.86, 0.09), "40-49": (0.83, 0.10),
        "50-59": (0.79, 0.11), "60-69": (0.74, 0.12), "70-79": (0.68, 0.13), "80+": (0.62, 0.14)
    },
    "attention": {
        "18-29": (0.90, 0.07), "30-39": (0.88, 0.08), "40-49": (0.85, 0.09),
        "50-59": (0.81, 0.10), "60-69": (0.76, 0.11), "70-79": (0.70, 0.12), "80+": (0.64, 0.13)
    },
    "executive": {
        "18-29": (0.87, 0.09), "30-39": (0.84, 0.10), "40-49": (0.80, 0.11),
        "50-59": (0.75, 0.12), "60-69": (0.69, 0.13), "70-79": (0.62, 0.14), "80+": (0.55, 0.15)
    },
    "processing_speed": {
        "18-29": (0.92, 0.06), "30-39": (0.88, 0.08), "40-49": (0.82, 0.10),
        "50-59": (0.75, 0.12), "60-69": (0.67, 0.13), "70-79": (0.58, 0.14), "80+": (0.50, 0.15)
    },
    "inhibition": {
        "18-29": (0.88, 0.08), "30-39": (0.86, 0.09), "40-49": (0.83, 0.10),
        "50-59": (0.80, 0.11), "60-69": (0.75, 0.12), "70-79": (0.69, 0.13), "80+": (0.63, 0.14)
    }
}


# =============================================================================
# NORMATIVE COMPARISON FUNCTIONS
# =============================================================================

def get_age_group(age: Optional[int]) -> str:
    """Map age to normative age group."""
    if age is None:
        return "50-59"  # Default middle-aged
    
    if age < 30:
        return "18-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80+"


def get_education_level(years: Optional[int]) -> str:
    """Map education years to category."""
    if years is None:
        return "all"
    
    if years >= 16:  # College degree
        return "high"
    elif years <= 12:  # High school or less
        return "low"
    else:
        return "all"


def calculate_z_score(
    raw_score: float,
    normative_mean: float,
    normative_sd: float,
    lower_is_better: bool = False
) -> float:
    """
    Calculate z-score relative to normative data.
    
    Args:
        raw_score: Observed score
        normative_mean: Population mean
        normative_sd: Population SD
        lower_is_better: If True, invert so higher z = better performance
        
    Returns:
        Z-score (negative = below average, positive = above average)
    """
    if normative_sd <= 0:
        return 0.0
    
    z = (raw_score - normative_mean) / normative_sd
    
    if lower_is_better:
        z = -z  # Invert so positive = better
    
    return float(z)


def z_to_percentile(z_score: float) -> int:
    """Convert z-score to percentile."""
    percentile = norm.cdf(z_score) * 100
    return int(max(1, min(99, round(percentile))))


def percentile_to_category(percentile: int) -> PerformanceCategory:
    """Convert percentile to clinical category."""
    if percentile >= 95:
        return PerformanceCategory.SUPERIOR
    elif percentile >= 75:
        return PerformanceCategory.HIGH_AVERAGE
    elif percentile >= 25:
        return PerformanceCategory.AVERAGE
    elif percentile >= 9:
        return PerformanceCategory.LOW_AVERAGE
    elif percentile >= 2:
        return PerformanceCategory.BORDERLINE
    else:
        return PerformanceCategory.IMPAIRED


def generate_interpretation(
    category: PerformanceCategory,
    domain: str,
    percentile: int
) -> str:
    """Generate clinical interpretation text."""
    ordinal = get_ordinal(percentile)
    domain_name = domain.replace("_", " ").title()
    
    interpretations = {
        PerformanceCategory.SUPERIOR: 
            f"{domain_name} performance is in the superior range ({ordinal} percentile), "
            f"indicating exceptional ability in this domain.",
        PerformanceCategory.HIGH_AVERAGE:
            f"{domain_name} performance is in the high average range ({ordinal} percentile), "
            f"indicating above-typical functioning.",
        PerformanceCategory.AVERAGE:
            f"{domain_name} performance is within normal limits ({ordinal} percentile), "
            f"consistent with typical age-matched peers.",
        PerformanceCategory.LOW_AVERAGE:
            f"{domain_name} performance is in the low average range ({ordinal} percentile), "
            f"slightly below but not clinically significant.",
        PerformanceCategory.BORDERLINE:
            f"{domain_name} performance is in the borderline range ({ordinal} percentile), "
            f"suggesting possible concerns requiring monitoring.",
        PerformanceCategory.IMPAIRED:
            f"{domain_name} performance is in the impaired range ({ordinal} percentile), "
            f"indicating significant difficulty warranting clinical attention."
    }
    
    return interpretations.get(category, f"Performance at {ordinal} percentile.")


def get_ordinal(n: int) -> str:
    """Convert number to ordinal string."""
    if 11 <= n <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# =============================================================================
# TASK-SPECIFIC NORMATIVE COMPARISONS
# =============================================================================

def compare_reaction_time(
    mean_rt_ms: float,
    age: Optional[int] = None,
    education_years: Optional[int] = None
) -> NormativeComparison:
    """Compare reaction time to age-adjusted norms."""
    age_group = get_age_group(age)
    edu_level = get_education_level(education_years)
    
    norms = RT_NORMS.get(age_group, RT_NORMS["50-59"])
    mean, sd = norms.get(edu_level, norms["all"])
    
    z = calculate_z_score(mean_rt_ms, mean, sd, lower_is_better=True)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=mean_rt_ms,
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=edu_level != "all",
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, "processing_speed", percentile)
    )


def compare_trail_making_a(
    completion_time_sec: float,
    age: Optional[int] = None,
    education_years: Optional[int] = None
) -> NormativeComparison:
    """Compare Trail Making A to norms."""
    age_group = get_age_group(age)
    edu_level = get_education_level(education_years)
    
    norms = TMT_A_NORMS.get(age_group, TMT_A_NORMS["50-59"])
    mean, sd = norms.get(edu_level, norms["all"])
    
    z = calculate_z_score(completion_time_sec, mean, sd, lower_is_better=True)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=completion_time_sec,
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=edu_level != "all",
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, "visual_attention", percentile)
    )


def compare_trail_making_b(
    completion_time_sec: float,
    age: Optional[int] = None,
    education_years: Optional[int] = None
) -> NormativeComparison:
    """Compare Trail Making B to norms."""
    age_group = get_age_group(age)
    edu_level = get_education_level(education_years)
    
    norms = TMT_B_NORMS.get(age_group, TMT_B_NORMS["50-59"])
    mean, sd = norms.get(edu_level, norms["all"])
    
    z = calculate_z_score(completion_time_sec, mean, sd, lower_is_better=True)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=completion_time_sec,
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=edu_level != "all",
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, "executive_function", percentile)
    )


def compare_stroop_effect(
    stroop_effect_ms: float,
    age: Optional[int] = None,
    education_years: Optional[int] = None
) -> NormativeComparison:
    """Compare Stroop interference effect to norms."""
    age_group = get_age_group(age)
    edu_level = get_education_level(education_years)
    
    norms = STROOP_EFFECT_NORMS.get(age_group, STROOP_EFFECT_NORMS["50-59"])
    mean, sd = norms.get(edu_level, norms["all"])
    
    z = calculate_z_score(stroop_effect_ms, mean, sd, lower_is_better=True)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=stroop_effect_ms,
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=edu_level != "all",
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, "cognitive_flexibility", percentile)
    )


def compare_nback_dprime(
    d_prime: float,
    age: Optional[int] = None,
    education_years: Optional[int] = None
) -> NormativeComparison:
    """Compare N-back d-prime to norms."""
    age_group = get_age_group(age)
    edu_level = get_education_level(education_years)
    
    norms = NBACK_DPRIME_NORMS.get(age_group, NBACK_DPRIME_NORMS["50-59"])
    mean, sd = norms.get(edu_level, norms["all"])
    
    z = calculate_z_score(d_prime, mean, sd, lower_is_better=False)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=d_prime,
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=edu_level != "all",
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, "working_memory", percentile)
    )


def compare_gonogo_commission(
    commission_rate: float,
    age: Optional[int] = None,
    education_years: Optional[int] = None
) -> NormativeComparison:
    """Compare Go/No-Go commission error rate to norms."""
    age_group = get_age_group(age)
    edu_level = get_education_level(education_years)
    
    norms = GO_NOGO_COMMISSION_NORMS.get(age_group, GO_NOGO_COMMISSION_NORMS["50-59"])
    mean, sd = norms.get(edu_level, norms["all"])
    
    z = calculate_z_score(commission_rate, mean, sd, lower_is_better=True)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=commission_rate,
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=edu_level != "all",
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, "inhibitory_control", percentile)
    )


def compare_digit_symbol(
    items_completed: int,
    age: Optional[int] = None,
    education_years: Optional[int] = None
) -> NormativeComparison:
    """Compare Digit Symbol performance to norms."""
    age_group = get_age_group(age)
    edu_level = get_education_level(education_years)
    
    norms = DIGIT_SYMBOL_NORMS.get(age_group, DIGIT_SYMBOL_NORMS["50-59"])
    mean, sd = norms.get(edu_level, norms["all"])
    
    z = calculate_z_score(float(items_completed), mean, sd, lower_is_better=False)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=float(items_completed),
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=edu_level != "all",
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, "processing_speed", percentile)
    )


def compare_domain_composite(
    score: float,
    domain: str,
    age: Optional[int] = None
) -> NormativeComparison:
    """Compare domain composite score to norms."""
    age_group = get_age_group(age)
    
    domain_norms = DOMAIN_COMPOSITE_NORMS.get(domain, DOMAIN_COMPOSITE_NORMS.get("attention"))
    if not domain_norms:
        # Fallback
        mean, sd = 0.75, 0.12
    else:
        mean, sd = domain_norms.get(age_group, (0.75, 0.12))
    
    z = calculate_z_score(score, mean, sd, lower_is_better=False)
    percentile = z_to_percentile(z)
    category = percentile_to_category(percentile)
    
    return NormativeComparison(
        raw_score=score,
        z_score=z,
        percentile=percentile,
        category=category,
        age_group=age_group,
        education_adjusted=False,
        normative_mean=mean,
        normative_sd=sd,
        interpretation=generate_interpretation(category, domain, percentile)
    )


# =============================================================================
# COMPOSITE SCORING
# =============================================================================

def calculate_global_cognitive_index(
    domain_comparisons: Dict[str, NormativeComparison],
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, int, PerformanceCategory]:
    """
    Calculate weighted global cognitive index from domain comparisons.
    
    Args:
        domain_comparisons: Dict of domain -> NormativeComparison
        weights: Optional domain weights (default: equal weighting)
        
    Returns:
        Tuple of (global_z, global_percentile, global_category)
    """
    if not domain_comparisons:
        return (0.0, 50, PerformanceCategory.AVERAGE)
    
    if weights is None:
        weights = {domain: 1.0 for domain in domain_comparisons}
    
    total_weight = 0.0
    weighted_z_sum = 0.0
    
    for domain, comparison in domain_comparisons.items():
        w = weights.get(domain, 1.0)
        weighted_z_sum += comparison.z_score * w
        total_weight += w
    
    if total_weight == 0:
        return (0.0, 50, PerformanceCategory.AVERAGE)
    
    global_z = weighted_z_sum / total_weight
    global_percentile = z_to_percentile(global_z)
    global_category = percentile_to_category(global_percentile)
    
    return (global_z, global_percentile, global_category)
