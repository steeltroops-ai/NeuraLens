"""
Cognitive Statistics Module - Research Grade
Statistical utilities for cognitive test analysis.

References:
- Ratcliff (1993): Outlier trimming for RT analysis
- Whelan (2008): Efficient RT analysis procedures
- Stanislaw & Todorov (1999): d-prime calculations
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OutlierMethod(str, Enum):
    """Methods for outlier detection in reaction time data."""
    NONE = "none"
    SD_CUTOFF = "sd_cutoff"  # Standard deviation based
    MAD = "mad"  # Median absolute deviation (robust)
    IQR = "iqr"  # Interquartile range
    RECURSIVE = "recursive"  # Recursive trimming


@dataclass
class RTStatistics:
    """
    Comprehensive reaction time statistics.
    
    Based on standard psychometric practices for RT analysis:
    - Central tendency: mean, median, trimmed mean
    - Variability: SD, CV, IQR
    - Distribution: skewness, kurtosis
    - Performance indices: inverse efficiency score
    """
    n_trials: int
    n_valid: int
    n_outliers: int
    
    # Central tendency
    mean_rt: float
    median_rt: float
    trimmed_mean_rt: float  # 10% trimmed
    
    # Variability
    std_rt: float
    coefficient_of_variation: float  # CV = std/mean
    iqr_rt: float  # Interquartile range
    
    # Distribution shape
    min_rt: float
    max_rt: float
    skewness: float
    kurtosis: float
    
    # Performance indices
    inverse_efficiency_score: Optional[float] = None  # Mean RT / Accuracy
    tau: Optional[float] = None  # Ex-Gaussian tau (slow tail)
    
    # Lapses (cognitive slips)
    lapse_count: int = 0
    lapse_threshold_ms: float = 500.0


@dataclass
class SignalDetectionMetrics:
    """
    Signal detection theory metrics for memory/attention tasks.
    
    Based on:
    - Macmillan & Creelman (2005): Detection Theory
    - Stanislaw & Todorov (1999): d-prime calculation
    """
    hits: int
    misses: int
    false_alarms: int
    correct_rejections: int
    
    hit_rate: float
    false_alarm_rate: float
    
    # Sensitivity (discriminability)
    d_prime: float
    
    # Response bias
    criterion_c: float  # Negative = liberal, Positive = conservative
    beta: float  # Likelihood ratio
    
    # Accuracy metrics
    accuracy: float
    balanced_accuracy: float  # (sensitivity + specificity) / 2


def calculate_rt_statistics(
    reaction_times: List[float],
    accuracy: float = 1.0,
    outlier_method: OutlierMethod = OutlierMethod.MAD,
    outlier_threshold: float = 2.5,
    min_rt: float = 100.0,
    max_rt: float = 2000.0,
    lapse_threshold: float = 500.0
) -> RTStatistics:
    """
    Calculate comprehensive RT statistics with outlier handling.
    
    Args:
        reaction_times: List of reaction times in milliseconds
        accuracy: Proportion correct (0-1) for inverse efficiency
        outlier_method: Method for detecting outliers
        outlier_threshold: Threshold for outlier detection
        min_rt: Minimum valid RT (physiological floor ~100ms)
        max_rt: Maximum valid RT (attentional lapse ceiling)
        lapse_threshold: RT above this is considered a lapse
        
    Returns:
        RTStatistics with all computed metrics
        
    Scientific Notes:
    - Using MAD for outlier detection as it's robust to non-normal distributions
    - Trimmed mean uses 10% trimming (5% each tail) as per convention
    - CV (coefficient of variation) normalizes variability by mean
    """
    if not reaction_times:
        return _empty_rt_statistics()
    
    rts = np.array(reaction_times, dtype=np.float64)
    n_total = len(rts)
    
    # Step 1: Apply physiological bounds
    valid_mask = (rts >= min_rt) & (rts <= max_rt)
    rts_bounded = rts[valid_mask]
    
    if len(rts_bounded) < 3:
        return _empty_rt_statistics()
    
    # Step 2: Outlier detection and removal
    if outlier_method == OutlierMethod.MAD:
        clean_rts, n_outliers = _remove_outliers_mad(rts_bounded, outlier_threshold)
    elif outlier_method == OutlierMethod.SD_CUTOFF:
        clean_rts, n_outliers = _remove_outliers_sd(rts_bounded, outlier_threshold)
    elif outlier_method == OutlierMethod.IQR:
        clean_rts, n_outliers = _remove_outliers_iqr(rts_bounded, outlier_threshold)
    else:
        clean_rts = rts_bounded
        n_outliers = 0
    
    if len(clean_rts) < 3:
        return _empty_rt_statistics()
    
    # Step 3: Calculate statistics
    mean_rt = float(np.mean(clean_rts))
    median_rt = float(np.median(clean_rts))
    std_rt = float(np.std(clean_rts, ddof=1))  # Sample SD
    
    # Trimmed mean (10% each tail)
    trim_n = max(1, int(len(clean_rts) * 0.1))
    sorted_rts = np.sort(clean_rts)
    trimmed_rts = sorted_rts[trim_n:-trim_n] if len(sorted_rts) > 2 * trim_n else sorted_rts
    trimmed_mean_rt = float(np.mean(trimmed_rts))
    
    # Coefficient of variation
    cv = std_rt / mean_rt if mean_rt > 0 else 0.0
    
    # IQR
    q75, q25 = np.percentile(clean_rts, [75, 25])
    iqr = float(q75 - q25)
    
    # Distribution shape
    skewness = _calculate_skewness(clean_rts)
    kurt = _calculate_kurtosis(clean_rts)
    
    # Inverse Efficiency Score (Townsend & Ashby, 1983)
    # IES = Mean RT / Proportion Correct
    ies = mean_rt / accuracy if accuracy > 0 else None
    
    # Lapse count
    lapse_count = int(np.sum(clean_rts > lapse_threshold))
    
    # Ex-Gaussian tau estimation (simplified)
    tau = _estimate_tau(clean_rts)
    
    return RTStatistics(
        n_trials=n_total,
        n_valid=len(clean_rts),
        n_outliers=n_outliers + (n_total - len(rts_bounded)),
        mean_rt=mean_rt,
        median_rt=median_rt,
        trimmed_mean_rt=trimmed_mean_rt,
        std_rt=std_rt,
        coefficient_of_variation=cv,
        iqr_rt=iqr,
        min_rt=float(np.min(clean_rts)),
        max_rt=float(np.max(clean_rts)),
        skewness=skewness,
        kurtosis=kurt,
        inverse_efficiency_score=ies,
        tau=tau,
        lapse_count=lapse_count,
        lapse_threshold_ms=lapse_threshold
    )


def calculate_signal_detection_metrics(
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
    correction: str = "loglinear"
) -> SignalDetectionMetrics:
    """
    Calculate signal detection theory metrics.
    
    Args:
        hits: Correct positive responses
        misses: Missed targets
        false_alarms: Incorrect positive responses
        correct_rejections: Correct rejections
        correction: Method for handling extreme rates
            - "loglinear": Add 0.5 to all cells (Hautus, 1995)
            - "bound": Bound rates to [0.01, 0.99]
            
    Returns:
        SignalDetectionMetrics with d', c, beta
        
    Scientific Notes:
    - d-prime measures sensitivity (ability to discriminate signal from noise)
    - Criterion c measures response bias (liberal vs conservative)
    - Beta measures likelihood ratio decision criterion
    """
    from scipy.stats import norm
    
    total_signal = hits + misses
    total_noise = false_alarms + correct_rejections
    
    if total_signal == 0 or total_noise == 0:
        return _empty_sdt_metrics(hits, misses, false_alarms, correct_rejections)
    
    # Calculate rates with correction for extreme values
    if correction == "loglinear":
        # Loglinear correction (Hautus, 1995)
        hr = (hits + 0.5) / (total_signal + 1)
        far = (false_alarms + 0.5) / (total_noise + 1)
    else:
        # Bounding correction
        hr = max(0.01, min(0.99, hits / total_signal))
        far = max(0.01, min(0.99, false_alarms / total_noise))
    
    # d-prime: z(hit rate) - z(false alarm rate)
    z_hr = norm.ppf(hr)
    z_far = norm.ppf(far)
    d_prime = float(z_hr - z_far)
    
    # Criterion c: -0.5 * (z(HR) + z(FAR))
    # Negative = liberal bias (tend to say "yes")
    # Positive = conservative bias (tend to say "no")
    criterion_c = float(-0.5 * (z_hr + z_far))
    
    # Beta: likelihood ratio
    # exp(-0.5 * (z_hr^2 - z_far^2))
    beta = float(np.exp(-0.5 * (z_hr**2 - z_far**2)))
    
    # Accuracy metrics
    total = hits + misses + false_alarms + correct_rejections
    accuracy = (hits + correct_rejections) / total if total > 0 else 0.0
    
    # Balanced accuracy (accounts for class imbalance)
    sensitivity = hits / total_signal if total_signal > 0 else 0.0
    specificity = correct_rejections / total_noise if total_noise > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return SignalDetectionMetrics(
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_rejections=correct_rejections,
        hit_rate=float(hr),
        false_alarm_rate=float(far),
        d_prime=d_prime,
        criterion_c=criterion_c,
        beta=beta,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy
    )


def calculate_learning_slope(
    trial_scores: List[float],
    method: str = "linear"
) -> Tuple[float, float]:
    """
    Calculate learning slope across trials.
    
    Args:
        trial_scores: Performance scores per trial/block
        method: "linear" or "exponential"
        
    Returns:
        Tuple of (slope, r_squared)
        
    Scientific Notes:
    - Positive slope indicates improvement (learning)
    - Negative slope indicates fatigue/decline
    - R-squared indicates reliability of trend
    """
    if len(trial_scores) < 3:
        return (0.0, 0.0)
    
    x = np.arange(len(trial_scores))
    y = np.array(trial_scores)
    
    if method == "linear":
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        
        # R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return (slope, float(r_squared))
    else:
        # Exponential learning curve: y = a * (1 - exp(-b*x)) + c
        # Simplified: use log-linear fit
        try:
            y_shifted = y - np.min(y) + 1
            log_y = np.log(y_shifted)
            coeffs = np.polyfit(x, log_y, 1)
            slope = float(coeffs[0])
            return (slope, 0.5)  # Approximate R2
        except Exception:
            return (0.0, 0.0)


def calculate_fatigue_index(
    performance_scores: List[float],
    block_size: int = 5
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate fatigue index using block-wise analysis.
    
    Args:
        performance_scores: Sequential performance scores
        block_size: Number of trials per block
        
    Returns:
        Tuple of (fatigue_index, details)
        
    Scientific Notes:
    - Compares first half vs second half performance
    - Also calculates linear decline slope
    - Fatigue index: (first_block - last_block) / first_block
    """
    if len(performance_scores) < block_size * 2:
        return (0.0, {"method": "insufficient_data"})
    
    scores = np.array(performance_scores)
    
    # Block-wise means
    n_blocks = len(scores) // block_size
    block_means = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_means.append(float(np.mean(scores[start:end])))
    
    if len(block_means) < 2:
        return (0.0, {"method": "insufficient_blocks"})
    
    first_block = block_means[0]
    last_block = block_means[-1]
    
    # Fatigue index (proportion decline)
    if first_block > 0:
        fatigue_index = (first_block - last_block) / first_block
    else:
        fatigue_index = 0.0
    
    # Clamp to [0, 1] - negative means improvement
    fatigue_index = max(0.0, min(1.0, fatigue_index))
    
    # Linear slope
    slope, r2 = calculate_learning_slope(block_means)
    
    return (fatigue_index, {
        "method": "block_comparison",
        "first_block_mean": first_block,
        "last_block_mean": last_block,
        "n_blocks": len(block_means),
        "decline_slope": -slope,  # Positive = decline
        "slope_r_squared": r2
    })


def calculate_consistency_score(
    values: List[float],
    expected_range: Tuple[float, float] = (0.0, 100.0)
) -> float:
    """
    Calculate consistency score based on performance variability.
    
    Args:
        values: Performance values to assess
        expected_range: Expected range for normalization
        
    Returns:
        Consistency score (0-1, higher = more consistent)
        
    Scientific Notes:
    - Uses coefficient of variation (CV) as base
    - Transforms to 0-1 scale where 1 = perfectly consistent
    - CV > 0.5 is considered highly variable in cognitive testing
    """
    if len(values) < 2:
        return 1.0
    
    arr = np.array(values)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    
    if mean_val == 0:
        return 1.0 if std_val == 0 else 0.0
    
    cv = std_val / mean_val
    
    # Transform: CV of 0 -> consistency 1.0, CV of 0.5+ -> consistency ~0
    consistency = np.exp(-2 * cv)  # Exponential decay
    
    return float(max(0.0, min(1.0, consistency)))


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================

def _remove_outliers_mad(data: np.ndarray, threshold: float = 2.5) -> Tuple[np.ndarray, int]:
    """Remove outliers using Median Absolute Deviation (robust method)."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Scale factor for normal distribution equivalence
    # MAD * 1.4826 approximates SD for normal data
    if mad == 0:
        return data, 0
    
    modified_z = 0.6745 * (data - median) / mad
    mask = np.abs(modified_z) < threshold
    
    return data[mask], int(np.sum(~mask))


def _remove_outliers_sd(data: np.ndarray, threshold: float = 2.5) -> Tuple[np.ndarray, int]:
    """Remove outliers using standard deviation cutoff."""
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return data, 0
    
    z_scores = (data - mean) / std
    mask = np.abs(z_scores) < threshold
    
    return data[mask], int(np.sum(~mask))


def _remove_outliers_iqr(data: np.ndarray, threshold: float = 1.5) -> Tuple[np.ndarray, int]:
    """Remove outliers using IQR method."""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    lower = q25 - threshold * iqr
    upper = q75 + threshold * iqr
    
    mask = (data >= lower) & (data <= upper)
    
    return data[mask], int(np.sum(~mask))


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate Fisher-Pearson skewness coefficient."""
    n = len(data)
    if n < 3:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return 0.0
    
    skew = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
    return float(skew)


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis (Fisher definition)."""
    n = len(data)
    if n < 4:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return 0.0
    
    m4 = np.mean((data - mean) ** 4)
    m2 = np.mean((data - mean) ** 2)
    
    if m2 == 0:
        return 0.0
    
    kurt = (m4 / m2 ** 2) - 3  # Excess kurtosis
    return float(kurt)


def _estimate_tau(data: np.ndarray) -> Optional[float]:
    """
    Estimate tau parameter from ex-Gaussian distribution.
    
    Tau represents the exponential component (slow tail) of RT distribution.
    Higher tau = more attentional lapses.
    
    Uses simplified method of moments estimator.
    """
    if len(data) < 10:
        return None
    
    try:
        mean = np.mean(data)
        std = np.std(data)
        skew = _calculate_skewness(data)
        
        # For ex-Gaussian, tau is related to skewness
        # Simplified: tau ~ (skew * std) / 2 for positive skew
        if skew > 0:
            tau = (skew * std) / 2
            return float(tau)
        return 0.0
    except Exception:
        return None


def _empty_rt_statistics() -> RTStatistics:
    """Return empty RTStatistics for invalid data."""
    return RTStatistics(
        n_trials=0,
        n_valid=0,
        n_outliers=0,
        mean_rt=0.0,
        median_rt=0.0,
        trimmed_mean_rt=0.0,
        std_rt=0.0,
        coefficient_of_variation=0.0,
        iqr_rt=0.0,
        min_rt=0.0,
        max_rt=0.0,
        skewness=0.0,
        kurtosis=0.0
    )


def _empty_sdt_metrics(hits: int, misses: int, fa: int, cr: int) -> SignalDetectionMetrics:
    """Return SDT metrics with zero discrimination."""
    return SignalDetectionMetrics(
        hits=hits,
        misses=misses,
        false_alarms=fa,
        correct_rejections=cr,
        hit_rate=0.5,
        false_alarm_rate=0.5,
        d_prime=0.0,
        criterion_c=0.0,
        beta=1.0,
        accuracy=0.0,
        balanced_accuracy=0.0
    )
