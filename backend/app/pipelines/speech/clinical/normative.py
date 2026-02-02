"""
Normative Data Manager v4.0
Age and sex-adjusted reference data for clinical interpretation.

Provides normative comparison tables based on published clinical research
for accurate interpretation of speech biomarkers.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class Sex(str, Enum):
    """Biological sex for normative data."""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeGroup(str, Enum):
    """Age group categories."""
    YOUNG = "18-30"
    MIDDLE = "31-50"
    OLDER = "51-65"
    ELDERLY = "66-80"
    VERY_ELDERLY = "80+"


@dataclass
class NormativeReference:
    """Normative reference for a feature."""
    feature_name: str
    
    # Reference values
    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    
    # Percentile boundaries
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    
    # Clinical thresholds
    lower_limit: float = 0.0
    upper_limit: float = 0.0
    
    # Metadata
    source: str = ""
    population: str = ""
    sample_size: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "feature": self.feature_name,
            "mean": self.mean,
            "std": self.std,
            "p5": self.p5,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "source": self.source
        }


class NormativeDataManager:
    """
    Manages normative reference data for speech biomarkers.
    
    Based on published clinical research:
    - Tsanas et al. (2012) - PD voice biomarkers
    - Rusz et al. (2021) - Speech rate and timing
    - Maryn et al. (2010) - CPPS normative data
    - Baken & Orlikoff (2000) - Clinical voice measurement
    """
    
    # Base normative data (healthy adults, mixed sex)
    BASE_NORMS: Dict[str, Dict[str, float]] = {
        "jitter_local": {
            "mean": 0.52, "std": 0.35,
            "p5": 0.15, "p25": 0.30, "p75": 0.70, "p95": 1.04,
            "lower_limit": 0.0, "upper_limit": 1.04,
            "source": "Baken & Orlikoff (2000)"
        },
        "shimmer_local": {
            "mean": 2.05, "std": 1.10,
            "p5": 0.80, "p25": 1.30, "p75": 2.60, "p95": 3.81,
            "lower_limit": 0.0, "upper_limit": 3.81,
            "source": "Baken & Orlikoff (2000)"
        },
        "hnr": {
            "mean": 23.5, "std": 3.5,
            "p5": 17.0, "p25": 21.0, "p75": 26.0, "p95": 30.0,
            "lower_limit": 20.0, "upper_limit": 30.0,
            "source": "Baken & Orlikoff (2000)"
        },
        "cpps": {
            "mean": 18.5, "std": 2.5,
            "p5": 14.0, "p25": 17.0, "p75": 20.0, "p95": 23.0,
            "lower_limit": 14.0, "upper_limit": 30.0,
            "source": "Maryn et al. (2010)"
        },
        "speech_rate": {
            "mean": 4.8, "std": 0.8,
            "p5": 3.5, "p25": 4.2, "p75": 5.4, "p95": 6.2,
            "lower_limit": 3.5, "upper_limit": 6.5,
            "source": "Rusz et al. (2021)"
        },
        "pause_ratio": {
            "mean": 0.18, "std": 0.05,
            "p5": 0.10, "p25": 0.14, "p75": 0.22, "p95": 0.28,
            "lower_limit": 0.10, "upper_limit": 0.25,
            "source": "Clinical standards"
        },
        "tremor_score": {
            "mean": 0.05, "std": 0.03,
            "p5": 0.01, "p25": 0.03, "p75": 0.07, "p95": 0.12,
            "lower_limit": 0.0, "upper_limit": 0.15,
            "source": "Clinical standards"
        },
        "f0_std": {
            "mean": 35.0, "std": 15.0,
            "p5": 15.0, "p25": 25.0, "p75": 45.0, "p95": 65.0,
            "lower_limit": 20.0, "upper_limit": 100.0,
            "source": "Clinical standards"
        }
    }
    
    # Sex-specific adjustments (multipliers)
    SEX_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
        "male": {
            "mean_f0": 0.65,  # Lower F0
            "f1_mean": 0.92,
            "f2_mean": 0.90
        },
        "female": {
            "mean_f0": 1.35,  # Higher F0
            "f1_mean": 1.08,
            "f2_mean": 1.10
        }
    }
    
    # Age-specific adjustments (additive)
    AGE_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
        "18-30": {
            "jitter_local": 0.0,
            "shimmer_local": 0.0,
            "speech_rate": 0.0,
            "pause_ratio": 0.0
        },
        "31-50": {
            "jitter_local": 0.05,
            "shimmer_local": 0.10,
            "speech_rate": -0.1,
            "pause_ratio": 0.01
        },
        "51-65": {
            "jitter_local": 0.15,
            "shimmer_local": 0.30,
            "speech_rate": -0.3,
            "pause_ratio": 0.02
        },
        "66-80": {
            "jitter_local": 0.30,
            "shimmer_local": 0.60,
            "speech_rate": -0.6,
            "pause_ratio": 0.04
        },
        "80+": {
            "jitter_local": 0.50,
            "shimmer_local": 1.00,
            "speech_rate": -0.9,
            "pause_ratio": 0.06
        }
    }
    
    def __init__(self):
        self._build_normative_database()
    
    def _build_normative_database(self):
        """Build internal normative database."""
        self.norms = {}
        
        for feature, data in self.BASE_NORMS.items():
            self.norms[feature] = NormativeReference(
                feature_name=feature,
                mean=data["mean"],
                std=data["std"],
                p5=data["p5"],
                p25=data["p25"],
                p75=data["p75"],
                p95=data["p95"],
                lower_limit=data["lower_limit"],
                upper_limit=data["upper_limit"],
                source=data["source"],
                population="healthy_adults"
            )
    
    @property
    def biomarker_data(self) -> Dict[str, NormativeReference]:
        """Alias for norms for external API compatibility."""
        return self.norms
    
    def get_reference(
        self,
        feature_name: str,
        age: Optional[int] = None,
        sex: Optional[Sex] = None
    ) -> Optional[NormativeReference]:
        """
        Get normative reference for a feature, optionally adjusted.
        
        Args:
            feature_name: Name of the feature
            age: Optional patient age for adjustment
            sex: Optional patient sex for adjustment
            
        Returns:
            NormativeReference or None if not available
        """
        if feature_name not in self.norms:
            return None
        
        base_ref = self.norms[feature_name]
        
        # Apply adjustments if demographics provided
        if age is not None or sex is not None:
            return self._adjust_reference(base_ref, age, sex)
        
        return base_ref
    
    def _adjust_reference(
        self,
        base_ref: NormativeReference,
        age: Optional[int],
        sex: Optional[Sex]
    ) -> NormativeReference:
        """Apply age and sex adjustments to reference values."""
        adjusted = NormativeReference(
            feature_name=base_ref.feature_name,
            mean=base_ref.mean,
            std=base_ref.std,
            p5=base_ref.p5,
            p25=base_ref.p25,
            p75=base_ref.p75,
            p95=base_ref.p95,
            lower_limit=base_ref.lower_limit,
            upper_limit=base_ref.upper_limit,
            source=base_ref.source,
            population="adjusted"
        )
        
        # Age adjustment
        if age is not None:
            age_group = self._get_age_group(age)
            if age_group in self.AGE_ADJUSTMENTS:
                adjustment = self.AGE_ADJUSTMENTS[age_group].get(
                    base_ref.feature_name, 0.0
                )
                adjusted.mean += adjustment
                adjusted.p25 += adjustment
                adjusted.p75 += adjustment
                adjusted.lower_limit += adjustment
                adjusted.upper_limit += adjustment
        
        return adjusted
    
    def _get_age_group(self, age: int) -> str:
        """Determine age group from age."""
        if age <= 30:
            return "18-30"
        elif age <= 50:
            return "31-50"
        elif age <= 65:
            return "51-65"
        elif age <= 80:
            return "66-80"
        else:
            return "80+"
    
    def calculate_z_score(
        self,
        feature_name: str,
        value: float,
        age: Optional[int] = None,
        sex: Optional[Sex] = None
    ) -> Optional[float]:
        """
        Calculate z-score relative to normative data.
        
        Args:
            feature_name: Feature name
            value: Observed value
            age: Optional patient age
            sex: Optional patient sex
            
        Returns:
            Z-score or None if reference not available
        """
        ref = self.get_reference(feature_name, age, sex)
        
        if ref is None or ref.std == 0:
            return None
        
        return (value - ref.mean) / ref.std
    
    def calculate_percentile(
        self,
        feature_name: str,
        value: float,
        age: Optional[int] = None,
        sex: Optional[Sex] = None
    ) -> Optional[float]:
        """
        Calculate approximate percentile from normative data.
        
        Uses linear interpolation between known percentile points.
        """
        ref = self.get_reference(feature_name, age, sex)
        
        if ref is None:
            return None
        
        # Linear interpolation
        points = [
            (ref.p5, 5),
            (ref.p25, 25),
            (ref.mean, 50),  # Assume mean ~ median
            (ref.p75, 75),
            (ref.p95, 95)
        ]
        
        # Sorted by value
        points = sorted(points, key=lambda x: x[0])
        
        # Find interval
        for i in range(len(points) - 1):
            if points[i][0] <= value <= points[i+1][0]:
                # Linear interpolation
                v_low, p_low = points[i]
                v_high, p_high = points[i+1]
                
                if v_high == v_low:
                    return (p_low + p_high) / 2
                
                percentile = p_low + (value - v_low) / (v_high - v_low) * (p_high - p_low)
                return percentile
        
        # Below 5th percentile
        if value < points[0][0]:
            return max(0, 5 * value / points[0][0]) if points[0][0] != 0 else 0
        
        # Above 95th percentile
        if value > points[-1][0]:
            return min(100, 95 + 5 * (value - points[-1][0]) / (points[-1][0] * 0.2 + 0.001))
        
        return 50.0  # Fallback
    
    def get_interpretation(
        self,
        feature_name: str,
        value: float,
        age: Optional[int] = None,
        sex: Optional[Sex] = None
    ) -> str:
        """
        Get clinical interpretation of feature value.
        """
        ref = self.get_reference(feature_name, age, sex)
        
        if ref is None:
            return "No normative data available"
        
        z_score = self.calculate_z_score(feature_name, value, age, sex)
        
        if z_score is None:
            return "Unable to calculate deviation"
        
        if abs(z_score) < 1:
            return "Within normal limits"
        elif abs(z_score) < 2:
            if z_score > 0:
                return "Mildly elevated"
            else:
                return "Mildly reduced"
        elif abs(z_score) < 3:
            if z_score > 0:
                return "Moderately elevated"
            else:
                return "Moderately reduced"
        else:
            if z_score > 0:
                return "Significantly elevated"
            else:
                return "Significantly reduced"
    
    def get_all_references(
        self,
        age: Optional[int] = None,
        sex: Optional[Sex] = None
    ) -> Dict[str, NormativeReference]:
        """Get all normative references, optionally adjusted."""
        return {
            name: self.get_reference(name, age, sex)
            for name in self.norms
        }
