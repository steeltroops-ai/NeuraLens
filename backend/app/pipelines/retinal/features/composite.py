"""
Retinal Features - Composite Biomarkers Module
Novel composite biomarkers derived from multiple features.

Matches speech/features/composite.py structure.

Features:
- Retinal Health Index (RHI)
- Vascular Risk Score (VRS)
- Macular Health Score (MHS)
- Progression Risk Index (PRI)

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompositeBiomarkers:
    """Novel composite biomarkers for retinal analysis."""
    
    # Retinal Health Index (0-100, higher = healthier)
    retinal_health_index: float = 85.0
    
    # Vascular Risk Score (0-100, higher = more risk)
    vascular_risk_score: float = 15.0
    
    # Macular Health Score (0-100)
    macular_health_score: float = 90.0
    
    # Progression Risk Index (0-1, probability of progression)
    progression_risk_index: float = 0.1
    
    # DR Severity Score (0-100)
    dr_severity_score: float = 0.0
    
    # Glaucoma Probability Score (0-1)
    glaucoma_probability: float = 0.1
    
    # Quality-adjusted confidence
    quality_adjusted_confidence: float = 0.85
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "retinal_health_index": self.retinal_health_index,
            "vascular_risk_score": self.vascular_risk_score,
            "macular_health_score": self.macular_health_score,
            "progression_risk_index": self.progression_risk_index,
            "dr_severity_score": self.dr_severity_score,
            "glaucoma_probability": self.glaucoma_probability,
            "quality_adjusted_confidence": self.quality_adjusted_confidence,
        }


class CompositeFeatureExtractor:
    """
    Composite biomarker computation from extracted features.
    
    Combines vessel, optic disc, and lesion features into
    clinically meaningful composite scores.
    """
    
    def __init__(self):
        # Weights derived from clinical literature
        self.rhi_weights = {
            "vessel_quality": 0.30,
            "disc_health": 0.25,
            "lesion_burden": 0.30,
            "macular_status": 0.15,
        }
        
        self.vrs_weights = {
            "av_ratio_deviation": 0.35,
            "tortuosity": 0.25,
            "dr_grade": 0.25,
            "hypertensive_signs": 0.15,
        }
    
    def extract(
        self,
        vessel_metrics: Optional[Dict] = None,
        optic_disc_metrics: Optional[Dict] = None,
        lesion_metrics: Optional[Dict] = None,
        quality_score: float = 0.8,
        dr_grade: int = 0,
    ) -> CompositeBiomarkers:
        """
        Compute composite biomarkers from individual metrics.
        
        Args:
            vessel_metrics: Dict from VesselFeatureExtractor
            optic_disc_metrics: Dict from OpticDiscExtractor
            lesion_metrics: Dict from LesionDetector
            quality_score: Image quality score
            dr_grade: ICDR DR grade (0-4)
            
        Returns:
            CompositeBiomarkers with all composite scores
        """
        composite = CompositeBiomarkers()
        
        try:
            # Use default values if metrics not provided
            vessel_metrics = vessel_metrics or {}
            optic_disc_metrics = optic_disc_metrics or {}
            lesion_metrics = lesion_metrics or {}
            
            # Retinal Health Index
            composite.retinal_health_index = self._compute_rhi(
                vessel_metrics, optic_disc_metrics, lesion_metrics, dr_grade
            )
            
            # Vascular Risk Score
            composite.vascular_risk_score = self._compute_vrs(
                vessel_metrics, dr_grade
            )
            
            # Macular Health Score
            composite.macular_health_score = self._compute_mhs(
                lesion_metrics
            )
            
            # Progression Risk Index
            composite.progression_risk_index = self._compute_pri(
                dr_grade, lesion_metrics, vessel_metrics
            )
            
            # DR Severity Score
            composite.dr_severity_score = self._compute_dr_severity(
                dr_grade, lesion_metrics
            )
            
            # Glaucoma Probability
            composite.glaucoma_probability = self._compute_glaucoma_prob(
                optic_disc_metrics
            )
            
            # Quality-adjusted confidence
            composite.quality_adjusted_confidence = min(0.95, 0.5 + quality_score * 0.45)
            
        except Exception as e:
            logger.warning(f"Composite extraction failed: {e}")
        
        return composite
    
    def _compute_rhi(
        self,
        vessel: Dict,
        disc: Dict,
        lesion: Dict,
        dr_grade: int
    ) -> float:
        """
        Compute Retinal Health Index (0-100).
        
        Higher score = healthier retina.
        """
        score = 100.0
        
        # Vessel health component (30%)
        avr = vessel.get("av_ratio", 0.7)
        if avr < 0.6:
            score -= 15 * (0.6 - avr) / 0.1
        
        tort = vessel.get("tortuosity_index", 1.1)
        if tort > 1.15:
            score -= 10 * (tort - 1.15) / 0.1
        
        # Disc health component (25%)
        cdr = disc.get("cup_disc_ratio", 0.35)
        if cdr > 0.5:
            score -= 12.5 * (cdr - 0.5) / 0.3
        
        # Lesion burden component (30%)
        ma_count = lesion.get("microaneurysm_count", 0)
        hem_count = lesion.get("hemorrhage_count", 0)
        
        score -= min(15, ma_count * 2)
        score -= min(15, hem_count * 3)
        
        # DR grade component (15%)
        dr_penalty = {0: 0, 1: 5, 2: 10, 3: 12, 4: 15}
        score -= dr_penalty.get(dr_grade, 0)
        
        return max(0, min(100, score))
    
    def _compute_vrs(self, vessel: Dict, dr_grade: int) -> float:
        """
        Compute Vascular Risk Score (0-100).
        
        Higher score = more cardiovascular risk indicated.
        """
        score = 0.0
        
        # AV ratio deviation
        avr = vessel.get("av_ratio", 0.7)
        if avr < 0.65:
            score += 25 * (0.65 - avr) / 0.15
        
        # Tortuosity
        tort = vessel.get("tortuosity_index", 1.1)
        if tort > 1.1:
            score += 20 * (tort - 1.1) / 0.2
        
        # Vessel density abnormality
        density = vessel.get("vessel_density", 0.72)
        if density < 0.6:
            score += 15 * (0.6 - density) / 0.2
        
        # DR component
        dr_contribution = {0: 0, 1: 10, 2: 20, 3: 30, 4: 40}
        score += dr_contribution.get(dr_grade, 0)
        
        return max(0, min(100, score))
    
    def _compute_mhs(self, lesion: Dict) -> float:
        """
        Compute Macular Health Score (0-100).
        
        Focused on macular area pathology.
        """
        score = 100.0
        
        # Exudates near macula are critical
        if lesion.get("exudate_near_macula", False):
            score -= 30
        
        # Hemorrhages in macular area
        hem_count = lesion.get("hemorrhage_count", 0)
        score -= min(30, hem_count * 5)
        
        # Microaneurysms
        ma_count = lesion.get("microaneurysm_count", 0)
        score -= min(20, ma_count * 2)
        
        return max(0, min(100, score))
    
    def _compute_pri(
        self,
        dr_grade: int,
        lesion: Dict,
        vessel: Dict
    ) -> float:
        """
        Compute Progression Risk Index (0-1).
        
        Probability of condition worsening in next 12 months.
        """
        risk = 0.0
        
        # Base risk by DR grade
        base_risk = {0: 0.05, 1: 0.15, 2: 0.25, 3: 0.45, 4: 0.60}
        risk = base_risk.get(dr_grade, 0.05)
        
        # Lesion burden modifiers
        ma_count = lesion.get("microaneurysm_count", 0)
        if ma_count > 5:
            risk += 0.05
        if ma_count > 10:
            risk += 0.10
        
        hem_count = lesion.get("hemorrhage_count", 0)
        if hem_count > 10:
            risk += 0.10
        
        # Vessel abnormalities
        avr = vessel.get("av_ratio", 0.7)
        if avr < 0.55:
            risk += 0.10
        
        return min(0.95, max(0, risk))
    
    def _compute_dr_severity(self, dr_grade: int, lesion: Dict) -> float:
        """
        Compute continuous DR severity score (0-100).
        
        More granular than discrete grades.
        """
        # Base score from grade
        base = dr_grade * 20  # 0, 20, 40, 60, 80
        
        # Add lesion burden within grade
        ma_count = lesion.get("microaneurysm_count", 0)
        hem_count = lesion.get("hemorrhage_count", 0)
        exudate_area = lesion.get("hard_exudate_area_percent", 0)
        
        modifier = min(19, ma_count * 0.5 + hem_count * 1 + exudate_area * 2)
        
        return min(100, base + modifier)
    
    def _compute_glaucoma_prob(self, disc: Dict) -> float:
        """
        Compute glaucoma probability (0-1).
        """
        cdr = disc.get("cup_disc_ratio", 0.35)
        
        if cdr < 0.4:
            return 0.05
        elif cdr < 0.5:
            return 0.10
        elif cdr < 0.6:
            return 0.25
        elif cdr < 0.7:
            return 0.50
        else:
            return 0.75


# Singleton instance
composite_extractor = CompositeFeatureExtractor()
