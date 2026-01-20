"""
Dermatology Pipeline Classifier

Melanoma, malignancy, and subtype classification.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from ..config import CLASSIFICATION_CONFIG, LESION_SUBTYPES, MALIGNANT_SUBTYPES
from ..schemas import (
    MelanomaResult,
    MalignancyResult,
    SubtypeResult,
    SubtypePrediction,
    ABCDEFeatures,
    MelanomaClass
)

logger = logging.getLogger(__name__)


class MelanomaClassifier:
    """
    Estimates melanoma probability.
    Uses ABCDE features as proxy when DL model unavailable.
    """
    
    def __init__(self, config=None):
        self.config = config or CLASSIFICATION_CONFIG
    
    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        abcde: ABCDEFeatures
    ) -> MelanomaResult:
        """Classify melanoma probability."""
        # ABCDE-based scoring
        abcde_prob = self._abcde_score(abcde)
        
        # DL model prediction (placeholder - returns ABCDE for now)
        dl_prob = self._dl_predict(image, mask)
        
        # Combined probability
        # When no DL model: use ABCDE only
        if dl_prob is None:
            combined = abcde_prob
            dl_prob = abcde_prob
        else:
            combined = (
                self.config.ensemble_weights.get('efficientnet', 0.7) * dl_prob +
                0.3 * abcde_prob
            )
        
        # Uncertainty
        uncertainty = self._estimate_uncertainty(dl_prob, abcde_prob)
        
        # Classification
        classification = self._classify(combined)
        
        # Concerning features
        concerning = self._get_concerning_features(abcde)
        
        # Recommendation
        recommendation = self._get_recommendation(combined, uncertainty)
        
        return MelanomaResult(
            probability=combined,
            dl_probability=dl_prob,
            abcde_probability=abcde_prob,
            uncertainty=uncertainty,
            classification=classification,
            concerning_features=concerning,
            recommendation=recommendation
        )
    
    def _abcde_score(self, abcde: ABCDEFeatures) -> float:
        """Compute melanoma risk from ABCDE."""
        score = 0.0
        
        # Weighted contributions
        if abcde.asymmetry.is_concerning:
            score += 0.25 * abcde.asymmetry.combined_score
        
        if abcde.border.is_concerning:
            score += 0.20 * abcde.border.irregularity_score
        
        if abcde.color.is_concerning:
            score += 0.25 * abcde.color.color_score
            # Extra weight for blue-white veil
            if abcde.color.has_blue_white_veil:
                score += 0.15
        
        if abcde.diameter.is_concerning:
            score += 0.15 * abcde.diameter.risk_contribution
        
        if abcde.evolution.is_concerning:
            score += 0.15 * abcde.evolution.evolution_score
        
        return min(score, 1.0)
    
    def _dl_predict(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Optional[float]:
        """Deep learning prediction (placeholder)."""
        # In production, this would load and run the model
        # For now, return None to indicate no model available
        return None
    
    def _estimate_uncertainty(self, dl_prob: float, abcde_prob: float) -> float:
        """Estimate prediction uncertainty."""
        # Disagreement between methods
        disagreement = abs(dl_prob - abcde_prob)
        
        # Boundary uncertainty (near 0.5)
        avg_prob = (dl_prob + abcde_prob) / 2
        boundary_dist = abs(0.5 - avg_prob)
        boundary_uncertainty = 1 - 2 * boundary_dist
        
        uncertainty = disagreement * 0.6 + boundary_uncertainty * 0.4
        return min(uncertainty, 1.0)
    
    def _classify(self, probability: float) -> MelanomaClass:
        """Convert probability to classification."""
        if probability >= self.config.melanoma_critical_threshold:
            return MelanomaClass.HIGH_SUSPICION
        elif probability >= self.config.melanoma_high_threshold:
            return MelanomaClass.MODERATE_SUSPICION
        elif probability >= self.config.melanoma_moderate_threshold:
            return MelanomaClass.LOW_SUSPICION
        else:
            return MelanomaClass.UNLIKELY
    
    def _get_concerning_features(self, abcde: ABCDEFeatures) -> List[str]:
        """List concerning features."""
        concerning = []
        
        if abcde.asymmetry.is_concerning:
            concerning.append(f"Asymmetry ({abcde.asymmetry.classification})")
        
        if abcde.border.is_concerning:
            concerning.append(f"Border irregularity ({abcde.border.classification})")
        
        if abcde.color.is_concerning:
            color_desc = f"{abcde.color.num_colors} colors"
            if abcde.color.has_blue_white_veil:
                color_desc += ", blue-white veil"
            concerning.append(f"Color variation ({color_desc})")
        
        if abcde.diameter.is_concerning:
            concerning.append(f"Diameter {abcde.diameter.max_dimension_mm:.1f}mm > 6mm")
        
        if abcde.evolution.is_concerning:
            concerning.append(f"Evolution indicators ({abcde.evolution.classification})")
        
        return concerning
    
    def _get_recommendation(self, probability: float, uncertainty: float) -> str:
        """Generate recommendation based on results."""
        if probability >= self.config.melanoma_critical_threshold:
            return "Immediate dermatology referral strongly recommended"
        elif probability >= self.config.melanoma_high_threshold:
            return "Prompt dermatology evaluation recommended within 1-2 weeks"
        elif probability >= self.config.melanoma_moderate_threshold:
            return "Schedule dermatology consultation within 1-3 months"
        elif uncertainty > 0.4:
            return "Moderate uncertainty - consider professional evaluation"
        else:
            return "Low risk - routine monitoring recommended"


class MalignancyClassifier:
    """
    Binary benign vs malignant classification.
    """
    
    def __init__(self, config=None):
        self.config = config or CLASSIFICATION_CONFIG
    
    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        abcde: ABCDEFeatures
    ) -> MalignancyResult:
        """Classify as benign or malignant."""
        # Feature-based estimation
        malignant_prob = self._estimate_malignancy(abcde)
        benign_prob = 1 - malignant_prob
        
        # Adjust based on criteria count
        if abcde.criteria_met >= 4:
            malignant_prob = min(malignant_prob * 1.3, 0.95)
            benign_prob = 1 - malignant_prob
        
        classification = "malignant" if malignant_prob > 0.5 else "benign"
        confidence = abs(malignant_prob - 0.5) * 2
        
        return MalignancyResult(
            classification=classification,
            benign_probability=benign_prob,
            malignant_probability=malignant_prob,
            confidence=confidence,
            needs_biopsy=malignant_prob > 0.3
        )
    
    def _estimate_malignancy(self, abcde: ABCDEFeatures) -> float:
        """Estimate malignancy probability from features."""
        # Base on ABCDE total score
        base_prob = abcde.total_score
        
        # Adjust for specific high-risk indicators
        if abcde.color.has_blue_white_veil:
            base_prob = min(base_prob + 0.2, 1.0)
        
        if abcde.criteria_met >= 3:
            base_prob = min(base_prob + 0.1, 1.0)
        
        return base_prob


class SubtypeClassifier:
    """
    Multi-class lesion subtype classification.
    """
    
    def __init__(self, config=None):
        self.config = config or CLASSIFICATION_CONFIG
        self.subtypes = LESION_SUBTYPES
        self.malignant = MALIGNANT_SUBTYPES
    
    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        abcde: ABCDEFeatures,
        melanoma_prob: float
    ) -> SubtypeResult:
        """Classify lesion subtype."""
        # Estimate probabilities for each subtype
        probs = self._estimate_probabilities(abcde, melanoma_prob)
        
        # Sort by probability
        sorted_preds = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # Create predictions list
        predictions = []
        for subtype, prob in sorted_preds[:5]:
            predictions.append(SubtypePrediction(
                subtype=subtype,
                probability=prob,
                is_malignant=subtype in self.malignant
            ))
        
        primary = predictions[0]
        
        return SubtypeResult(
            primary_subtype=primary.subtype,
            primary_probability=primary.probability,
            is_malignant=primary.is_malignant,
            all_predictions=predictions,
            confidence=primary.probability
        )
    
    def _estimate_probabilities(
        self,
        abcde: ABCDEFeatures,
        melanoma_prob: float
    ) -> Dict[str, float]:
        """Estimate subtype probabilities."""
        probs = {}
        
        # Melanoma probability from classifier
        probs['melanoma'] = melanoma_prob
        
        # Nevus: low risk features
        nevus_score = max(0, 1 - abcde.total_score - melanoma_prob)
        probs['nevus'] = nevus_score * 0.7
        
        # BCC: specific patterns (less asymmetry, more color variation)
        bcc_score = (
            (1 - abcde.asymmetry.combined_score) * 0.3 +
            abcde.color.color_score * 0.3 +
            (1 - abcde.diameter.risk_contribution) * 0.2 +
            0.2
        ) * (1 - melanoma_prob) * 0.3
        probs['basal_cell_carcinoma'] = bcc_score
        
        # SCC: irregular but different pattern
        scc_score = (
            abcde.border.irregularity_score * 0.3 +
            (1 - abcde.color.homogeneity) * 0.3 +
            0.2
        ) * (1 - melanoma_prob) * 0.2
        probs['squamous_cell_carcinoma'] = scc_score
        
        # Benign keratosis
        probs['benign_keratosis'] = max(0, (1 - abcde.total_score) * 0.2)
        
        # Actinic keratosis
        probs['actinic_keratosis'] = max(0, (1 - melanoma_prob) * 0.1)
        
        # Dermatofibroma
        probs['dermatofibroma'] = max(0, (1 - abcde.total_score) * 0.1)
        
        # Vascular lesion
        if 'red' in abcde.color.concerning_colors:
            probs['vascular_lesion'] = 0.1
        else:
            probs['vascular_lesion'] = 0.02
        
        # Normalize to sum to 1
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs


class DermatologyClassifier:
    """
    Combined classifier for dermatology pipeline.
    """
    
    def __init__(self, config=None):
        self.config = config or CLASSIFICATION_CONFIG
        self.melanoma_classifier = MelanomaClassifier(config)
        self.malignancy_classifier = MalignancyClassifier(config)
        self.subtype_classifier = SubtypeClassifier(config)
    
    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        abcde: ABCDEFeatures
    ) -> Dict[str, Any]:
        """Run all classifiers."""
        # Melanoma classification
        melanoma = self.melanoma_classifier.classify(image, mask, abcde)
        
        # Malignancy classification
        malignancy = self.malignancy_classifier.classify(image, mask, abcde)
        
        # Subtype classification
        subtype = self.subtype_classifier.classify(
            image, mask, abcde, melanoma.probability
        )
        
        return {
            'melanoma': melanoma,
            'malignancy': malignancy,
            'subtype': subtype
        }
