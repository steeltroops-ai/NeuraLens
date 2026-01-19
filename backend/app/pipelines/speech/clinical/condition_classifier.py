"""
Condition Classifier
ML-based condition classification with probabilistic output.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result from condition classifier."""
    condition: str
    probability: float
    confidence: float
    uncertainty: float              # Epistemic + aleatoric
    calibrated: bool = True


class ConditionClassifier:
    """
    Probabilistic condition classifier.
    
    Uses either:
    - Rule-based scoring (current implementation)
    - ML model with uncertainty (when trained model available)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.use_ml = False
        
        # Load ML model if available
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load trained ML model."""
        # Placeholder for ML model loading
        # Would load PyTorch/ONNX model here
        pass
    
    def classify(
        self,
        features: Dict[str, float],
        embeddings: Optional[np.ndarray] = None
    ) -> List[ClassificationResult]:
        """
        Classify conditions from features and embeddings.
        
        Args:
            features: Dict of extracted features
            embeddings: Optional learned embeddings
            
        Returns:
            List of classification results
        """
        if self.use_ml and self.model is not None:
            return self._classify_ml(features, embeddings)
        else:
            return self._classify_rules(features)
    
    def _classify_ml(
        self,
        features: Dict,
        embeddings: Optional[np.ndarray]
    ) -> List[ClassificationResult]:
        """ML-based classification with Monte Carlo dropout for uncertainty."""
        # Placeholder for ML inference
        # Would run model with MC dropout here
        return self._classify_rules(features)
    
    def _classify_rules(self, features: Dict) -> List[ClassificationResult]:
        """Rule-based classification (fallback)."""
        results = []
        
        # Parkinson's
        pd_prob = self._score_parkinsons(features)
        results.append(ClassificationResult(
            condition="parkinsons",
            probability=pd_prob,
            confidence=0.75,
            uncertainty=0.15 if pd_prob > 0.1 else 0.05
        ))
        
        # Alzheimer's
        ad_prob = self._score_alzheimers(features)
        results.append(ClassificationResult(
            condition="alzheimers",
            probability=ad_prob,
            confidence=0.70,
            uncertainty=0.18 if ad_prob > 0.1 else 0.05
        ))
        
        # Depression
        dep_prob = self._score_depression(features)
        results.append(ClassificationResult(
            condition="depression",
            probability=dep_prob,
            confidence=0.65,
            uncertainty=0.12
        ))
        
        # Dysarthria
        dys_prob = self._score_dysarthria(features)
        results.append(ClassificationResult(
            condition="dysarthria",
            probability=dys_prob,
            confidence=0.80,
            uncertainty=0.10 if dys_prob > 0.1 else 0.03
        ))
        
        return results
    
    def _score_parkinsons(self, f: Dict) -> float:
        """Score Parkinson's risk."""
        score = 0.0
        
        tremor = f.get("tremor_score", 0)
        if tremor > 0.15:
            score += 0.35
        elif tremor > 0.05:
            score += 0.15
        
        jitter = f.get("jitter_local", 0)
        if jitter > 1.5:
            score += 0.25
        elif jitter > 1.0:
            score += 0.10
        
        speech_rate = f.get("speech_rate", 4.5)
        if speech_rate < 3.0:
            score += 0.20
        elif speech_rate < 3.5:
            score += 0.10
        
        nii = f.get("nii", 0)
        if nii > 0.3:
            score += 0.20
        elif nii > 0.2:
            score += 0.10
        
        return min(0.95, score)
    
    def _score_alzheimers(self, f: Dict) -> float:
        """Score Alzheimer's/cognitive decline risk."""
        score = 0.0
        
        pause_ratio = f.get("pause_ratio", 0.15)
        if pause_ratio > 0.35:
            score += 0.35
        elif pause_ratio > 0.25:
            score += 0.15
        
        speech_rate = f.get("speech_rate", 4.5)
        if speech_rate < 2.5:
            score += 0.25
        elif speech_rate < 3.0:
            score += 0.10
        
        f0_cv = f.get("f0_cv", 0.15)
        if f0_cv < 0.08:
            score += 0.20
        elif f0_cv < 0.10:
            score += 0.10
        
        return min(0.90, score * 0.8)  # Scale down
    
    def _score_depression(self, f: Dict) -> float:
        """Score depression risk."""
        score = 0.0
        
        f0_cv = f.get("f0_cv", 0.15)
        if f0_cv < 0.08:
            score += 0.40
        elif f0_cv < 0.12:
            score += 0.20
        
        speech_rate = f.get("speech_rate", 4.5)
        if speech_rate < 3.5:
            score += 0.30
        
        intensity_std = f.get("intensity_std", 10)
        if intensity_std < 5:
            score += 0.30
        
        return min(0.85, score * 0.6)
    
    def _score_dysarthria(self, f: Dict) -> float:
        """Score dysarthria/motor speech disorder risk."""
        score = 0.0
        
        fcr = f.get("fcr", 1.0)
        if fcr > 1.2:
            score += 0.35
        elif fcr > 1.1:
            score += 0.15
        
        hnr = f.get("hnr", 25)
        if hnr < 12:
            score += 0.30
        elif hnr < 18:
            score += 0.15
        
        shimmer = f.get("shimmer_local", 2)
        if shimmer > 5:
            score += 0.25
        elif shimmer > 3.8:
            score += 0.10
        
        cpps = f.get("cpps", 18)
        if cpps < 10:
            score += 0.10
        
        return min(0.95, score)
