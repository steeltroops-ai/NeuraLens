"""
Enhanced Speech Analysis Response Models
Comprehensive response schemas with all 9 biomarkers and confidence indicators.

Feature: speech-pipeline-fix
**Validates: Requirements 9.1, 9.2, 9.3, 9.4**
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import uuid


class BiomarkerResult(BaseModel):
    """
    Individual biomarker with metadata for clinical interpretation.
    
    **Validates: Requirements 9.2, 9.4**
    """
    value: float = Field(..., description="The biomarker value")
    unit: str = Field(..., description="Unit of measurement")
    normal_range: Tuple[float, float] = Field(..., description="Normal range (min, max)")
    is_estimated: bool = Field(default=False, description="Whether value is estimated/fallback")
    confidence: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for this biomarker"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "value": 0.02,
                "unit": "ratio",
                "normal_range": [0.0, 0.05],
                "is_estimated": False,
                "confidence": 0.95
            }
        }


class EnhancedBiomarkers(BaseModel):
    """
    All 9 biomarkers with full metadata.
    
    **Validates: Requirements 9.1, 9.2, 9.4**
    """
    jitter: BiomarkerResult = Field(..., description="Fundamental frequency variation (0-1)")
    shimmer: BiomarkerResult = Field(..., description="Amplitude variation (0-1)")
    hnr: BiomarkerResult = Field(..., description="Harmonics-to-Noise Ratio (0-30 dB)")
    speech_rate: BiomarkerResult = Field(..., description="Syllables per second (0.5-10)")
    pause_ratio: BiomarkerResult = Field(..., description="Proportion of silence (0-1)")
    fluency_score: BiomarkerResult = Field(..., description="Speech fluency measure (0-1)")
    voice_tremor: BiomarkerResult = Field(..., description="Tremor intensity (0-1)")
    articulation_clarity: BiomarkerResult = Field(..., description="Clarity of articulation (0-1)")
    prosody_variation: BiomarkerResult = Field(..., description="Prosodic richness (0-1)")

    
    @classmethod
    def from_extracted_biomarkers(
        cls,
        extracted: "ExtractedBiomarkers",
        confidence_scores: Optional[Dict[str, float]] = None
    ) -> "EnhancedBiomarkers":
        """
        Create EnhancedBiomarkers from ExtractedBiomarkers.
        
        Args:
            extracted: ExtractedBiomarkers from BiomarkerExtractor
            confidence_scores: Optional confidence scores per biomarker
            
        Returns:
            EnhancedBiomarkers instance
        """
        # Import here to avoid circular imports
        from app.services.biomarker_extractor import ExtractedBiomarkers, BiomarkerExtractor
        
        confidence_scores = confidence_scores or {}
        
        # Define normal ranges and units for each biomarker
        biomarker_metadata = {
            'jitter': {'unit': 'ratio', 'normal_range': (0.0, 0.05)},
            'shimmer': {'unit': 'ratio', 'normal_range': (0.0, 0.1)},
            'hnr': {'unit': 'dB', 'normal_range': (10.0, 25.0)},
            'speech_rate': {'unit': 'syllables/s', 'normal_range': (3.0, 6.0)},
            'pause_ratio': {'unit': 'ratio', 'normal_range': (0.2, 0.4)},
            'fluency_score': {'unit': 'score', 'normal_range': (0.6, 1.0)},
            'voice_tremor': {'unit': 'score', 'normal_range': (0.0, 0.2)},
            'articulation_clarity': {'unit': 'score', 'normal_range': (0.6, 1.0)},
            'prosody_variation': {'unit': 'score', 'normal_range': (0.3, 0.7)},
        }
        
        def create_result(name: str, value: float, is_estimated: bool) -> BiomarkerResult:
            meta = biomarker_metadata[name]
            return BiomarkerResult(
                value=value,
                unit=meta['unit'],
                normal_range=meta['normal_range'],
                is_estimated=is_estimated,
                confidence=confidence_scores.get(name)
            )
        
        return cls(
            jitter=create_result('jitter', extracted.jitter, extracted.estimated_flags.get('jitter', False)),
            shimmer=create_result('shimmer', extracted.shimmer, extracted.estimated_flags.get('shimmer', False)),
            hnr=create_result('hnr', extracted.hnr, extracted.estimated_flags.get('hnr', False)),
            speech_rate=create_result('speech_rate', extracted.speech_rate, extracted.estimated_flags.get('speech_rate', False)),
            pause_ratio=create_result('pause_ratio', extracted.pause_ratio, extracted.estimated_flags.get('pause_ratio', False)),
            fluency_score=create_result('fluency_score', extracted.fluency_score, extracted.estimated_flags.get('fluency_score', False)),
            voice_tremor=create_result('voice_tremor', extracted.voice_tremor, extracted.estimated_flags.get('voice_tremor', False)),
            articulation_clarity=create_result('articulation_clarity', extracted.articulation_clarity, extracted.estimated_flags.get('articulation_clarity', False)),
            prosody_variation=create_result('prosody_variation', extracted.prosody_variation, extracted.estimated_flags.get('prosody_variation', False)),
        )


class BaselineComparison(BaseModel):
    """
    Comparison to baseline values when available.
    
    **Validates: Requirements 9.3**
    """
    biomarker_name: str = Field(..., description="Name of the biomarker")
    current_value: float = Field(..., description="Current measurement")
    baseline_value: float = Field(..., description="Baseline measurement")
    delta: float = Field(..., description="Change from baseline (current - baseline)")
    delta_percent: float = Field(..., description="Percentage change from baseline")
    direction: str = Field(..., description="Direction of change: improved, worsened, stable")
    
    @classmethod
    def calculate(
        cls,
        biomarker_name: str,
        current_value: float,
        baseline_value: float,
        higher_is_better: bool = True
    ) -> "BaselineComparison":
        """
        Calculate baseline comparison for a biomarker.
        
        Args:
            biomarker_name: Name of the biomarker
            current_value: Current measurement
            baseline_value: Baseline measurement
            higher_is_better: Whether higher values indicate improvement
            
        Returns:
            BaselineComparison instance
        """
        delta = current_value - baseline_value
        delta_percent = (delta / baseline_value * 100) if baseline_value != 0 else 0.0
        
        # Determine direction based on whether higher is better
        threshold = 0.05  # 5% change threshold for "stable"
        if abs(delta_percent) < threshold * 100:
            direction = "stable"
        elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
            direction = "improved"
        else:
            direction = "worsened"
        
        return cls(
            biomarker_name=biomarker_name,
            current_value=current_value,
            baseline_value=baseline_value,
            delta=delta,
            delta_percent=delta_percent,
            direction=direction
        )


class EnhancedSpeechAnalysisResponse(BaseModel):
    """
    Enhanced response with all biomarkers and clinical metadata.
    
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    risk_score: float = Field(ge=0.0, le=1.0, description="Neurological risk score")
    quality_score: float = Field(ge=0.0, le=1.0, description="Audio quality score")
    
    # Enhanced biomarkers with all 9 metrics
    biomarkers: EnhancedBiomarkers = Field(..., description="All extracted biomarkers")
    
    # Legacy biomarkers for backward compatibility
    legacy_biomarkers: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Legacy biomarker format for backward compatibility"
    )
    
    # File and processing info
    file_info: Optional[Dict[str, Any]] = Field(default=None, description="Audio file metadata")
    
    # Clinical recommendations
    recommendations: List[str] = Field(default_factory=list, description="Clinical recommendations")
    
    # Baseline comparison (when available)
    baseline_comparison: Optional[List[BaselineComparison]] = Field(
        default=None,
        description="Comparison to baseline values when available"
    )
    
    # Status
    status: str = Field(default="completed", description="Analysis status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "processing_time": 2.5,
                "timestamp": "2024-01-15T10:30:00Z",
                "confidence": 0.92,
                "risk_score": 0.35,
                "quality_score": 0.88,
                "biomarkers": {
                    "jitter": {
                        "value": 0.02,
                        "unit": "ratio",
                        "normal_range": [0.0, 0.05],
                        "is_estimated": False,
                        "confidence": 0.95
                    },
                    "shimmer": {
                        "value": 0.05,
                        "unit": "ratio",
                        "normal_range": [0.0, 0.1],
                        "is_estimated": False,
                        "confidence": 0.93
                    }
                },
                "recommendations": [
                    "Voice quality within normal range",
                    "Consider follow-up assessment in 6 months"
                ],
                "status": "completed"
            }
        }
    
    @classmethod
    def create_from_analysis(
        cls,
        session_id: str,
        processing_time: float,
        confidence: float,
        risk_score: float,
        quality_score: float,
        biomarkers: EnhancedBiomarkers,
        file_info: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None,
        baseline_values: Optional[Dict[str, float]] = None
    ) -> "EnhancedSpeechAnalysisResponse":
        """
        Create an enhanced response from analysis results.
        
        Args:
            session_id: Session identifier
            processing_time: Processing time in seconds
            confidence: Overall confidence score
            risk_score: Neurological risk score
            quality_score: Audio quality score
            biomarkers: Enhanced biomarkers
            file_info: Optional file metadata
            recommendations: Optional recommendations
            baseline_values: Optional baseline values for comparison
            
        Returns:
            EnhancedSpeechAnalysisResponse instance
        """
        # Calculate baseline comparisons if baseline values provided
        baseline_comparison = None
        if baseline_values:
            # Define which biomarkers have "higher is better" semantics
            higher_is_better = {
                'jitter': False,  # Lower jitter is better
                'shimmer': False,  # Lower shimmer is better
                'hnr': True,  # Higher HNR is better
                'speech_rate': None,  # Depends on context
                'pause_ratio': False,  # Lower pause ratio is generally better
                'fluency_score': True,  # Higher fluency is better
                'voice_tremor': False,  # Lower tremor is better
                'articulation_clarity': True,  # Higher clarity is better
                'prosody_variation': True,  # Higher variation is generally better
            }
            
            baseline_comparison = []
            biomarker_dict = {
                'jitter': biomarkers.jitter.value,
                'shimmer': biomarkers.shimmer.value,
                'hnr': biomarkers.hnr.value,
                'speech_rate': biomarkers.speech_rate.value,
                'pause_ratio': biomarkers.pause_ratio.value,
                'fluency_score': biomarkers.fluency_score.value,
                'voice_tremor': biomarkers.voice_tremor.value,
                'articulation_clarity': biomarkers.articulation_clarity.value,
                'prosody_variation': biomarkers.prosody_variation.value,
            }
            
            for name, current_value in biomarker_dict.items():
                if name in baseline_values:
                    hib = higher_is_better.get(name, True)
                    if hib is not None:
                        comparison = BaselineComparison.calculate(
                            biomarker_name=name,
                            current_value=current_value,
                            baseline_value=baseline_values[name],
                            higher_is_better=hib
                        )
                        baseline_comparison.append(comparison)
        
        # Create legacy biomarkers for backward compatibility
        legacy_biomarkers = {
            'fluency_score': biomarkers.fluency_score.value,
            'pause_pattern': biomarkers.pause_ratio.value,
            'voice_tremor': biomarkers.voice_tremor.value,
            'articulation_clarity': biomarkers.articulation_clarity.value,
            'prosody_variation': biomarkers.prosody_variation.value,
            'speaking_rate': biomarkers.speech_rate.value * 60,  # Convert to words per minute approx
            'pause_frequency': biomarkers.pause_ratio.value * 10,  # Approximate pauses per minute
        }
        
        return cls(
            session_id=session_id,
            processing_time=processing_time,
            timestamp=datetime.utcnow(),
            confidence=confidence,
            risk_score=risk_score,
            quality_score=quality_score,
            biomarkers=biomarkers,
            legacy_biomarkers=legacy_biomarkers,
            file_info=file_info,
            recommendations=recommendations or [],
            baseline_comparison=baseline_comparison,
            status="completed"
        )
