"""
Research-Grade Speech Analysis Service v4.0
Comprehensive pipeline orchestrating all speech analysis components.

Major upgrades from v3.0:
- Enhanced quality gate system
- Unified feature extraction with parallel processing
- Clinical risk assessment with uncertainty quantification
- Real-time streaming support
- SHAP-style feature explanations
- Age/sex-adjusted normative comparisons

Requirements: 4.1-4.10 (Service Layer)
"""

import asyncio
import logging
import time
import numpy as np
import io
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass

# Third-party
import librosa
from pydub import AudioSegment
import parselmouth
from parselmouth.praat import call

# Core infrastructure
from ..config import ResearchPipelineConfig

# Quality gate system
from ..quality import EnhancedQualityGate, QualityReport

# Feature extraction
from ..features import UnifiedFeatureExtractor, UnifiedFeatures

# Clinical analysis
from ..clinical import (
    ClinicalRiskScorer, 
    RiskAssessmentResult,
    UncertaintyEstimator,
    UncertaintyResult,
    RiskExplainer,
    RiskExplanation,
    NormativeDataManager
)

# Streaming support
from ..streaming import (
    StreamingSessionManager,
    StreamingSession,
    StreamingAnalyzer,
    StreamingResult
)

# Monitoring
from ..monitoring.quality_checker import QualityChecker, QualityReport as LegacyQualityReport
from ..monitoring.drift_detector import DriftDetector
from ..monitoring.audit_logger import AuditLogger

# Schemas
try:
    from app.schemas.assessment import (
        EnhancedSpeechAnalysisResponse, BiomarkerResult,
        EnhancedBiomarkers, FileInfo
    )
except ImportError:
    # Fallback for standalone usage
    pass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result with all components."""
    # Session info
    session_id: str = ""
    timestamp: str = ""
    processing_time_s: float = 0.0
    
    # Quality
    quality_report: Optional[QualityReport] = None
    quality_acceptable: bool = True
    
    # Features
    features: Optional[UnifiedFeatures] = None
    
    # Risk assessment
    risk_result: Optional[RiskAssessmentResult] = None
    risk_score: float = 0.0
    risk_level: str = "unknown"
    
    # Uncertainty
    uncertainty: Optional[UncertaintyResult] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Explainability
    explanation: Optional[RiskExplanation] = None
    
    # Normative comparison
    normative_comparisons: Dict[str, Dict] = None
    
    # Recommendations
    recommendations: List[str] = None
    clinical_notes: str = ""
    
    # Flags
    requires_review: bool = False
    review_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.normative_comparisons is None:
            self.normative_comparisons = {}
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "processing_time_s": self.processing_time_s,
            "quality": {
                "acceptable": self.quality_acceptable,
                "score": self.quality_report.quality_score if self.quality_report else 0,
                "issues": self.quality_report.issues if self.quality_report else []
            },
            "risk": {
                "score": self.risk_score,
                "level": self.risk_level,
                "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None
            },
            "recommendations": self.recommendations,
            "clinical_notes": self.clinical_notes,
            "requires_review": self.requires_review,
            "review_reason": self.review_reason
        }
        
        if self.features:
            result["features"] = self.features.to_dict()
        
        if self.explanation:
            result["explanation"] = self.explanation.to_dict()
        
        if self.normative_comparisons:
            result["normative_comparisons"] = self.normative_comparisons
        
        return result


class ResearchGradeSpeechServiceV4:
    """
    Research-grade speech analysis service v4.0.
    
    Orchestrates the complete speech analysis pipeline with:
    - Multi-format audio input support
    - Quality-gated processing
    - Parallel feature extraction
    - Clinical risk assessment with uncertainty
    - Real-time streaming analysis
    - Explainable predictions
    """
    
    VERSION = "4.0.0"
    
    def __init__(
        self,
        config: Optional[ResearchPipelineConfig] = None,
        enable_streaming: bool = True,
        enable_explanations: bool = True,
        enable_uncertainty: bool = True
    ):
        self.config = config or ResearchPipelineConfig()
        self.enable_streaming = enable_streaming
        self.enable_explanations = enable_explanations
        self.enable_uncertainty = enable_uncertainty
        
        # Initialize core components
        self._init_components()
        
        logger.info(
            f"ResearchGradeSpeechServiceV4 initialized: "
            f"streaming={enable_streaming}, explanations={enable_explanations}, "
            f"uncertainty={enable_uncertainty}"
        )
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # Quality gate
        self.quality_gate = EnhancedQualityGate(
            sample_rate=self.config.sample_rate,
            min_snr_db=self.config.quality_gate.min_snr_db,
            max_clipping_ratio=self.config.quality_gate.max_clipping_ratio,
            min_speech_ratio=self.config.quality_gate.min_speech_ratio
        )
        
        # Feature extraction
        self.feature_extractor = UnifiedFeatureExtractor(
            sample_rate=self.config.sample_rate,
            config=self.config,
            use_parallel=True,
            extract_embeddings=self.config.deep_learning.extract_embeddings
        )
        
        # Clinical risk scoring
        self.risk_scorer = ClinicalRiskScorer()
        
        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(
            n_samples=50,  # Reduced for performance
            feature_noise_std=0.05
        )
        
        # Explainability
        self.risk_explainer = RiskExplainer(
            n_perturbations=30
        )
        
        # Normative data
        self.normative_manager = NormativeDataManager()
        
        # Streaming (lazy loaded)
        self._session_manager = None
        self._streaming_analyzers: Dict[str, StreamingAnalyzer] = {}
        
        # Monitoring
        self.drift_detector = DriftDetector()
        self.audit_logger = AuditLogger()
    
    @property
    def session_manager(self) -> StreamingSessionManager:
        """Lazy-loaded session manager."""
        if self._session_manager is None:
            self._session_manager = StreamingSessionManager()
        return self._session_manager
    
    async def analyze(
        self,
        audio_bytes: bytes,
        session_id: str,
        filename: str = "",
        content_type: Optional[str] = None,
        patient_id: Optional[str] = None,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive speech analysis.
        
        Args:
            audio_bytes: Audio file content
            session_id: Unique session identifier
            filename: Original filename
            content_type: MIME type
            patient_id: Optional patient identifier
            patient_age: Optional age for normalization
            patient_sex: Optional sex for normalization
            
        Returns:
            AnalysisResult with complete analysis
        """
        start_time = time.time()
        
        result = AnalysisResult(
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # 1. Quality validation and audio loading
            audio = await self._validate_and_load_audio(
                audio_bytes, filename, content_type, result
            )
            
            if not result.quality_acceptable:
                result.processing_time_s = time.time() - start_time
                return result
            
            # 2. Feature extraction
            result.features = await self._extract_features(audio)
            
            # 3. Clinical risk assessment
            await self._assess_risk(
                result, patient_age, patient_sex
            )
            
            # 4. Uncertainty quantification
            if self.enable_uncertainty:
                await self._estimate_uncertainty(result)
            
            # 5. Explainability
            if self.enable_explanations:
                await self._generate_explanation(result)
            
            # 6. Normative comparison
            await self._compare_normative(
                result, patient_age, patient_sex
            )
            
            # 7. Final recommendations
            self._generate_recommendations(result)
            
            # 8. Drift detection
            self._check_drift(result)
            
            # 9. Audit logging
            await self._log_audit(
                result, audio_bytes, patient_id
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            result.quality_acceptable = False
            result.recommendations.append(f"Analysis error: {str(e)}")
        
        result.processing_time_s = time.time() - start_time
        return result
    
    async def _validate_and_load_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: Optional[str],
        result: AnalysisResult
    ) -> Optional[np.ndarray]:
        """Validate audio quality and load for processing."""
        # Use enhanced quality gate
        quality_report = self.quality_gate.validate_audio_bytes(
            audio_bytes, filename, content_type
        )
        result.quality_report = quality_report
        
        if not quality_report.is_acceptable:
            result.quality_acceptable = False
            result.recommendations.extend(quality_report.issues)
            return None
        
        # Get converted audio from quality gate
        audio = quality_report.format_result.converted_audio
        
        if audio is None:
            result.quality_acceptable = False
            result.recommendations.append("Audio conversion failed")
            return None
        
        # Duration validation
        duration = len(audio) / self.config.sample_rate
        if duration < self.config.input_constraints.min_duration_s:
            result.quality_acceptable = False
            result.recommendations.append(
                f"Audio too short: {duration:.1f}s "
                f"(minimum {self.config.input_constraints.min_duration_s}s)"
            )
            return None
        
        # Trim if too long
        max_duration = self.config.input_constraints.max_duration_s
        if duration > max_duration:
            logger.warning(f"Truncating audio from {duration:.1f}s to {max_duration}s")
            audio = audio[:int(max_duration * self.config.sample_rate)]
        
        # Silence trimming
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        return audio
    
    async def _extract_features(
        self,
        audio: np.ndarray
    ) -> UnifiedFeatures:
        """Extract all features from audio."""
        # Run in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            None,
            self.feature_extractor.extract_full,
            audio
        )
        return features
    
    async def _assess_risk(
        self,
        result: AnalysisResult,
        patient_age: Optional[int],
        patient_sex: Optional[str]
    ):
        """Perform clinical risk assessment."""
        if not result.features:
            return
        
        # Map features to scorer format
        feature_dict = self._map_features_for_scorer(result.features.to_dict())
        
        # Quality score from report
        quality_score = (
            result.quality_report.quality_score / 100.0
            if result.quality_report else 0.9
        )
        
        # Assess risk
        risk_result = self.risk_scorer.assess_risk(
            features=feature_dict,
            signal_quality=quality_score,
            patient_age=patient_age,
            patient_sex=patient_sex
        )
        
        result.risk_result = risk_result
        result.risk_score = risk_result.overall_score
        result.risk_level = risk_result.risk_level.value
        result.confidence_interval = risk_result.confidence_interval
        result.requires_review = risk_result.requires_review
        result.review_reason = risk_result.review_reason
        
        # Clinical notes
        result.clinical_notes = self._generate_clinical_notes(risk_result)
    
    async def _estimate_uncertainty(self, result: AnalysisResult):
        """Estimate prediction uncertainty."""
        if not result.features or not result.risk_result:
            return
        
        feature_dict = self._map_features_for_scorer(result.features.to_dict())
        
        def score_function(features):
            assessment = self.risk_scorer.assess_risk(features)
            return assessment.overall_score
        
        result.uncertainty = self.uncertainty_estimator.estimate(
            features=feature_dict,
            score_function=score_function
        )
        
        if result.uncertainty:
            result.confidence_interval = result.uncertainty.ci_95
    
    async def _generate_explanation(self, result: AnalysisResult):
        """Generate SHAP-style explanation."""
        if not result.features or not result.risk_result:
            return
        
        feature_dict = self._map_features_for_scorer(result.features.to_dict())
        
        def score_function(features):
            assessment = self.risk_scorer.assess_risk(features)
            return assessment.overall_score
        
        # Get normal ranges from normative manager
        normal_ranges = {}
        for name in feature_dict.keys():
            ref = self.normative_manager.get_reference(name)
            if ref:
                normal_ranges[name] = (ref.lower_limit, ref.upper_limit)
        
        result.explanation = self.risk_explainer.explain(
            features=feature_dict,
            score_function=score_function,
            normal_ranges=normal_ranges
        )
    
    async def _compare_normative(
        self,
        result: AnalysisResult,
        patient_age: Optional[int],
        patient_sex: Optional[str]
    ):
        """Compare features against normative data."""
        if not result.features:
            return
        
        from ..clinical.normative import Sex
        
        sex = None
        if patient_sex:
            sex = Sex.MALE if patient_sex.lower() == "male" else Sex.FEMALE
        
        feature_dict = result.features.to_dict()
        comparisons = {}
        
        for name in ["jitter_local", "shimmer_local", "hnr", "cpps", 
                     "speech_rate", "pause_ratio", "tremor_score"]:
            value = feature_dict.get(name)
            if value is None or np.isnan(value):
                continue
            
            z_score = self.normative_manager.calculate_z_score(
                name, value, patient_age, sex
            )
            percentile = self.normative_manager.calculate_percentile(
                name, value, patient_age, sex
            )
            interpretation = self.normative_manager.get_interpretation(
                name, value, patient_age, sex
            )
            
            comparisons[name] = {
                "value": value,
                "z_score": z_score,
                "percentile": percentile,
                "interpretation": interpretation
            }
        
        result.normative_comparisons = comparisons
    
    def _generate_recommendations(self, result: AnalysisResult):
        """Generate final recommendations."""
        if result.risk_result:
            result.recommendations.extend(result.risk_result.recommendations)
        
        # Add explanation-based recommendations
        if result.explanation and result.explanation.top_risk_factors:
            factors = ", ".join(result.explanation.top_risk_factors[:3])
            result.recommendations.append(
                f"Key areas of concern: {factors}"
            )
    
    def _check_drift(self, result: AnalysisResult):
        """Check for distribution drift."""
        if not result.features:
            return
        
        feature_dict = result.features.to_dict()
        drift_report = self.drift_detector.check_input_drift(feature_dict)
        
        if drift_report.has_drift:
            logger.warning(f"Drift detected: {drift_report.recommendation}")
    
    async def _log_audit(
        self,
        result: AnalysisResult,
        audio_bytes: bytes,
        patient_id: Optional[str]
    ):
        """Log analysis for audit trail."""
        try:
            entry = self.audit_logger.create_entry(
                session_id=result.session_id,
                patient_id=patient_id,
                audio_bytes=audio_bytes,
                audio_duration=result.features.audio_duration if result.features else 0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                risk_score=result.risk_score,
                risk_level=result.risk_level,
                condition_probs={},
                confidence=result.risk_result.confidence if result.risk_result else 0,
                signal_quality=result.quality_report.quality_score if result.quality_report else 0,
                quality_issues=result.quality_report.issues if result.quality_report else [],
                features_extracted=list(result.features.to_dict().keys()) if result.features else [],
                requires_review=result.requires_review,
                review_reason=result.review_reason
            )
            self.audit_logger.log(entry)
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
    
    def _map_features_for_scorer(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Map extracted features to scorer expected format."""
        return {
            "jitter": features.get("jitter_local", 0.0),
            "shimmer": features.get("shimmer_local", 0.0),
            "hnr": features.get("hnr", 20.0),
            "cpps": features.get("cpps", 15.0),
            "speech_rate": features.get("speech_rate", 4.5),
            "pause_ratio": features.get("pause_ratio", 0.2),
            "voice_tremor": features.get("tremor_score", 0.0),
            "fluency_score": 1.0 - features.get("pause_ratio", 0.2),
            "articulation_clarity": features.get("fcr", 1.0),
            "prosody_variation": features.get("std_f0", 30.0),
            "energy_mean": 0.5
        }
    
    def _generate_clinical_notes(
        self,
        risk_result: RiskAssessmentResult
    ) -> str:
        """Generate clinical interpretation notes."""
        notes = []
        
        if risk_result.risk_level.value == "low":
            notes.append("Voice biomarkers within normal ranges.")
        elif risk_result.risk_level.value == "moderate":
            notes.append("Some biomarkers outside typical ranges - monitoring recommended.")
        elif risk_result.risk_level.value == "high":
            notes.append("Multiple biomarkers indicate potential changes - evaluation recommended.")
        else:
            notes.append("Significant abnormalities - clinical evaluation strongly recommended.")
        
        for dev in risk_result.biomarker_deviations:
            if dev.status == "abnormal" and dev.risk_contribution > 0.1:
                notes.append(f"{dev.name}: {dev.status} (z={dev.z_score:.1f})")
        
        return " ".join(notes)
    
    # Streaming API
    
    def create_streaming_session(
        self,
        user_id: Optional[str] = None,
        client_info: Optional[Dict] = None
    ) -> StreamingSession:
        """Create a new streaming session."""
        if not self.enable_streaming:
            raise RuntimeError("Streaming not enabled")
        
        session = self.session_manager.create_session(
            user_id=user_id,
            client_info=client_info
        )
        
        # Create analyzer for session
        self._streaming_analyzers[session.session_id] = StreamingAnalyzer(
            sample_rate=self.config.sample_rate
        )
        
        return session
    
    def get_streaming_session(
        self,
        session_id: str
    ) -> Optional[StreamingSession]:
        """Get streaming session by ID."""
        return self.session_manager.get_session(session_id)
    
    async def process_streaming_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray
    ) -> Dict:
        """Process a streaming audio chunk."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Add to session
        self.session_manager.add_audio_chunk(session_id, audio_chunk)
        
        # Process with analyzer
        analyzer = self._streaming_analyzers.get(session_id)
        if analyzer:
            result = analyzer.process_chunk(audio_chunk)
            return result.to_dict()
        
        return {"error": "Analyzer not found"}
    
    async def finalize_streaming_session(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> StreamingResult:
        """Finalize streaming session with full analysis."""
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Stop recording
        self.session_manager.stop_recording(session_id)
        
        # Get analyzer
        analyzer = self._streaming_analyzers.get(session_id)
        if not analyzer:
            raise ValueError(f"Analyzer not found: {session_id}")
        
        # Finalize
        result = analyzer.finalize_session(session)
        
        # Complete session
        self.session_manager.complete_session(session_id)
        
        # Cleanup
        if session_id in self._streaming_analyzers:
            del self._streaming_analyzers[session_id]
        
        return result


# Backward compatibility alias
ResearchGradeSpeechService = ResearchGradeSpeechServiceV4
