"""
Research-Grade Speech Analysis Service v3.0
Comprehensive speech analysis pipeline for medical biomarker detection.

Integrates:
- Acoustic feature extraction (Parselmouth/Praat)
- Prosodic analysis (speech rate, tremor, pauses)
- Composite biomarkers (NII, VFMT, ACE, RPCS)
- Self-supervised embeddings (Whisper, Wav2Vec)
- Clinical risk assessment with uncertainty
- Production monitoring (quality, drift, audit)
"""

import logging
import numpy as np
import io
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# Third-party
import librosa
from pydub import AudioSegment
import parselmouth
from parselmouth.praat import call

# Feature extractors
from ..features.acoustic import AcousticFeatureExtractor, AcousticFeatures
from ..features.prosodic import ProsodicFeatureExtractor, ProsodicFeatures
from ..features.composite import CompositeFeatureExtractor, CompositeBiomarkers
from ..features.embeddings import EmbeddingExtractor, SpeechEmbeddings

# Clinical analysis
from ..clinical.risk_scorer import ClinicalRiskScorer, RiskAssessmentResult
from ..clinical.uncertainty import UncertaintyEstimator

# Monitoring
from ..monitoring.quality_checker import QualityChecker, QualityReport
from ..monitoring.drift_detector import DriftDetector
from ..monitoring.audit_logger import AuditLogger

# Schemas
from app.schemas.assessment import (
    EnhancedSpeechAnalysisResponse, BiomarkerResult,
    EnhancedBiomarkers, FileInfo
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the speech analysis pipeline."""
    sample_rate: int = 16000
    min_duration: float = 0.5
    max_duration: float = 60.0
    
    # Feature extraction
    extract_embeddings: bool = False  # Disabled by default (heavy)
    extract_composite: bool = True
    
    # Quality thresholds
    min_snr_db: float = 10.0
    
    # Clinical
    use_uncertainty: bool = True
    
    # Monitoring
    enable_audit: bool = True
    enable_drift_detection: bool = True


class ResearchGradeSpeechService:
    """
    Research-grade speech analysis service.
    
    Provides comprehensive voice biomarker extraction with
    clinical-grade accuracy and uncertainty quantification.
    """
    
    VERSION = "3.0.0"
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize extractors
        self.acoustic_extractor = AcousticFeatureExtractor(
            sample_rate=self.config.sample_rate
        )
        self.prosodic_extractor = ProsodicFeatureExtractor(
            sample_rate=self.config.sample_rate
        )
        self.composite_extractor = CompositeFeatureExtractor(
            sample_rate=self.config.sample_rate
        )
        
        # Embeddings (lazy loaded)
        self.embedding_extractor = None
        if self.config.extract_embeddings:
            self.embedding_extractor = EmbeddingExtractor(
                sample_rate=self.config.sample_rate,
                load_whisper=True
            )
        
        # Clinical analysis
        self.risk_scorer = ClinicalRiskScorer()
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Monitoring
        self.quality_checker = QualityChecker(
            sample_rate=self.config.sample_rate,
            min_snr_db=self.config.min_snr_db
        )
        self.drift_detector = DriftDetector()
        self.audit_logger = AuditLogger() if self.config.enable_audit else None
        
        logger.info(f"ResearchGradeSpeechService v{self.VERSION} initialized")
    
    async def analyze(
        self,
        audio_bytes: bytes,
        session_id: str,
        filename: str,
        content_type: Optional[str] = None,
        patient_id: Optional[str] = None,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> EnhancedSpeechAnalysisResponse:
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
            EnhancedSpeechAnalysisResponse with full analysis
        """
        start_time = datetime.now()
        
        try:
            # 1. Load and validate audio
            audio, duration = self._load_audio(audio_bytes, filename)
            
            # 2. Quality check
            quality_report = self.quality_checker.check(audio)
            
            if not quality_report.is_acceptable:
                raise ValueError(
                    f"Audio quality insufficient: {'; '.join(quality_report.issues)}"
                )
            
            # 3. Extract features
            all_features, f0_contour, f1_contour, f2_contour = await self._extract_all_features(audio)
            
            # 4. Clinical risk assessment
            risk_result = self.risk_scorer.assess_risk(
                features=all_features,
                signal_quality=quality_report.quality_score,
                patient_age=patient_age,
                patient_sex=patient_sex
            )
            
            # 5. Drift detection (if enabled)
            if self.config.enable_drift_detection:
                drift_report = self.drift_detector.check_input_drift(all_features)
                if drift_report.has_drift:
                    logger.warning(f"Drift detected: {drift_report.recommendation}")
            
            # 6. Build response
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            response = self._build_response(
                session_id=session_id,
                processing_time=processing_time,
                risk_result=risk_result,
                all_features=all_features,
                quality_report=quality_report,
                file_info=FileInfo(
                    filename=filename,
                    size=len(audio_bytes),
                    content_type=content_type,
                    duration=duration,
                    sample_rate=self.config.sample_rate
                )
            )
            
            # 7. Audit log
            if self.audit_logger:
                self._log_audit(
                    session_id=session_id,
                    patient_id=patient_id,
                    audio_bytes=audio_bytes,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                    risk_result=risk_result,
                    quality_report=quality_report,
                    all_features=all_features
                )
            
            return response
            
        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise ValueError(f"Speech analysis failed: {str(e)}")
    
    def _load_audio(
        self,
        audio_bytes: bytes,
        filename: str
    ) -> Tuple[np.ndarray, float]:
        """Load and preprocess audio."""
        suffix = os.path.splitext(filename)[1] if filename else ".wav"
        if not suffix:
            suffix = ".wav"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            # Load with pydub (robust format handling)
            audio = AudioSegment.from_file(temp_path)
            
            # Duration check
            if audio.duration_seconds < self.config.min_duration:
                raise ValueError(
                    f"Audio too short: {audio.duration_seconds:.1f}s "
                    f"(minimum: {self.config.min_duration}s)"
                )
            
            if audio.duration_seconds > self.config.max_duration:
                logger.warning(
                    f"Audio truncated from {audio.duration_seconds:.1f}s "
                    f"to {self.config.max_duration}s"
                )
                audio = audio[:int(self.config.max_duration * 1000)]
            
            # Convert to mono, resample
            audio = audio.set_channels(1).set_frame_rate(self.config.sample_rate)
            
            # Convert to numpy float32
            samples = np.array(audio.get_array_of_samples())
            
            if audio.sample_width == 2:
                audio_data = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:
                audio_data = samples.astype(np.float32) / 2147483648.0
            else:
                audio_data = samples.astype(np.float32) / (2**(8*audio.sample_width - 1))
            
            # Trim silence
            audio_data, _ = librosa.effects.trim(audio_data, top_db=25)
            
            if len(audio_data) < self.config.sample_rate * self.config.min_duration:
                raise ValueError("Audio contains insufficient speech after silence removal")
            
            return audio_data, audio.duration_seconds
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def _extract_all_features(
        self,
        audio: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """Extract all features from audio."""
        
        # Acoustic features (Parselmouth)
        acoustic = self.acoustic_extractor.extract(audio)
        
        # Get F0 contour for prosodic analysis
        sound = parselmouth.Sound(audio, sampling_frequency=self.config.sample_rate)
        pitch = sound.to_pitch_ac(time_step=0.01)
        f0_contour = pitch.selected_array['frequency']
        
        # Get formant contours
        formants = sound.to_formant_burg(time_step=0.01)
        n_frames = formants.get_number_of_frames()
        f1_contour = np.array([
            call(formants, "Get value at time", 1, i * 0.01, "Hertz", "Linear")
            for i in range(n_frames)
        ])
        f2_contour = np.array([
            call(formants, "Get value at time", 2, i * 0.01, "Hertz", "Linear")
            for i in range(n_frames)
        ])
        
        # Prosodic features
        prosodic = self.prosodic_extractor.extract(audio, f0_contour)
        
        # Composite biomarkers
        if self.config.extract_composite:
            composite = self.composite_extractor.extract(
                audio=audio,
                acoustic_features=acoustic.to_dict(),
                prosodic_features=prosodic.to_dict(),
                f0_contour=f0_contour,
                f1_contour=f1_contour,
                f2_contour=f2_contour
            )
        else:
            composite = CompositeBiomarkers()
        
        # Combine all features
        all_features = {
            **acoustic.to_dict(),
            **prosodic.to_dict(),
            **composite.to_dict()
        }
        
        return all_features, f0_contour, f1_contour, f2_contour
    
    def _build_response(
        self,
        session_id: str,
        processing_time: float,
        risk_result: RiskAssessmentResult,
        all_features: Dict[str, float],
        quality_report: QualityReport,
        file_info: FileInfo
    ) -> EnhancedSpeechAnalysisResponse:
        """Build the API response."""
        from app.schemas.assessment import ExtendedBiomarkers, ConditionRisk
        
        # Map features to biomarker results
        biomarkers = self._map_to_biomarkers(all_features, risk_result)
        extended = self._map_extended_biomarkers(all_features)
        
        # Build condition risks
        condition_risks = []
        for cr in risk_result.condition_risks:
            condition_risks.append(ConditionRisk(
                condition=cr.condition,
                probability=cr.probability,
                confidence=cr.confidence,
                confidence_interval=cr.confidence_interval,
                risk_level=cr.risk_level,
                contributing_factors=cr.contributing_factors
            ))
        
        # Generate clinical notes
        clinical_notes = self._generate_clinical_notes(risk_result, all_features)
        
        return EnhancedSpeechAnalysisResponse(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            confidence=risk_result.confidence,
            risk_score=risk_result.overall_score / 100.0,  # Scale to 0-1
            quality_score=quality_report.quality_score,
            biomarkers=biomarkers,
            file_info=file_info,
            recommendations=risk_result.recommendations,
            status="completed",
            # Research-grade extensions
            extended_biomarkers=extended,
            condition_risks=condition_risks if condition_risks else None,
            confidence_interval=risk_result.confidence_interval,
            clinical_notes=clinical_notes,
            requires_review=risk_result.requires_review,
            review_reason=risk_result.review_reason
        )
    
    def _map_extended_biomarkers(self, features: Dict[str, float]):
        """Map extended research biomarkers."""
        from app.schemas.assessment import ExtendedBiomarkers
        
        def mk_result(value, normal_range, unit="", confidence=0.7):
            if value is None or np.isnan(value) or np.isinf(value):
                return None
            return BiomarkerResult(
                value=float(value),
                unit=unit,
                normal_range=normal_range,
                is_estimated=False,
                confidence=float(confidence)
            )
        
        return ExtendedBiomarkers(
            mean_f0=mk_result(features.get("mean_f0"), (85.0, 255.0), "Hz", 0.9),
            f0_range=mk_result(features.get("f0_range"), (50.0, 200.0), "Hz", 0.85),
            nii=mk_result(features.get("nii"), (0.0, 0.30), "idx", 0.6),
            vfmt=mk_result(features.get("vfmt_ratio"), (0.0, 0.5), "ratio", 0.5),
            ace=mk_result(features.get("ace"), (0.0, 1.0), "bits", 0.5),
            rpcs=mk_result(features.get("rpcs"), (0.5, 1.0), "coherence", 0.5)
        )
    
    def _generate_clinical_notes(
        self, 
        risk_result: RiskAssessmentResult,
        features: Dict[str, float]
    ) -> str:
        """Generate clinical interpretation notes."""
        notes = []
        
        # Risk level summary
        if risk_result.risk_level.value == "low":
            notes.append("Voice biomarkers within normal ranges.")
        elif risk_result.risk_level.value == "moderate":
            notes.append("Some biomarkers outside typical ranges - recommend monitoring.")
        elif risk_result.risk_level.value == "high":
            notes.append("Multiple biomarkers indicate potential voice/neurological changes.")
        else:
            notes.append("Significant abnormalities detected - clinical evaluation recommended.")
        
        # Specific findings
        for dev in risk_result.biomarker_deviations:
            if dev.status == "abnormal" and dev.risk_contribution > 0.1:
                notes.append(f"{dev.name}: {dev.status} (z={dev.z_score:.1f})")
        
        # Condition-specific notes
        for cr in risk_result.condition_risks:
            if cr.probability > 0.3:
                notes.append(f"Patterns consistent with {cr.condition} ({cr.probability:.0%})")
        
        return " ".join(notes) if notes else None

    
    def _map_to_biomarkers(
        self,
        features: Dict[str, float],
        risk_result: RiskAssessmentResult
    ) -> EnhancedBiomarkers:
        """Map features to API biomarker schema."""
        
        def mk_result(value, normal_range, unit="", confidence=0.9):
            if np.isnan(value) or np.isinf(value):
                value = 0.0
                confidence = 0.0
            return BiomarkerResult(
                value=float(value),
                unit=unit,
                normal_range=normal_range,
                is_estimated=False,
                confidence=float(confidence)
            )
        
        # Find confidence from risk result for specific biomarkers
        dev_map = {d.name: d for d in risk_result.biomarker_deviations}
        
        def get_conf(name, default=0.85):
            return dev_map.get(name, type('obj', (object,), {'confidence': default})).confidence
        
        return EnhancedBiomarkers(
            jitter=mk_result(
                features.get("jitter_local", 0),
                (0.00, 1.04), "%",
                get_conf("jitter_local", 0.95)
            ),
            shimmer=mk_result(
                features.get("shimmer_local", 0),
                (0.00, 3.81), "%",
                get_conf("shimmer_local", 0.95)
            ),
            hnr=mk_result(
                features.get("hnr", 0),
                (20.0, 30.0), "dB",
                get_conf("hnr", 0.9)
            ),
            cpps=mk_result(
                features.get("cpps", 0),
                (14.0, 30.0), "dB",
                get_conf("cpps", 0.95)
            ),
            speech_rate=mk_result(
                features.get("speech_rate", 0),
                (3.5, 6.5), "syl/s",
                0.85
            ),
            voice_tremor=mk_result(
                features.get("tremor_score", 0),
                (0.0, 0.15), "idx",
                get_conf("tremor_score", 0.7)
            ),
            articulation_clarity=mk_result(
                features.get("fcr", 1.0),
                (0.9, 1.1), "ratio",
                0.7
            ),
            prosody_variation=mk_result(
                features.get("std_f0", 0),
                (20.0, 100.0), "Hz",
                0.9
            ),
            fluency_score=mk_result(
                1.0 - features.get("pause_ratio", 0),
                (0.75, 1.0), "score",
                0.6
            ),
            pause_ratio=mk_result(
                features.get("pause_ratio", 0),
                (0.0, 0.25), "ratio",
                0.75
            )
        )
    
    def _log_audit(
        self,
        session_id: str,
        patient_id: Optional[str],
        audio_bytes: bytes,
        duration: float,
        start_time: datetime,
        end_time: datetime,
        risk_result: RiskAssessmentResult,
        quality_report: QualityReport,
        all_features: Dict[str, float]
    ):
        """Log audit entry."""
        if not self.audit_logger:
            return
        
        try:
            # Build condition probabilities
            condition_probs = {
                cr.condition: cr.probability
                for cr in risk_result.condition_risks
            }
            
            entry = self.audit_logger.create_entry(
                session_id=session_id,
                patient_id=patient_id,
                audio_bytes=audio_bytes,
                audio_duration=duration,
                start_time=start_time,
                end_time=end_time,
                risk_score=risk_result.overall_score,
                risk_level=risk_result.risk_level.value,
                condition_probs=condition_probs,
                confidence=risk_result.confidence,
                signal_quality=quality_report.quality_score,
                quality_issues=quality_report.issues,
                features_extracted=list(all_features.keys()),
                requires_review=risk_result.requires_review,
                review_reason=risk_result.review_reason
            )
            
            self.audit_logger.log(entry)
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")


# Backward compatibility: alias for existing code
SpeechPipelineService = ResearchGradeSpeechService
