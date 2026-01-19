"""
Cardiology Pipeline - Main Service
Entry point for cardiology analysis orchestrating all modules.
"""

import numpy as np
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Import from parent package (cardiology/) since we're in core/ subfolder
from ..schemas import (
    CardiologyAnalysisResponse,
    CardiologyErrorResponse,
    RhythmAnalysis,
    HRVMetrics,
    HRVTimeDomain,
    AutonomicInterpretation,
    ECGIntervals,
    ECGAnalysisResult,
    RiskAssessment,
    QualityAssessment,
    ECGQuality,
    ClinicalFinding,
    StageStatus,
    ReceiptConfirmation,
    ArrhythmiaDetection,
)

from ..input.validator import validate_cardiology_input, ValidationReport
from ..input.ecg_parser import ECGParser
from ..preprocessing.ecg_processor import ECGProcessor
from ..preprocessing.quality_gate import QualityGate
from ..features.ecg_features import ECGFeatureExtractor
from ..analysis.rhythm_classifier import RhythmClassifier
from ..analysis.arrhythmia_detector import ArrhythmiaDetector
from ..clinical.risk_scorer import CardiacRiskScorer
from ..clinical.recommendations import RecommendationGenerator
from ..errors.codes import PipelineError, ValidationError

logger = logging.getLogger(__name__)


class CardiologyAnalysisService:
    """
    Main cardiology analysis service.
    
    Orchestrates the complete analysis pipeline:
    1. Receipt & Validation
    2. Preprocessing
    3. Feature Extraction
    4. Analysis (Rhythm, Arrhythmia)
    5. Risk Scoring
    6. Response Formatting
    """
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
        
        # Initialize components
        self.ecg_processor = ECGProcessor(target_sample_rate=sample_rate)
        self.quality_gate = QualityGate()
        self.feature_extractor = ECGFeatureExtractor(sample_rate)
        self.rhythm_classifier = RhythmClassifier(sample_rate)
        self.arrhythmia_detector = ArrhythmiaDetector(sample_rate)
        self.risk_scorer = CardiacRiskScorer()
        self.recommendation_generator = RecommendationGenerator()
        self.ecg_parser = ECGParser()
    
    def analyze(
        self,
        ecg_signal: Optional[np.ndarray] = None,
        ecg_sample_rate: int = 500,
        ecg_file_content: Optional[bytes] = None,
        ecg_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CardiologyAnalysisResponse:
        """
        Perform complete cardiology analysis.
        
        Args:
            ecg_signal: Raw ECG signal array
            ecg_sample_rate: ECG sample rate
            ecg_file_content: Raw ECG file bytes (alternative to signal)
            ecg_filename: ECG filename for format detection
            metadata: Clinical metadata
        
        Returns:
            CardiologyAnalysisResponse with all results
        """
        start_time = time.time()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        stages = []
        
        try:
            # Stage 1: Receipt
            stage_start = time.time()
            modalities = []
            
            if ecg_signal is not None or ecg_file_content is not None:
                modalities.append("ecg_signal")
            if metadata:
                modalities.append("clinical_metadata")
            
            stages.append(StageStatus(
                stage="RECEIPT",
                status="success",
                time_ms=int((time.time() - stage_start) * 1000),
            ))
            
            # Parse ECG file if provided
            if ecg_file_content is not None and ecg_filename:
                stage_start = time.time()
                ecg_signal, ecg_sample_rate, _ = self.ecg_parser.parse(
                    ecg_file_content, ecg_filename, ecg_sample_rate
                )
            
            # Stage 2: Validation
            stage_start = time.time()
            validation = validate_cardiology_input(
                ecg_signal=ecg_signal,
                ecg_sample_rate=ecg_sample_rate,
                metadata=metadata,
            )
            
            if not validation.is_valid:
                raise ValidationError(
                    code=validation.errors[0].check_id if validation.errors else "E_VAL_001",
                    message=validation.errors[0].message if validation.errors else "Validation failed",
                    details=validation.to_dict(),
                )
            
            stages.append(StageStatus(
                stage="VALIDATION",
                status="success",
                time_ms=int((time.time() - stage_start) * 1000),
            ))
            
            # Stage 3: Preprocessing
            stage_start = time.time()
            processed = self.ecg_processor.preprocess(
                ecg_signal,
                original_sample_rate=ecg_sample_rate,
                remove_powerline=True,
                normalize=True,
            )
            
            # Quality gate
            quality_result = self.quality_gate.assess(processed.signal, processed.sample_rate)
            
            stages.append(StageStatus(
                stage="PREPROCESSING",
                status="success",
                time_ms=int((time.time() - stage_start) * 1000),
            ))
            
            # Stage 4: Feature Extraction
            stage_start = time.time()
            features = self.feature_extractor.extract(processed.signal)
            
            stages.append(StageStatus(
                stage="FEATURE_EXTRACTION",
                status="success",
                time_ms=int((time.time() - stage_start) * 1000),
            ))
            
            # Stage 5: Analysis
            stage_start = time.time()
            
            # Rhythm classification
            rhythm = self.rhythm_classifier.classify(
                features.rr_intervals,
                features.heart_rate_bpm,
            )
            
            # Arrhythmia detection
            arrhythmia_result = self.arrhythmia_detector.detect(
                features.rr_intervals,
                features.heart_rate_bpm,
                processed.signal,
                features.r_peaks,
            )
            
            stages.append(StageStatus(
                stage="ANALYSIS",
                status="success",
                time_ms=int((time.time() - stage_start) * 1000),
            ))
            
            # Stage 6: Risk Scoring
            stage_start = time.time()
            
            ecg_results = {
                "heart_rate_bpm": features.heart_rate_bpm,
                "confidence": rhythm.confidence,
                "hrv_metrics": {
                    "time_domain": features.hrv.to_dict(),
                },
                "arrhythmias_detected": [a.to_dict() for a in arrhythmia_result.arrhythmias],
            }
            
            risk = self.risk_scorer.calculate(
                ecg_results=ecg_results,
                metadata=metadata,
            )
            
            stages.append(StageStatus(
                stage="SCORING",
                status="success",
                time_ms=int((time.time() - stage_start) * 1000),
            ))
            
            # Stage 7: Generate recommendations
            recommendations = self.recommendation_generator.generate(
                rhythm=rhythm.classification,
                heart_rate=features.heart_rate_bpm,
                hrv_metrics=ecg_results.get("hrv_metrics"),
                arrhythmias=[a.to_dict() for a in arrhythmia_result.arrhythmias],
                risk_category=risk.risk_category,
            )
            
            # Generate findings
            findings = self._generate_findings(rhythm, arrhythmia_result, features)
            
            # Build response
            processing_time = int((time.time() - start_time) * 1000)
            
            return CardiologyAnalysisResponse(
                success=True,
                request_id=request_id,
                processing_time_ms=processing_time,
                receipt=ReceiptConfirmation(
                    acknowledged=True,
                    modalities_received=modalities,
                ),
                stages_completed=stages,
                ecg_analysis=ECGAnalysisResult(
                    rhythm_analysis=RhythmAnalysis(
                        classification=rhythm.classification,
                        heart_rate_bpm=int(features.heart_rate_bpm),
                        confidence=rhythm.confidence,
                        regularity=rhythm.regularity,
                        r_peaks_detected=len(features.r_peaks),
                        rr_variability_cv=rhythm.rr_variability,
                    ),
                    hrv_metrics=HRVMetrics(
                        time_domain=HRVTimeDomain(
                            rmssd_ms=features.hrv.rmssd_ms,
                            sdnn_ms=features.hrv.sdnn_ms,
                            pnn50_percent=features.hrv.pnn50_percent,
                            mean_rr_ms=features.hrv.mean_rr_ms,
                            sdsd_ms=features.hrv.sdsd_ms,
                            cv_rr_percent=features.hrv.cv_rr_percent,
                        ),
                        interpretation=AutonomicInterpretation(
                            autonomic_balance=self._interpret_autonomic(features.hrv),
                            parasympathetic="adequate" if (features.hrv.rmssd_ms or 0) > 20 else "low",
                            sympathetic="normal",
                        ),
                    ),
                    intervals=ECGIntervals(
                        qrs_duration_ms=features.intervals.qrs_duration_ms,
                        qt_interval_ms=features.intervals.qt_interval_ms,
                        qtc_ms=features.intervals.qtc_ms,
                        all_normal=features.intervals.all_normal,
                    ),
                    arrhythmias_detected=[
                        ArrhythmiaDetection(
                            type=a.type,
                            confidence=a.confidence,
                            urgency=a.urgency,
                            count=a.count,
                            description=a.description,
                        )
                        for a in arrhythmia_result.arrhythmias
                    ],
                    signal_quality_score=quality_result.quality_score,
                ),
                findings=findings,
                risk_assessment=RiskAssessment(
                    risk_score=risk.risk_score,
                    risk_category=risk.risk_category,
                    risk_factors=[
                        {"factor": rf.factor, "severity": rf.severity}
                        for rf in risk.risk_factors
                    ],
                    confidence=risk.confidence,
                ),
                recommendations=recommendations,
                quality_assessment=QualityAssessment(
                    overall_quality=quality_result.quality_grade,
                    ecg_quality=ECGQuality(
                        signal_quality_score=quality_result.quality_score,
                        snr_db=quality_result.snr_db,
                        usable_segments_percent=quality_result.usable_ratio * 100,
                        artifacts_detected=len(quality_result.artifacts),
                    ),
                ),
                metadata_used=metadata is not None,
            )
            
        except PipelineError as e:
            logger.error(f"Pipeline error: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            # Return error response as success=False
            return CardiologyAnalysisResponse(
                success=False,
                request_id=request_id,
                processing_time_ms=processing_time,
                stages_completed=stages,
                risk_assessment=RiskAssessment(
                    risk_score=0,
                    risk_category="unknown",
                    confidence=0,
                ),
                quality_assessment=QualityAssessment(overall_quality="unknown"),
                warnings=[{"code": e.code, "message": e.message}],
            )
        
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return CardiologyAnalysisResponse(
                success=False,
                request_id=request_id,
                processing_time_ms=processing_time,
                stages_completed=stages,
                risk_assessment=RiskAssessment(
                    risk_score=0,
                    risk_category="unknown",
                    confidence=0,
                ),
                quality_assessment=QualityAssessment(overall_quality="unknown"),
                warnings=[{"code": "E_SYS_001", "message": str(e)}],
            )
    
    def _interpret_autonomic(self, hrv) -> str:
        """Interpret autonomic balance from HRV."""
        rmssd = hrv.rmssd_ms or 40
        
        if rmssd < 20:
            return "sympathetic_dominant"
        elif rmssd > 60:
            return "parasympathetic_dominant"
        else:
            return "balanced"
    
    def _generate_findings(self, rhythm, arrhythmia_result, features) -> List[ClinicalFinding]:
        """Generate clinical findings list."""
        findings = []
        finding_id = 0
        
        # Rhythm finding
        finding_id += 1
        severity = "normal" if rhythm.classification == "Normal Sinus Rhythm" else "mild"
        findings.append(ClinicalFinding(
            id=f"finding_{finding_id:03d}",
            type="observation",
            title=rhythm.classification,
            severity=severity,
            description=f"Heart rate {features.heart_rate_bpm:.0f} bpm, {rhythm.regularity} rhythm",
            source="ecg",
            confidence=rhythm.confidence,
        ))
        
        # Arrhythmia findings
        for arr in arrhythmia_result.arrhythmias:
            finding_id += 1
            findings.append(ClinicalFinding(
                id=f"finding_{finding_id:03d}",
                type="abnormality",
                title=arr.type.replace("_", " ").title(),
                severity="moderate" if arr.urgency == "moderate" else "mild",
                description=arr.description,
                source="ecg",
                confidence=arr.confidence,
            ))
        
        # HRV finding
        if features.hrv.rmssd_ms:
            finding_id += 1
            hrv_status = "normal" if 20 <= features.hrv.rmssd_ms <= 60 else "abnormal"
            findings.append(ClinicalFinding(
                id=f"finding_{finding_id:03d}",
                type="observation",
                title=f"Heart Rate Variability: {hrv_status.title()}",
                severity="normal" if hrv_status == "normal" else "mild",
                description=f"RMSSD {features.hrv.rmssd_ms:.1f} ms",
                source="ecg",
            ))
        
        return findings
