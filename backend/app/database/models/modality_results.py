"""
SQLAlchemy Models - Modality-Specific Results
Retinal, Speech, Cardiology, Radiology, Dermatology, Cognitive
"""

from sqlalchemy import (
    Column, String, DECIMAL, SmallInteger, Integer,
    ForeignKey, JSON, Boolean, DateTime, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class RetinalResult(Base):
    """Retinal analysis results"""
    __tablename__ = "retinal_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # DR Grading
    dr_grade = Column(SmallInteger)
    dr_severity = Column(String(20))
    
    # Eye & Image metadata
    eye_laterality = Column(String(10), index=True)
    image_quality = Column(String(20))
    
    # 4-2-1 Rule
    four_two_one_met = Column(Boolean, default=False)
    hemorrhages_4_quadrants = Column(Boolean)
    venous_beading_2_quadrants = Column(Boolean)
    irma_1_quadrant = Column(Boolean)
    
    # DME
    dme_present = Column(Boolean)
    dme_severity = Column(String(20))
    
    # Biomarker aggregates
    vessel_density = Column(DECIMAL(5, 4))
    av_ratio = Column(DECIMAL(4, 3))
    tortuosity_index = Column(DECIMAL(4, 3))
    hemorrhage_count = Column(Integer)
    microaneurysm_count = Column(Integer)
    exudate_area_percent = Column(DECIMAL(5, 2))
    
    # Visualizations
    heatmap_data = Column(JSON)
    vessel_segmentation = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="retinal_result")
    
    def __repr__(self):
        return f"<RetinalResult(assessment_id={self.assessment_id}, dr_grade={self.dr_grade})>"


class SpeechResult(Base):
    """Speech/voice analysis results"""
    __tablename__ = "speech_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Audio metadata
    duration_seconds = Column(DECIMAL(6, 2))
    sample_rate = Column(Integer)
    audio_quality_score = Column(DECIMAL(4, 3), index=True)
    
    # Core biomarkers
    jitter = Column(DECIMAL(8, 6))
    shimmer = Column(DECIMAL(8, 6))
    hnr = Column(DECIMAL(6, 2))
    cpps = Column(DECIMAL(6, 2))
    speech_rate = Column(DECIMAL(5, 2))
    pause_ratio = Column(DECIMAL(4, 3))
    voice_tremor = Column(DECIMAL(4, 3))
    articulation_clarity = Column(DECIMAL(4, 3))
    prosody_variation = Column(DECIMAL(6, 2))
    fluency_score = Column(DECIMAL(4, 3))
    
    # Condition risks
    parkinsons_probability = Column(DECIMAL(4, 3))
    cognitive_decline_probability = Column(DECIMAL(4, 3))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="speech_result")
    
    def __repr__(self):
        return f"<SpeechResult(assessment_id={self.assessment_id})>"


class CardiologyResult(Base):
    """Cardiology/ECG analysis results"""
    __tablename__ = "cardiology_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Rhythm
    rhythm_classification = Column(String(50), index=True)
    heart_rate_bpm = Column(SmallInteger)
    regularity = Column(String(20))
    r_peaks_detected = Column(Integer)
    
    # HRV
    rmssd_ms = Column(DECIMAL(8, 2))
    sdnn_ms = Column(DECIMAL(8, 2))
    pnn50_percent = Column(DECIMAL(5, 2))
    
    # Intervals
    pr_interval_ms = Column(SmallInteger)
    qrs_duration_ms = Column(SmallInteger)
    qt_interval_ms = Column(SmallInteger)
    qtc_ms = Column(SmallInteger)
    
    # Risk flags
    arrhythmia_detected = Column(Boolean, default=False, index=True)
    arrhythmia_types = Column(JSON, default=list)
    
    signal_quality_score = Column(DECIMAL(4, 3))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="cardiology_result")
    
    def __repr__(self):
        return f"<CardiologyResult(assessment_id={self.assessment_id}, rhythm={self.rhythm_classification})>"


class RadiologyResult(Base):
    """Radiology/X-Ray analysis results"""
    __tablename__ = "radiology_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Modality
    modality_type = Column(String(50), index=True)
    
    # Primary findings
    primary_condition = Column(String(100), index=True)
    primary_probability = Column(DECIMAL(5, 2))
    primary_severity = Column(String(20))
    
    # Findings (structured)
    findings = Column(JSON, default=list)
    
    # Anatomical assessments
    lungs_status = Column(String(20))
    heart_status = Column(String(20))
    
    # Quality
    image_quality = Column(String(20))
    quality_score = Column(DECIMAL(4, 3))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="radiology_result")
    
    def __repr__(self):
        return f"<RadiologyResult(assessment_id={self.assessment_id}, condition={self.primary_condition})>"


class DermatologyResult(Base):
    """Dermatology/skin lesion analysis results"""
    __tablename__ = "dermatology_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Classification
    primary_classification = Column(String(100))
    melanoma_suspicion = Column(String(50), index=True)
    
    # ABCDE scores
    asymmetry_score = Column(DECIMAL(4, 3))
    border_score = Column(DECIMAL(4, 3))
    color_score = Column(DECIMAL(4, 3))
    diameter_mm = Column(DECIMAL(5, 2))
    evolution_score = Column(DECIMAL(4, 3))
    
    # Location
    body_location = Column(String(50))
    fitzpatrick_type = Column(SmallInteger)
    
    # Quality
    image_quality = Column(String(20))
    lesion_centered = Column(Boolean)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="dermatology_result")
    
    def __repr__(self):
        return f"<DermatologyResult(assessment_id={self.assessment_id}, suspicion={self.melanoma_suspicion})>"


class CognitiveResult(Base):
    """Cognitive testing results"""
    __tablename__ = "cognitive_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Overall scores
    overall_risk_score = Column(DECIMAL(4, 3))
    risk_level = Column(String(20), index=True)
    confidence_score = Column(DECIMAL(4, 3))
    
    # Domain scores
    attention_score = Column(DECIMAL(4, 3))
    memory_score = Column(DECIMAL(4, 3))
    executive_function_score = Column(DECIMAL(4, 3))
    processing_speed_score = Column(DECIMAL(4, 3))
    
    # Task metrics
    tasks_completed = Column(Integer)
    valid_tasks = Column(Integer)
    fatigue_index = Column(DECIMAL(4, 3))
    consistency_score = Column(DECIMAL(4, 3))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="cognitive_result")
    
    def __repr__(self):
        return f"<CognitiveResult(assessment_id={self.assessment_id}, risk={self.risk_level})>"
