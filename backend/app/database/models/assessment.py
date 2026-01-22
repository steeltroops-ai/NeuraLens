"""
SQLAlchemy Models - Assessment Core
Assessments, Pipeline Stages, Biomarkers
"""

from sqlalchemy import (
    Column, String, DateTime, Integer, ForeignKey,
    JSON, DECIMAL, SmallInteger, Text, Boolean
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class Assessment(Base):
    """Central assessment entity for all pipeline types"""
    __tablename__ = "assessments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), index=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Pipeline identity
    pipeline_type = Column(String(50), nullable=False, index=True)
    pipeline_version = Column(String(20), nullable=False, default='1.0.0')
    
    # Results
    risk_score = Column(DECIMAL(5, 2))
    risk_level = Column(String(20))
    confidence = Column(DECIMAL(4, 3))
    
    # Status
    status = Column(String(20), default='pending', index=True)
    current_stage = Column(String(50))
    
    # Processing metadata
    processing_time_ms = Column(Integer)
    quality_score = Column(DECIMAL(4, 3))
    requires_review = Column(Boolean, default=False, index=True)
    review_reason = Column(Text)
    
    # Full results JSON
    results = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="assessments")
    organization = relationship("Organization", back_populates="assessments")
    patient = relationship("Patient", back_populates="assessments")
    pipeline_stages = relationship("PipelineStage", back_populates="assessment", cascade="all, delete-orphan")
    biomarker_values = relationship("BiomarkerValue", back_populates="assessment", cascade="all, delete-orphan")
    ai_explanations = relationship("AIExplanation", back_populates="assessment", cascade="all, delete-orphan")
    uploaded_files = relationship("UploadedFile", back_populates="assessment")
    
    # Modality-specific results (one-to-one)
    retinal_result = relationship("RetinalResult", back_populates="assessment", uselist=False)
    speech_result = relationship("SpeechResult", back_populates="assessment", uselist=False)
    cardiology_result = relationship("CardiologyResult", back_populates="assessment", uselist=False)
    radiology_result = relationship("RadiologyResult", back_populates="assessment", uselist=False)
    dermatology_result = relationship("DermatologyResult", back_populates="assessment", uselist=False)
    cognitive_result = relationship("CognitiveResult", back_populates="assessment", uselist=False)
    
    def __repr__(self):
        return f"<Assessment(id={self.id}, pipeline={self.pipeline_type}, status={self.status})>"


class PipelineStage(Base):
    """Pipeline execution stage tracking"""
    __tablename__ = "pipeline_stages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, index=True)
    
    stage_name = Column(String(100), nullable=False)
    stage_index = Column(SmallInteger, nullable=False)
    status = Column(String(20), nullable=False)
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_ms = Column(Integer)
    
    error_code = Column(String(50))
    error_message = Column(Text)
    stage_metadata = Column(JSON, default=dict)
    
    # Relationships
    assessment = relationship("Assessment", back_populates="pipeline_stages")
    
    def __repr__(self):
        return f"<PipelineStage(assessment_id={self.assessment_id}, stage={self.stage_name})>"


class BiomarkerValue(Base):
    """Normalized biomarker values for analytics"""
    __tablename__ = "biomarker_values"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, index=True)
    
    biomarker_name = Column(String(100), nullable=False, index=True)
    biomarker_category = Column(String(50), index=True)
    
    value = Column(DECIMAL(12, 6), nullable=False)
    unit = Column(String(20))
    
    normal_range_min = Column(DECIMAL(12, 6))
    normal_range_max = Column(DECIMAL(12, 6))
    
    status = Column(String(20))
    confidence = Column(DECIMAL(4, 3))
    percentile = Column(SmallInteger)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="biomarker_values")
    
    def __repr__(self):
        return f"<BiomarkerValue(name={self.biomarker_name}, value={self.value})>"
