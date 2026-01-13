"""
Simplified Assessment database models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class Assessment(Base):
    """Assessment session model"""
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), ForeignKey("users.id"), nullable=True)
    
    # Assessment metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="in_progress")  # in_progress, completed, failed
    modalities = Column(JSON)  # List of modalities included
    
    # Results relationship
    results = relationship("AssessmentResult", back_populates="assessment", cascade="all, delete-orphan")


class AssessmentResult(Base):
    """Assessment results model"""
    __tablename__ = "assessment_results"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    
    # Result metadata
    modality = Column(String(50), nullable=False)  # speech, retinal, motor, cognitive
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Scores and metrics
    risk_score = Column(Float)  # 0.0 to 1.0
    confidence = Column(Float)  # 0.0 to 1.0
    biomarkers = Column(JSON)  # Biomarker measurements
    
    # Relationships
    assessment = relationship("Assessment", back_populates="results")


class NRIResult(Base):
    """NRI Fusion results model"""
    __tablename__ = "nri_results"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    
    # NRI scores
    nri_score = Column(Float, nullable=False)  # 0-100 NRI score
    risk_category = Column(String(50))  # low, moderate, high, very_high
    confidence = Column(Float)
    modality_contributions = Column(JSON)  # List of modality contributions
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    value = Column(Float, nullable=False)
    target_value = Column(Float)
    
    # Context
    timestamp = Column(DateTime, default=datetime.utcnow)
    environment = Column(String(50), default="development")
    version = Column(String(50))
    
    # Additional data
    extra_data = Column(JSON)
