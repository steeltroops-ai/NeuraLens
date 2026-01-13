"""
Validation study database models for clinical validation tracking
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class ValidationStudy(Base):
    """Clinical validation study model"""
    __tablename__ = "validation_studies"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(String(255), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Study metadata
    name = Column(String(255), nullable=False)
    description = Column(String(1000))
    modality = Column(String(50), nullable=False)  # speech, retinal, motor, cognitive
    
    # Study parameters
    sample_size = Column(Integer)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    status = Column(String(50), default="active")  # active, completed, archived
    
    # Validation metrics
    accuracy = Column(Float)
    sensitivity = Column(Float)
    specificity = Column(Float)
    auc_roc = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Results relationship
    results = relationship("ValidationResult", back_populates="study", cascade="all, delete-orphan")


class ValidationResult(Base):
    """Individual validation result model"""
    __tablename__ = "validation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("validation_studies.id"), nullable=False)
    
    # Result metadata
    participant_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Prediction vs ground truth
    predicted_score = Column(Float)
    ground_truth = Column(Float)
    prediction_correct = Column(Integer)  # 1 for correct, 0 for incorrect
    
    # Additional metrics
    confidence = Column(Float)
    processing_time_ms = Column(Float)
    extra_data = Column(JSON)
    
    # Relationships
    study = relationship("ValidationStudy", back_populates="results")
