import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, Integer, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.core.database import Base

class RetinalAssessment(Base):
    __tablename__ = "retinal_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True) # ForeignKey("users.id") - User model missing
    patient_id = Column(String, nullable=False, index=True)
    
    # Image metadata
    original_image_url = Column(String, nullable=False)
    processed_image_url = Column(String)
    heatmap_url = Column(String)
    segmentation_url = Column(String)
    image_format = Column(String)  # jpeg, png, dicom
    image_resolution = Column(String)  # "1024x1024"
    
    # Quality metrics
    quality_score = Column(Float)
    snr_db = Column(Float)
    has_optic_disc = Column(Boolean)
    has_macula = Column(Boolean)
    
    # Biomarkers - Vessels
    vessel_density = Column(Float)
    vessel_tortuosity = Column(Float)
    avr_ratio = Column(Float)
    branching_coefficient = Column(Float)
    vessel_confidence = Column(Float)
    
    # Biomarkers - Optic Disc
    cup_to_disc_ratio = Column(Float)
    disc_area_mm2 = Column(Float)
    rim_area_mm2 = Column(Float)
    optic_disc_confidence = Column(Float)
    
    # Biomarkers - Macula
    macular_thickness_um = Column(Float)
    macular_volume_mm3 = Column(Float)
    macula_confidence = Column(Float)
    
    # Biomarkers - Amyloid Beta
    amyloid_presence_score = Column(Float)
    amyloid_distribution = Column(String)
    amyloid_confidence = Column(Float)
    
    # Risk Assessment
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String, nullable=False)  # minimal, low, moderate, elevated, high, critical
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    
    # Processing metadata
    model_version = Column(String, nullable=False)
    processing_time_ms = Column(Integer)
    status = Column(String, default="completed")  # pending, processing, completed, failed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    # user = relationship("User", back_populates="retinal_assessments")
    # nri_contributions = relationship("NRIContribution", back_populates="retinal_assessment")
    
    # Indexes
    __table_args__ = (
        Index('idx_patient_created', 'patient_id', 'created_at'),
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_risk_category', 'risk_category'),
    )

class RetinalAuditLog(Base):
    __tablename__ = "retinal_audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    assessment_id = Column(String, ForeignKey("retinal_assessments.id"))
    user_id = Column(String, nullable=False) # ForeignKey("users.id")
    action = Column(String, nullable=False)  # create, view, update, delete, export
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata_json = Column(JSON, name="metadata") # 'metadata' is reserved in SQLAlchemy Base sometimes, using metadata_json mapped to metadata column name if needed, but here just simple column
    
    __table_args__ = (
        Index('idx_assessment_timestamp', 'assessment_id', 'timestamp'),
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
    )
