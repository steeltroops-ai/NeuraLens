"""
SQLAlchemy Models - Patient Management
"""

from sqlalchemy import (
    Column, String, Integer, Text, DateTime,
    func, Date, JSON, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.database import Base

class Patient(Base):
    """
    Patient entity for clinical profiles.
    Distinct from User accounts - patients may or may not have login access.
    """
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Identity
    full_name = Column(String(255), nullable=False, index=True)
    date_of_birth = Column(Date)
    gender = Column(String(20))
    phone_number = Column(String(50), index=True) # Unique constrained per org ideally, but simple index for now
    email = Column(String(255))
    
    # Contact & Admin
    address = Column(JSON, default=dict)
    emergency_contact = Column(JSON, default=dict)
    insurance_provider = Column(String(100))
    insurance_policy_number = Column(String(100))
    
    # Clinical Context
    medical_notes = Column(Text)
    medical_history = Column(JSON, default=list) # Past conditions
    medications = Column(JSON, default=list)
    allergies = Column(JSON, default=list)
    family_history = Column(JSON, default=list)
    
    # Multi-tenancy & Assignment
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), index=True)
    assigned_doctor_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_visit_at = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))
    
    # Relationships
    organization = relationship("Organization", back_populates="patients")
    assigned_doctor = relationship("User", back_populates="assigned_patients")
    assessments = relationship("Assessment", back_populates="patient", cascade="all, delete-orphan")
    uploaded_files = relationship("UploadedFile", back_populates="patient")

    def __repr__(self):
        return f"<Patient(id={self.id}, name={self.full_name})>"
