"""
SQLAlchemy Models - Patient Management
"""

from sqlalchemy import (
    Column, String, Integer, Text, DateTime,
    func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.database import Base

class Patient(Base):
    """Patient entity for medical records"""
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Identity
    full_name = Column(String(255), nullable=False, index=True)
    phone_number = Column(String(50), unique=True, nullable=False, index=True)
    
    # Demographics
    age = Column(Integer)
    gender = Column(String(20))
    
    # Medical context
    medical_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    assessments = relationship("Assessment", back_populates="patient", cascade="all, delete-orphan")
    uploaded_files = relationship("UploadedFile", back_populates="patient")

    def __repr__(self):
        return f"<Patient(id={self.id}, name={self.full_name}, phone={self.phone_number})>"
