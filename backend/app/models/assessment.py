"""MediLens Database Models"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
import uuid
from app.database import Base


class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True, default=lambda: str(uuid.uuid4()))
    pipeline = Column(String(50))
    risk_score = Column(Float)
    confidence = Column(Float)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
