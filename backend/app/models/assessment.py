"""MediLens Database Models"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, LargeBinary, Text
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


class AudioRecording(Base):
    """Store recorded audio files for data records"""
    __tablename__ = "audio_recordings"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), index=True)
    filename = Column(String(255))
    content_type = Column(String(100))
    file_size = Column(Integer)
    duration_seconds = Column(Float)
    sample_rate = Column(Integer, default=16000)
    audio_data = Column(LargeBinary)  # Store raw audio bytes
    wav_data = Column(LargeBinary, nullable=True)  # Store converted WAV for processing
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Analysis metadata
    analysis_session_id = Column(String(255), nullable=True)
    risk_score = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

