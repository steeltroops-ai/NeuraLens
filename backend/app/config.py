"""
MediLens Backend Configuration
Simple Pydantic settings
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "MediLens API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./medilens.db"
    
    # CORS
    ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # API Keys
    ELEVENLABS_API_KEY: str = ""
    
    # Audio
    MAX_AUDIO_SIZE: int = 10 * 1024 * 1024
    AUDIO_TIMEOUT: float = 30.0
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
