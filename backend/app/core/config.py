"""
Simplified configuration settings for NeuroLens-X Backend
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Minimal application settings"""
    
    # Application
    APP_NAME: str = "NeuroLens-X API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://neurolens-x.vercel.app"
    ]
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./neurolens.db"
    DATABASE_ECHO: bool = False
    
    # Security
    SECRET_KEY: str = "neurolens-x-hackathon-secret-key-change-in-production"
    
    # ML Model Configuration
    MODEL_PATH: str = "./models"
    ENABLE_GPU: bool = False  # Set to True if GPU available
    MAX_BATCH_SIZE: int = 8
    MODEL_CACHE_SIZE: int = 100  # MB
    
    # File Upload Limits
    MAX_AUDIO_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024   # 5MB
    MAX_VIDEO_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Processing Timeouts
    SPEECH_PROCESSING_TIMEOUT: int = 30  # seconds
    RETINAL_PROCESSING_TIMEOUT: int = 20  # seconds
    MOTOR_PROCESSING_TIMEOUT: int = 45   # seconds
    NRI_FUSION_TIMEOUT: int = 10         # seconds
    
    # Clinical Validation Settings
    ENABLE_VALIDATION_LOGGING: bool = True
    VALIDATION_DATA_PATH: str = "./data/validation"
    SYNTHETIC_DATA_PATH: str = "./data/samples"
    
    # Performance Monitoring
    ENABLE_METRICS: bool = True
    METRICS_ENDPOINT: str = "/metrics"
    LOG_LEVEL: str = "INFO"
    
    # Security
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    ENABLE_API_KEY_AUTH: bool = False  # Disabled for hackathon demo
    
    # External Services (for production scaling)
    REDIS_URL: Optional[str] = None
    SENTRY_DSN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


# Environment-specific overrides
if settings.ENVIRONMENT == "production":
    settings.DEBUG = False
    settings.DATABASE_ECHO = False
    settings.LOG_LEVEL = "WARNING"
    settings.ENABLE_API_KEY_AUTH = True
elif settings.ENVIRONMENT == "testing":
    settings.DATABASE_URL = "sqlite:///./test_neurolens_x.db"
    settings.ENABLE_VALIDATION_LOGGING = False


# Validation
def validate_settings():
    """Validate critical settings"""
    if settings.ENVIRONMENT == "production":
        if settings.SECRET_KEY == "neurolens-x-hackathon-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be changed in production!")
        
        if not settings.ALLOWED_ORIGINS:
            raise ValueError("ALLOWED_ORIGINS must be configured in production!")
    
    # Ensure model path exists
    os.makedirs(settings.MODEL_PATH, exist_ok=True)
    os.makedirs(settings.VALIDATION_DATA_PATH, exist_ok=True)
    os.makedirs(settings.SYNTHETIC_DATA_PATH, exist_ok=True)


# Run validation on import
validate_settings()
