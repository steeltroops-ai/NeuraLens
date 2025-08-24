"""
Configuration settings for NeuroLens-X Backend
Hackathon-optimized with environment-based configuration
"""

from pydantic_settings import BaseSettings, Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "NeuroLens-X API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "neurolens-x-hackathon-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "https://neurolens-x.vercel.app",
        "https://neurolens-x-git-main.vercel.app",
        "https://neurolens-x-steeltroops-ai.vercel.app"
    ]
    
    # Database Configuration (PostgreSQL via Supabase)
    DATABASE_URL: str = Field(
        default="sqlite:///./neurolens_x.db",  # Fallback for development
        env="DATABASE_URL"
    )
    DATABASE_ECHO: bool = False  # Set to True for SQL debugging

    # Supabase Configuration
    SUPABASE_URL: str = Field(default="", env="SUPABASE_URL")
    SUPABASE_ANON_KEY: str = Field(default="", env="SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(default="", env="SUPABASE_SERVICE_ROLE_KEY")

    # Supabase Database Details
    SUPABASE_DB_HOST: str = Field(default="", env="SUPABASE_DB_HOST")
    SUPABASE_DB_USER: str = Field(default="postgres", env="SUPABASE_DB_USER")
    SUPABASE_DB_PASSWORD: str = Field(default="", env="SUPABASE_DB_PASSWORD")
    SUPABASE_DB_NAME: str = Field(default="postgres", env="SUPABASE_DB_NAME")
    SUPABASE_DB_PORT: int = Field(default=5432, env="SUPABASE_DB_PORT")

    # Use Supabase if configured, otherwise fallback to SQLite
    @property
    def effective_database_url(self) -> str:
        """Get the effective database URL based on configuration"""
        if self.SUPABASE_DB_HOST and self.SUPABASE_DB_PASSWORD:
            return (
                f"postgresql://{self.SUPABASE_DB_USER}:{self.SUPABASE_DB_PASSWORD}"
                f"@{self.SUPABASE_DB_HOST}:{self.SUPABASE_DB_PORT}/{self.SUPABASE_DB_NAME}"
            )
        return self.DATABASE_URL

    @property
    def is_using_supabase(self) -> bool:
        """Check if we're using Supabase PostgreSQL"""
        return bool(self.SUPABASE_DB_HOST and self.SUPABASE_DB_PASSWORD)
    
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
