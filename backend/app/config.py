"""
MediLens Backend Configuration
Enhanced Pydantic settings with performance tuning
"""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "MediLens API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False  # Default to False for safety
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./medilens.db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 3600  # Recycle connections after 1 hour
    DB_POOL_PRE_PING: bool = True  # Verify connections before use
    
    # CORS
    ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # API Keys
    ELEVENLABS_API_KEY: str = ""
    
    # Audio
    MAX_AUDIO_SIZE: int = 10 * 1024 * 1024
    AUDIO_TIMEOUT: float = 30.0
    
    # Performance Settings
    REQUEST_TIMEOUT: float = 60.0  # Request timeout in seconds
    WORKER_THREADS: int = 4  # ThreadPool for CPU-bound tasks
    
    # Caching
    CACHE_TTL: int = 3600  # Cache TTL in seconds
    CACHE_MAX_SIZE: int = 100  # Max cached items

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def db_echo(self) -> bool:
        """Only enable SQL echo in non-production debug mode."""
        return self.DEBUG and not self.is_production
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()

