"""
Supabase Configuration for NeuraLens
PostgreSQL database and Storage integration
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)


class SupabaseSettings(BaseSettings):
    """Supabase-specific configuration"""
    
    # Supabase Project Configuration
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL")
    SUPABASE_ANON_KEY: str = Field(..., env="SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    
    # Database Configuration
    SUPABASE_DB_HOST: str = Field(..., env="SUPABASE_DB_HOST")
    SUPABASE_DB_NAME: str = Field(default="postgres", env="SUPABASE_DB_NAME")
    SUPABASE_DB_USER: str = Field(..., env="SUPABASE_DB_USER")
    SUPABASE_DB_PASSWORD: str = Field(..., env="SUPABASE_DB_PASSWORD")
    SUPABASE_DB_PORT: int = Field(default=5432, env="SUPABASE_DB_PORT")
    
    # Storage Configuration
    SUPABASE_STORAGE_BUCKET_AUDIO: str = Field(default="neuralens-audio", env="SUPABASE_STORAGE_BUCKET_AUDIO")
    SUPABASE_STORAGE_BUCKET_IMAGES: str = Field(default="neuralens-images", env="SUPABASE_STORAGE_BUCKET_IMAGES")
    SUPABASE_STORAGE_BUCKET_REPORTS: str = Field(default="neuralens-reports", env="SUPABASE_STORAGE_BUCKET_REPORTS")
    
    # File Upload Configuration
    MAX_AUDIO_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_AUDIO_SIZE")  # 10MB
    MAX_IMAGE_SIZE: int = Field(default=5 * 1024 * 1024, env="MAX_IMAGE_SIZE")   # 5MB
    MAX_REPORT_SIZE: int = Field(default=2 * 1024 * 1024, env="MAX_REPORT_SIZE") # 2MB
    
    # Security Configuration
    ENABLE_RLS: bool = Field(default=True, env="ENABLE_RLS")
    FILE_UPLOAD_SECURITY: bool = Field(default=True, env="FILE_UPLOAD_SECURITY")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection string for SQLAlchemy"""
        return (
            f"postgresql://{self.SUPABASE_DB_USER}:{self.SUPABASE_DB_PASSWORD}"
            f"@{self.SUPABASE_DB_HOST}:{self.SUPABASE_DB_PORT}/{self.SUPABASE_DB_NAME}"
        )


class SupabaseClient:
    """Supabase client wrapper for NeuraLens"""
    
    def __init__(self, settings: SupabaseSettings):
        self.settings = settings
        self._client: Optional[Client] = None
        self._service_client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Get Supabase client with anon key (for frontend operations)"""
        if not self._client:
            self._client = create_client(
                self.settings.SUPABASE_URL,
                self.settings.SUPABASE_ANON_KEY
            )
        return self._client
    
    @property
    def service_client(self) -> Client:
        """Get Supabase client with service role key (for backend operations)"""
        if not self._service_client:
            self._service_client = create_client(
                self.settings.SUPABASE_URL,
                self.settings.SUPABASE_SERVICE_ROLE_KEY
            )
        return self._service_client
    
    async def initialize_storage(self):
        """Initialize storage buckets for NeuraLens"""
        try:
            # Create storage buckets if they don't exist
            buckets_to_create = [
                {
                    "name": self.settings.SUPABASE_STORAGE_BUCKET_AUDIO,
                    "public": False,
                    "file_size_limit": self.settings.MAX_AUDIO_SIZE,
                    "allowed_mime_types": ["audio/wav", "audio/mp3", "audio/m4a", "audio/webm"]
                },
                {
                    "name": self.settings.SUPABASE_STORAGE_BUCKET_IMAGES,
                    "public": False,
                    "file_size_limit": self.settings.MAX_IMAGE_SIZE,
                    "allowed_mime_types": ["image/jpeg", "image/png", "image/jpg"]
                },
                {
                    "name": self.settings.SUPABASE_STORAGE_BUCKET_REPORTS,
                    "public": False,
                    "file_size_limit": self.settings.MAX_REPORT_SIZE,
                    "allowed_mime_types": ["application/pdf", "text/plain", "application/json"]
                }
            ]
            
            for bucket_config in buckets_to_create:
                try:
                    # Try to create bucket
                    result = self.service_client.storage.create_bucket(
                        bucket_config["name"],
                        options={
                            "public": bucket_config["public"],
                            "file_size_limit": bucket_config["file_size_limit"],
                            "allowed_mime_types": bucket_config["allowed_mime_types"]
                        }
                    )
                    logger.info(f"Created storage bucket: {bucket_config['name']}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Storage bucket already exists: {bucket_config['name']}")
                    else:
                        logger.error(f"Failed to create bucket {bucket_config['name']}: {e}")
            
            logger.info("✅ Supabase storage initialization completed")
            
        except Exception as e:
            logger.error(f"❌ Supabase storage initialization failed: {e}")
            raise
    
    async def health_check(self) -> dict:
        """Check Supabase connection health"""
        try:
            # Test database connection
            result = self.service_client.table("_health_check").select("*").limit(1).execute()
            
            # Test storage connection
            buckets = self.service_client.storage.list_buckets()
            
            return {
                "status": "healthy",
                "database": "connected",
                "storage": "connected",
                "buckets": len(buckets) if buckets else 0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global instances
supabase_settings = SupabaseSettings()
supabase_client = SupabaseClient(supabase_settings)
