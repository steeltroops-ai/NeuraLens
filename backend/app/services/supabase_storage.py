"""
Supabase Storage Service for NeuraLens
Handles file upload, retrieval, and management for audio and image files
"""

import os
import uuid
import mimetypes
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime, timedelta
import logging
from pathlib import Path

from app.core.supabase_config import supabase_client, supabase_settings

logger = logging.getLogger(__name__)


class SupabaseStorageService:
    """Service for managing file storage in Supabase"""
    
    def __init__(self):
        self.client = supabase_client.service_client
        self.settings = supabase_settings
    
    async def upload_audio_file(
        self,
        file_data: bytes,
        filename: str,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload audio file for speech analysis
        
        Args:
            file_data: Audio file bytes
            filename: Original filename
            session_id: Assessment session ID
            user_id: User ID (optional)
            metadata: Additional metadata
            
        Returns:
            Dict with file information and storage details
        """
        try:
            # Generate unique file path
            file_extension = Path(filename).suffix.lower()
            unique_filename = f"{session_id}_{uuid.uuid4()}{file_extension}"
            file_path = f"speech/{datetime.now().strftime('%Y/%m/%d')}/{unique_filename}"
            
            # Validate file type
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type or not mime_type.startswith('audio/'):
                raise ValueError(f"Invalid audio file type: {mime_type}")
            
            # Validate file size
            if len(file_data) > self.settings.MAX_AUDIO_SIZE:
                raise ValueError(f"File too large: {len(file_data)} bytes (max: {self.settings.MAX_AUDIO_SIZE})")
            
            # Upload to Supabase Storage
            result = self.client.storage.from_(self.settings.SUPABASE_STORAGE_BUCKET_AUDIO).upload(
                path=file_path,
                file=file_data,
                file_options={
                    "content-type": mime_type,
                    "cache-control": "3600",
                    "upsert": False
                }
            )
            
            if result.get("error"):
                raise Exception(f"Upload failed: {result['error']}")
            
            # Generate file info
            file_info = {
                "file_id": str(uuid.uuid4()),
                "original_filename": filename,
                "storage_path": file_path,
                "bucket": self.settings.SUPABASE_STORAGE_BUCKET_AUDIO,
                "mime_type": mime_type,
                "file_size": len(file_data),
                "session_id": session_id,
                "user_id": user_id,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            logger.info(f"Audio file uploaded successfully: {file_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Audio file upload failed: {str(e)}")
            raise
    
    async def upload_image_file(
        self,
        file_data: bytes,
        filename: str,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload image file for retinal analysis
        
        Args:
            file_data: Image file bytes
            filename: Original filename
            session_id: Assessment session ID
            user_id: User ID (optional)
            metadata: Additional metadata
            
        Returns:
            Dict with file information and storage details
        """
        try:
            # Generate unique file path
            file_extension = Path(filename).suffix.lower()
            unique_filename = f"{session_id}_{uuid.uuid4()}{file_extension}"
            file_path = f"retinal/{datetime.now().strftime('%Y/%m/%d')}/{unique_filename}"
            
            # Validate file type
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type or not mime_type.startswith('image/'):
                raise ValueError(f"Invalid image file type: {mime_type}")
            
            # Validate file size
            if len(file_data) > self.settings.MAX_IMAGE_SIZE:
                raise ValueError(f"File too large: {len(file_data)} bytes (max: {self.settings.MAX_IMAGE_SIZE})")
            
            # Upload to Supabase Storage
            result = self.client.storage.from_(self.settings.SUPABASE_STORAGE_BUCKET_IMAGES).upload(
                path=file_path,
                file=file_data,
                file_options={
                    "content-type": mime_type,
                    "cache-control": "3600",
                    "upsert": False
                }
            )
            
            if result.get("error"):
                raise Exception(f"Upload failed: {result['error']}")
            
            # Generate file info
            file_info = {
                "file_id": str(uuid.uuid4()),
                "original_filename": filename,
                "storage_path": file_path,
                "bucket": self.settings.SUPABASE_STORAGE_BUCKET_IMAGES,
                "mime_type": mime_type,
                "file_size": len(file_data),
                "session_id": session_id,
                "user_id": user_id,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            logger.info(f"Image file uploaded successfully: {file_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Image file upload failed: {str(e)}")
            raise
    
    async def get_file_url(
        self,
        bucket: str,
        file_path: str,
        expires_in: int = 3600
    ) -> str:
        """
        Generate signed URL for file access
        
        Args:
            bucket: Storage bucket name
            file_path: File path in bucket
            expires_in: URL expiration time in seconds
            
        Returns:
            Signed URL for file access
        """
        try:
            result = self.client.storage.from_(bucket).create_signed_url(
                path=file_path,
                expires_in=expires_in
            )
            
            if result.get("error"):
                raise Exception(f"URL generation failed: {result['error']}")
            
            return result["signedURL"]
            
        except Exception as e:
            logger.error(f"File URL generation failed: {str(e)}")
            raise
    
    async def delete_file(
        self,
        bucket: str,
        file_path: str
    ) -> bool:
        """
        Delete file from storage
        
        Args:
            bucket: Storage bucket name
            file_path: File path in bucket
            
        Returns:
            True if deletion successful
        """
        try:
            result = self.client.storage.from_(bucket).remove([file_path])
            
            if result.get("error"):
                raise Exception(f"File deletion failed: {result['error']}")
            
            logger.info(f"File deleted successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"File deletion failed: {str(e)}")
            return False
    
    async def cleanup_old_files(
        self,
        bucket: str,
        older_than_days: int = 30
    ) -> int:
        """
        Clean up old files from storage
        
        Args:
            bucket: Storage bucket name
            older_than_days: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        try:
            # List all files in bucket
            result = self.client.storage.from_(bucket).list()
            
            if result.get("error"):
                raise Exception(f"File listing failed: {result['error']}")
            
            # Filter old files
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            old_files = []
            
            for file_info in result:
                if file_info.get("updated_at"):
                    file_date = datetime.fromisoformat(file_info["updated_at"].replace("Z", "+00:00"))
                    if file_date < cutoff_date:
                        old_files.append(file_info["name"])
            
            # Delete old files
            if old_files:
                delete_result = self.client.storage.from_(bucket).remove(old_files)
                if delete_result.get("error"):
                    raise Exception(f"Bulk deletion failed: {delete_result['error']}")
            
            logger.info(f"Cleaned up {len(old_files)} old files from {bucket}")
            return len(old_files)
            
        except Exception as e:
            logger.error(f"File cleanup failed: {str(e)}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics
        
        Returns:
            Dict with storage statistics
        """
        try:
            stats = {}
            
            for bucket_name in [
                self.settings.SUPABASE_STORAGE_BUCKET_AUDIO,
                self.settings.SUPABASE_STORAGE_BUCKET_IMAGES,
                self.settings.SUPABASE_STORAGE_BUCKET_REPORTS
            ]:
                try:
                    files = self.client.storage.from_(bucket_name).list()
                    if not files.get("error"):
                        total_size = sum(file_info.get("metadata", {}).get("size", 0) for file_info in files)
                        stats[bucket_name] = {
                            "file_count": len(files),
                            "total_size_bytes": total_size,
                            "total_size_mb": round(total_size / (1024 * 1024), 2)
                        }
                    else:
                        stats[bucket_name] = {"error": files["error"]}
                except Exception as e:
                    stats[bucket_name] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Storage stats retrieval failed: {str(e)}")
            return {"error": str(e)}


# Global storage service instance
storage_service = SupabaseStorageService()
