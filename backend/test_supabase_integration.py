#!/usr/bin/env python3
"""
NeuraLens Supabase Integration Test
Tests database connection, storage functionality, and API endpoints
"""

import asyncio
import os
import sys
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.core.database import engine, SessionLocal
from app.core.supabase_config import supabase_client
from app.services.supabase_storage import storage_service
from app.services.user_service import UserService
from app.services.assessment_service import AssessmentService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseIntegrationTest:
    """Test Supabase integration functionality"""
    
    def __init__(self):
        self.user_service = UserService()
        self.assessment_service = AssessmentService()
        self.test_user_id = None
        self.test_assessment_id = None
    
    async def test_database_connection(self):
        """Test PostgreSQL database connection"""
        logger.info("🔍 Testing database connection...")
        
        try:
            db = SessionLocal()
            try:
                from sqlalchemy import text
                result = db.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"✅ Database connected: {version}")
                return True
            finally:
                db.close()
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def test_supabase_client(self):
        """Test Supabase client connection"""
        logger.info("🔍 Testing Supabase client...")
        
        try:
            health = await supabase_client.health_check()
            if health["status"] == "healthy":
                logger.info("✅ Supabase client connected")
                return True
            else:
                logger.error(f"❌ Supabase client unhealthy: {health}")
                return False
        except Exception as e:
            logger.error(f"❌ Supabase client test failed: {e}")
            return False
    
    async def test_storage_buckets(self):
        """Test storage bucket functionality"""
        logger.info("🔍 Testing storage buckets...")
        
        try:
            stats = await storage_service.get_storage_stats()
            logger.info("✅ Storage buckets accessible:")
            for bucket, info in stats.items():
                if "error" not in info:
                    logger.info(f"   - {bucket}: {info.get('file_count', 0)} files")
                else:
                    logger.warning(f"   - {bucket}: {info['error']}")
            return True
        except Exception as e:
            logger.error(f"❌ Storage bucket test failed: {e}")
            return False
    
    async def test_user_operations(self):
        """Test user CRUD operations"""
        logger.info("🔍 Testing user operations...")
        
        try:
            db = SessionLocal()
            try:
                # Create test user
                timestamp = int(datetime.now().timestamp())
                user = self.user_service.create_user(
                    db,
                    email=f"test_{timestamp}@neuralens.com",
                    username=f"test_user_{timestamp}",
                    age=45,
                    sex="female",
                    education_years=16,
                    consent_given=True
                )
                self.test_user_id = user.id
                logger.info(f"✅ User created: {user.email}")
                
                # Read user
                retrieved_user = self.user_service.get_user_by_id(db, user.id)
                if retrieved_user and retrieved_user.email == user.email:
                    logger.info("✅ User retrieval successful")
                else:
                    raise Exception("User retrieval failed")
                
                return True
            finally:
                db.close()
        except Exception as e:
            logger.error(f"❌ User operations test failed: {e}")
            return False
    
    async def test_assessment_operations(self):
        """Test assessment CRUD operations"""
        logger.info("🔍 Testing assessment operations...")
        
        if not self.test_user_id:
            logger.error("❌ No test user available for assessment test")
            return False
        
        try:
            db = SessionLocal()
            try:
                # Create test assessment
                assessment = self.assessment_service.create_assessment(
                    db,
                    user_id=self.test_user_id,
                    modalities=["speech", "retinal"],
                    assessment_type="test"
                )
                self.test_assessment_id = assessment.id
                logger.info(f"✅ Assessment created: {assessment.session_id}")
                
                # Read assessment
                retrieved_assessment = self.assessment_service.get_assessment_by_id(db, assessment.id)
                if retrieved_assessment and retrieved_assessment.session_id == assessment.session_id:
                    logger.info("✅ Assessment retrieval successful")
                else:
                    raise Exception("Assessment retrieval failed")
                
                return True
            finally:
                db.close()
        except Exception as e:
            logger.error(f"❌ Assessment operations test failed: {e}")
            return False
    
    async def test_file_upload(self):
        """Test file upload functionality"""
        logger.info("🔍 Testing file upload...")
        
        try:
            # Create test image
            test_image = Image.new('RGB', (100, 100), color='red')
            image_bytes = io.BytesIO()
            test_image.save(image_bytes, format='JPEG')
            image_data = image_bytes.getvalue()
            
            # Upload test image
            session_id = f"test_session_{int(datetime.now().timestamp())}"
            file_info = await storage_service.upload_image_file(
                file_data=image_data,
                filename="test_retinal.jpg",
                session_id=session_id,
                user_id=self.test_user_id,
                metadata={"test": True}
            )
            
            logger.info(f"✅ Image uploaded: {file_info['storage_path']}")
            
            # Generate signed URL
            signed_url = await storage_service.get_file_url(
                bucket=file_info['bucket'],
                file_path=file_info['storage_path'],
                expires_in=300
            )
            
            if signed_url:
                logger.info("✅ Signed URL generated successfully")
            else:
                raise Exception("Signed URL generation failed")
            
            # Clean up test file
            deleted = await storage_service.delete_file(
                bucket=file_info['bucket'],
                file_path=file_info['storage_path']
            )
            
            if deleted:
                logger.info("✅ Test file cleaned up")
            else:
                logger.warning("⚠️ Test file cleanup failed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ File upload test failed: {e}")
            return False
    
    async def test_audio_upload(self):
        """Test audio file upload functionality"""
        logger.info("🔍 Testing audio upload...")
        
        try:
            # Create dummy audio data (WAV header + silence)
            wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
            silence_data = b'\x00' * 2048  # 2KB of silence
            audio_data = wav_header + silence_data
            
            # Upload test audio
            session_id = f"test_audio_session_{int(datetime.now().timestamp())}"
            file_info = await storage_service.upload_audio_file(
                file_data=audio_data,
                filename="test_speech.wav",
                session_id=session_id,
                user_id=self.test_user_id,
                metadata={"test": True, "duration": 1.0}
            )
            
            logger.info(f"✅ Audio uploaded: {file_info['storage_path']}")
            
            # Clean up test file
            deleted = await storage_service.delete_file(
                bucket=file_info['bucket'],
                file_path=file_info['storage_path']
            )
            
            if deleted:
                logger.info("✅ Test audio file cleaned up")
            else:
                logger.warning("⚠️ Test audio file cleanup failed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio upload test failed: {e}")
            return False
    
    async def cleanup_test_data(self):
        """Clean up test data"""
        logger.info("🧹 Cleaning up test data...")
        
        try:
            db = SessionLocal()
            try:
                # Delete test assessment
                if self.test_assessment_id:
                    assessment = self.assessment_service.get_assessment_by_id(db, self.test_assessment_id)
                    if assessment:
                        db.delete(assessment)
                        logger.info("✅ Test assessment deleted")
                
                # Delete test user
                if self.test_user_id:
                    user = self.user_service.get_user_by_id(db, self.test_user_id)
                    if user:
                        db.delete(user)
                        logger.info("✅ Test user deleted")
                
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"⚠️ Test data cleanup failed: {e}")
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting Supabase integration tests...")
        start_time = datetime.now()
        
        tests = [
            ("Database Connection", self.test_database_connection),
            ("Supabase Client", self.test_supabase_client),
            ("Storage Buckets", self.test_storage_buckets),
            ("User Operations", self.test_user_operations),
            ("Assessment Operations", self.test_assessment_operations),
            ("File Upload", self.test_file_upload),
            ("Audio Upload", self.test_audio_upload),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"❌ {test_name} test crashed: {e}")
                results[test_name] = False
        
        # Cleanup
        await self.cleanup_test_data()
        
        # Report results
        duration = datetime.now() - start_time
        logger.info("\n" + "="*60)
        logger.info("📊 TEST RESULTS:")
        logger.info("="*60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} {test_name}")
            if result:
                passed += 1
        
        logger.info("="*60)
        logger.info(f"📈 Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        logger.info(f"⏱️ Duration: {duration.total_seconds():.2f} seconds")
        
        if passed == total:
            logger.info("🎉 All tests passed! Supabase integration is working correctly.")
        else:
            logger.error(f"💥 {total-passed} tests failed. Please check the configuration.")
            sys.exit(1)


async def main():
    """Main test function"""
    test_runner = SupabaseIntegrationTest()
    await test_runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
