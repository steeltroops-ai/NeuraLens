#!/usr/bin/env python3
"""
Complete NeuraLens Supabase Setup Script
Creates database schema, RLS policies, storage buckets, and tests integration
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import json

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.core.supabase_config import supabase_client, supabase_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseCompleteSetup:
    """Complete Supabase setup for NeuraLens"""
    
    def __init__(self):
        self.project_id = "juyebmhkqjjnttvelvpp"
        self.project_url = f"https://{self.project_id}.supabase.co"
        self.db_host = f"db.{self.project_id}.supabase.co"
    
    async def check_project_status(self):
        """Check if Supabase project is active and ready"""
        logger.info("🔍 Checking Supabase project status...")
        
        try:
            # Test basic connection
            health = await supabase_client.health_check()
            if health["status"] == "healthy":
                logger.info("✅ Supabase project is active and healthy")
                return True
            else:
                logger.error(f"❌ Supabase project health check failed: {health}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to Supabase project: {e}")
            logger.info("💡 Please ensure your project is active in the Supabase dashboard")
            return False
    
    async def create_database_schema(self):
        """Create the complete database schema"""
        logger.info("🏗️ Creating database schema...")
        
        try:
            # Read the schema SQL file
            schema_file = Path(__file__).parent / "supabase_schema.sql"
            if not schema_file.exists():
                raise FileNotFoundError("supabase_schema.sql not found")
            
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema creation
            result = supabase_client.service_client.rpc('exec_sql', {'sql': schema_sql})
            
            if result.get('error'):
                raise Exception(f"Schema creation failed: {result['error']}")
            
            logger.info("✅ Database schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database schema creation failed: {e}")
            logger.info("💡 You may need to run the SQL manually in the Supabase SQL editor")
            return False
    
    async def setup_rls_policies(self):
        """Set up Row Level Security policies"""
        logger.info("🔒 Setting up Row Level Security policies...")
        
        try:
            # Read the RLS policies SQL file
            rls_file = Path(__file__).parent / "supabase_rls_policies.sql"
            if not rls_file.exists():
                raise FileNotFoundError("supabase_rls_policies.sql not found")
            
            with open(rls_file, 'r') as f:
                rls_sql = f.read()
            
            # Execute RLS policies
            result = supabase_client.service_client.rpc('exec_sql', {'sql': rls_sql})
            
            if result.get('error'):
                raise Exception(f"RLS setup failed: {result['error']}")
            
            logger.info("✅ Row Level Security policies created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RLS policies setup failed: {e}")
            logger.info("💡 You may need to run the RLS SQL manually in the Supabase SQL editor")
            return False
    
    async def create_storage_buckets(self):
        """Create storage buckets for file uploads"""
        logger.info("🗄️ Creating storage buckets...")
        
        buckets_config = [
            {
                "id": "neuralens-audio",
                "name": "neuralens-audio",
                "public": False,
                "file_size_limit": 10485760,  # 10MB
                "allowed_mime_types": ["audio/wav", "audio/mp3", "audio/m4a", "audio/webm", "audio/ogg"]
            },
            {
                "id": "neuralens-images",
                "name": "neuralens-images", 
                "public": False,
                "file_size_limit": 5242880,  # 5MB
                "allowed_mime_types": ["image/jpeg", "image/png", "image/jpg"]
            },
            {
                "id": "neuralens-reports",
                "name": "neuralens-reports",
                "public": False,
                "file_size_limit": 2097152,  # 2MB
                "allowed_mime_types": ["application/pdf", "text/plain", "application/json"]
            }
        ]
        
        created_buckets = []
        
        for bucket_config in buckets_config:
            try:
                # Try to create bucket
                result = supabase_client.service_client.storage.create_bucket(
                    bucket_config["id"],
                    options={
                        "public": bucket_config["public"],
                        "file_size_limit": bucket_config["file_size_limit"],
                        "allowed_mime_types": bucket_config["allowed_mime_types"]
                    }
                )
                
                if result.get("error"):
                    if "already exists" in str(result["error"]).lower():
                        logger.info(f"✅ Bucket already exists: {bucket_config['id']}")
                    else:
                        logger.error(f"❌ Failed to create bucket {bucket_config['id']}: {result['error']}")
                        continue
                else:
                    logger.info(f"✅ Created bucket: {bucket_config['id']}")
                
                created_buckets.append(bucket_config["id"])
                
            except Exception as e:
                logger.error(f"❌ Error creating bucket {bucket_config['id']}: {e}")
        
        if len(created_buckets) == len(buckets_config):
            logger.info("✅ All storage buckets created successfully")
            return True
        else:
            logger.warning(f"⚠️ Only {len(created_buckets)}/{len(buckets_config)} buckets created")
            return False
    
    async def setup_storage_policies(self):
        """Set up storage bucket policies"""
        logger.info("🔐 Setting up storage policies...")
        
        try:
            # Read the storage policies SQL file
            storage_file = Path(__file__).parent / "supabase_storage_setup.sql"
            if not storage_file.exists():
                raise FileNotFoundError("supabase_storage_setup.sql not found")
            
            with open(storage_file, 'r') as f:
                storage_sql = f.read()
            
            # Execute storage policies
            result = supabase_client.service_client.rpc('exec_sql', {'sql': storage_sql})
            
            if result.get('error'):
                raise Exception(f"Storage policies setup failed: {result['error']}")
            
            logger.info("✅ Storage policies created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Storage policies setup failed: {e}")
            logger.info("💡 You may need to run the storage SQL manually in the Supabase SQL editor")
            return False
    
    async def test_database_operations(self):
        """Test basic database operations"""
        logger.info("🧪 Testing database operations...")
        
        try:
            # Test table creation by inserting a test record
            test_user_data = {
                "email": f"test_{int(datetime.now().timestamp())}@neuralens.com",
                "username": f"test_user_{int(datetime.now().timestamp())}",
                "age": 45,
                "sex": "female",
                "consent_given": True
            }
            
            # Insert test user
            result = supabase_client.service_client.table("users").insert(test_user_data).execute()
            
            if result.data:
                user_id = result.data[0]["id"]
                logger.info(f"✅ Test user created: {user_id}")
                
                # Clean up test user
                supabase_client.service_client.table("users").delete().eq("id", user_id).execute()
                logger.info("✅ Test user cleaned up")
                
                return True
            else:
                logger.error("❌ Failed to create test user")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database operations test failed: {e}")
            return False
    
    async def test_storage_operations(self):
        """Test storage operations"""
        logger.info("🧪 Testing storage operations...")
        
        try:
            # Test file upload
            test_content = b"Test audio content for NeuraLens"
            test_filename = f"test_audio_{int(datetime.now().timestamp())}.wav"
            
            # Upload test file
            result = supabase_client.service_client.storage.from_("neuralens-audio").upload(
                path=f"test/{test_filename}",
                file=test_content,
                file_options={"content-type": "audio/wav"}
            )
            
            if result.get("error"):
                logger.error(f"❌ Storage upload test failed: {result['error']}")
                return False
            
            logger.info("✅ Test file uploaded successfully")
            
            # Clean up test file
            supabase_client.service_client.storage.from_("neuralens-audio").remove([f"test/{test_filename}"])
            logger.info("✅ Test file cleaned up")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Storage operations test failed: {e}")
            return False
    
    async def generate_configuration(self):
        """Generate configuration files for the application"""
        logger.info("⚙️ Generating configuration files...")
        
        try:
            # Create environment configuration
            env_config = f"""# NeuraLens Supabase Configuration
# Generated on {datetime.now().isoformat()}

# Application Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Supabase Project Configuration
SUPABASE_URL={self.project_url}
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# Supabase Database Configuration
SUPABASE_DB_HOST={self.db_host}
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=your_database_password_here
SUPABASE_DB_NAME=postgres
SUPABASE_DB_PORT=5432

# Database URL (automatically constructed)
DATABASE_URL=postgresql://postgres:your_password@{self.db_host}:5432/postgres

# Storage Bucket Names
SUPABASE_STORAGE_BUCKET_AUDIO=neuralens-audio
SUPABASE_STORAGE_BUCKET_IMAGES=neuralens-images
SUPABASE_STORAGE_BUCKET_REPORTS=neuralens-reports

# File Upload Limits (in bytes)
MAX_AUDIO_SIZE=10485760    # 10MB
MAX_IMAGE_SIZE=5242880     # 5MB
MAX_REPORT_SIZE=2097152    # 2MB

# Security Configuration
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=480
ENABLE_RLS=true
FILE_UPLOAD_SECURITY=true

# CORS Configuration
ALLOWED_ORIGINS=["http://localhost:3000","https://neuralens.vercel.app"]
"""
            
            # Write environment file
            env_file = Path(__file__).parent / ".env.production"
            with open(env_file, 'w') as f:
                f.write(env_config)
            
            logger.info(f"✅ Configuration file created: {env_file}")
            
            # Create setup summary
            setup_summary = {
                "project_id": self.project_id,
                "project_url": self.project_url,
                "database_host": self.db_host,
                "buckets_created": ["neuralens-audio", "neuralens-images", "neuralens-reports"],
                "setup_date": datetime.now().isoformat(),
                "next_steps": [
                    "Update .env file with your actual Supabase credentials",
                    "Test the backend with: python -m uvicorn app.main:app --reload",
                    "Run integration tests with: python test_supabase_integration.py",
                    "Generate demo data with: python generate_demo_data.py"
                ]
            }
            
            summary_file = Path(__file__).parent / "supabase_setup_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(setup_summary, f, indent=2)
            
            logger.info(f"✅ Setup summary created: {summary_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration generation failed: {e}")
            return False
    
    async def run_complete_setup(self):
        """Run the complete Supabase setup process"""
        logger.info("🚀 Starting complete NeuraLens Supabase setup...")
        start_time = datetime.now()
        
        setup_steps = [
            ("Project Status Check", self.check_project_status),
            ("Database Schema Creation", self.create_database_schema),
            ("RLS Policies Setup", self.setup_rls_policies),
            ("Storage Buckets Creation", self.create_storage_buckets),
            ("Storage Policies Setup", self.setup_storage_policies),
            ("Database Operations Test", self.test_database_operations),
            ("Storage Operations Test", self.test_storage_operations),
            ("Configuration Generation", self.generate_configuration),
        ]
        
        results = {}
        
        for step_name, step_func in setup_steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 {step_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await step_func()
                results[step_name] = result
                
                if result:
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
                    
            except Exception as e:
                logger.error(f"💥 {step_name} crashed: {e}")
                results[step_name] = False
        
        # Final summary
        duration = datetime.now() - start_time
        logger.info("\n" + "="*80)
        logger.info("📊 SETUP RESULTS SUMMARY")
        logger.info("="*80)
        
        passed = 0
        total = len(results)
        
        for step_name, result in results.items():
            status = "✅ SUCCESS" if result else "❌ FAILED"
            logger.info(f"{status} {step_name}")
            if result:
                passed += 1
        
        logger.info("="*80)
        logger.info(f"📈 Overall: {passed}/{total} steps completed ({passed/total*100:.1f}%)")
        logger.info(f"⏱️ Duration: {duration.total_seconds():.2f} seconds")
        
        if passed == total:
            logger.info("🎉 Complete setup successful! NeuraLens is ready for production.")
            self.print_next_steps()
        else:
            logger.error(f"💥 {total-passed} steps failed. Please check the logs and retry.")
            self.print_troubleshooting()
    
    def print_next_steps(self):
        """Print next steps for the user"""
        logger.info("\n" + "="*80)
        logger.info("🎯 NEXT STEPS:")
        logger.info("="*80)
        logger.info("1. Update .env.production with your actual Supabase credentials:")
        logger.info("   - Get API keys from: https://app.supabase.com/project/juyebmhkqjjnttvelvpp/settings/api")
        logger.info("   - Get database password from: https://app.supabase.com/project/juyebmhkqjjnttvelvpp/settings/database")
        logger.info("2. Copy .env.production to .env and update values")
        logger.info("3. Test the backend: python -m uvicorn app.main:app --reload")
        logger.info("4. Run integration tests: python test_supabase_integration.py")
        logger.info("5. Generate demo data: python generate_demo_data.py")
        logger.info("6. Test frontend connection: cd frontend && bun run dev")
        logger.info("\n📚 Resources:")
        logger.info(f"   - Supabase Dashboard: https://app.supabase.com/project/{self.project_id}")
        logger.info("   - API Documentation: http://localhost:8000/docs")
        logger.info("="*80)
    
    def print_troubleshooting(self):
        """Print troubleshooting information"""
        logger.info("\n" + "="*80)
        logger.info("🔧 TROUBLESHOOTING:")
        logger.info("="*80)
        logger.info("If setup failed, try these steps:")
        logger.info("1. Ensure your Supabase project is active")
        logger.info("2. Check your internet connection")
        logger.info("3. Verify your Supabase credentials")
        logger.info("4. Run SQL files manually in Supabase SQL editor:")
        logger.info("   - supabase_schema.sql")
        logger.info("   - supabase_rls_policies.sql")
        logger.info("   - supabase_storage_setup.sql")
        logger.info("5. Create storage buckets manually in Supabase dashboard")
        logger.info("="*80)


async def main():
    """Main setup function"""
    setup = SupabaseCompleteSetup()
    await setup.run_complete_setup()


if __name__ == "__main__":
    asyncio.run(main())
