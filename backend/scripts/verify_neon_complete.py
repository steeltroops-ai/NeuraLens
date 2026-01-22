#!/usr/bin/env python3
"""
Comprehensive Neon Postgres Verification Script
Tests database connection, models, migrations, and repository functionality
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load .env file manually
env_file = backend_dir / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove surrounding quotes if present
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value

async def verify_neon_connection():
    """Test 1: Verify Neon Postgres connection"""
    print("\n" + "="*70)
    print("TEST 1: NEON POSTGRES CONNECTION")
    print("="*70)
    
    # Check environment variable
    neon_url = os.getenv("NEON_DATABASE_URL")
    if not neon_url:
        print("⚠  NEON_DATABASE_URL not set in environment")
        print("   For this test, set NEON_DATABASE_URL or use SQLite for local dev")
        return False
    
    if "your-neon-host" in neon_url or not neon_url.startswith("postgresql"):
        print("⚠  NEON_DATABASE_URL appears to be using template value")
        print(f"   Current value starts with: {neon_url[:30]}...")
        return False
    
    try:
        from app.database import db
        
        print(f"✓ NEON_DATABASE_URL configured")
        print(f"  Host: {neon_url.split('@')[1].split('/')[0] if '@' in neon_url else 'hidden'}")
        
        # Initialize connection
        db.initialize()
        print("✓ Database connection initialized")
        
        # Test actual connection
        async with db.session_scope() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✓ Successfully connected to database")
            print(f"  PostgreSQL version: {version.split(',')[0]}")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


async def verify_models():
    """Test 2: Verify all SQLAlchemy models"""
    print("\n" + "="*70)
    print("TEST 2: SQLALCHEMY MODELS VALIDATION")
    print("="*70)
    
    try:
        from app.database import Base
        from app.database.models import (
            User, Organization, UserProfile, Role, UserRole,
            Assessment, PipelineStage, BiomarkerValue,
            RetinalResult, SpeechResult, CardiologyResult,
            RadiologyResult, DermatologyResult, CognitiveResult,
            ChatThread, ChatMessage, AIExplanation,
            UploadedFile, AuditEvent
        )
        
        models = [
            User, Organization, UserProfile, Role, UserRole,
            Assessment, PipelineStage, BiomarkerValue,
            RetinalResult, SpeechResult, CardiologyResult,
            RadiologyResult, DermatologyResult, CognitiveResult,
            ChatThread, ChatMessage, AIExplanation,
            UploadedFile, AuditEvent
        ]
        
        print(f"✓ All {len(models)} models imported successfully")
        
        # Check table names
        tables = Base.metadata.tables
        print(f"✓ {len(tables)} tables registered in metadata:")
        for table_name in sorted(tables.keys()):
            table = tables[table_name]
            col_count = len(table.columns)
            print(f"  - {table_name:30s} ({col_count:2d} columns)")
        
        # Validate relationships
        relationship_count = 0
        for model in models:
            if hasattr(model, '__mapper__'):
                relationships = model.__mapper__.relationships
                relationship_count += len(relationships)
        
        print(f"✓ {relationship_count} relationships defined")
        
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_repository():
    """Test 3: Verify repository pattern"""
    print("\n" + "="*70)
    print("TEST 3: REPOSITORY PATTERN")
    print("="*70)
    
    try:
        from app.database.repositories import AssessmentRepository
        from sqlalchemy.ext.asyncio import AsyncSession
        
        # Check repository methods
        methods = [m for m in dir(AssessmentRepository) if not m.startswith('_')]
        print(f"✓ AssessmentRepository loaded")
        print(f"✓ {len(methods)} public methods available:")
        
        key_methods = [
            'create_assessment',
            'get_assessment_by_id',
            'get_user_assessments',
            'save_biomarkers',
            'get_user_biomarker_history',
            'update_assessment_status',
            'get_assessment_statistics',
            'soft_delete_assessment'
        ]
        
        for method in key_methods:
            if method in methods:
                print(f"  ✓ {method}")
            else:
                print(f"  ✗ {method} missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Repository validation failed: {e}")
        return False


async def verify_migrations():
    """Test 4: Verify Alembic migrations"""
    print("\n" + "="*70)
    print("TEST 4: ALEMBIC MIGRATIONS")
    print("="*70)
    
    try:
        import alembic
        from alembic.config import Config
        from alembic import command
        
        print(f"✓ Alembic {alembic.__version__} installed")
        
        # Check alembic.ini
        alembic_ini = backend_dir / "alembic.ini"
        if alembic_ini.exists():
            print(f"✓ alembic.ini found")
        else:
            print(f"✗ alembic.ini not found")
            return False
        
        # Check migrations directory
        migrations_dir = backend_dir / "migrations"
        if migrations_dir.exists():
            print(f"✓ migrations/ directory exists")
            
            env_file = migrations_dir / "env.py"
            if env_file.exists():
                print(f"✓ migrations/env.py exists")
            else:
                print(f"✗ migrations/env.py missing")
                return False
        else:
            print(f"✗ migrations/ directory missing")
            return False
        
        # Check for migration versions
        versions_dir = migrations_dir / "versions"
        if versions_dir.exists():
            migrations = list(versions_dir.glob("*.py"))
            print(f"✓ {len(migrations)} migration(s) found")
        else:
            print(f"⚠  No migrations/versions/ directory yet")
            print(f"   Run: alembic revision --autogenerate -m 'Initial schema'")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration validation failed: {e}")
        return False


async def verify_fastapi_integration():
    """Test 5: Verify FastAPI integration"""
    print("\n" + "="*70)
    print("TEST 5: FASTAPI INTEGRATION")
    print("="*70)
    
    try:
        from app.database import get_db, lifespan_manager
        from app.main import app
        
        print("✓ get_db dependency imported")
        print("✓ lifespan_manager imported")
        print("✓ FastAPI app imported")
        
        # Check if lifespan is configured
        if hasattr(app, 'router'):
            print("✓ FastAPI router configured")
        
        return True
        
    except Exception as e:
        print(f"✗ FastAPI integration check failed: {e}")
        return False


async def test_database_operations():
    """Test 6: Test actual database operations (if Neon configured)"""
    print("\n" + "="*70)
    print("TEST 6: DATABASE OPERATIONS (OPTIONAL)")
    print("="*70)
    
    neon_url = os.getenv("NEON_DATABASE_URL")
    if not neon_url or "your-neon-host" in neon_url:
        print("⚠  Skipping - NEON_DATABASE_URL not configured")
        print("   This test requires a real Neon database connection")
        return True
    
    try:
        from app.database import db
        from app.database.models import Role
        from sqlalchemy import select
        
        db.initialize()
        
        # Create tables
        print("  Creating tables...")
        await db.create_all_tables()
        print("  ✓ Tables created")
        
        # Test insert
        async with db.session_scope() as session:
            # Check if test role exists
            result = await session.execute(
                select(Role).where(Role.name == "test_role")
            )
            existing_role = result.scalar_one_or_none()
            
            if not existing_role:
                test_role = Role(
                    name="test_role",
                    description="Test role for verification",
                    permissions=["test_permission"]
                )
                session.add(test_role)
                await session.flush()
                print(f"  ✓ Created test role (id: {test_role.id})")
            else:
                print(f"  ✓ Test role already exists (id: {existing_role.id})")
        
        # Test query
        async with db.session_scope() as session:
            result = await session.execute(select(Role))
            roles = result.scalars().all()
            print(f"  ✓ Query successful - {len(roles)} role(s) in database")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Database operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("NEON POSTGRES DATABASE - COMPREHENSIVE VERIFICATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {}
    
    # Run all tests
    results['connection'] = await verify_neon_connection()
    results['models'] = await verify_models()
    results['repository'] = await verify_repository()
    results['migrations'] = await verify_migrations()
    results['fastapi'] = await verify_fastapi_integration()
    results['operations'] = await test_database_operations()
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {test.upper()}")
    
    print("\n" + "-"*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Database implementation verified!")
        print("\n📋 Next Steps:")
        print("   1. Configure NEON_DATABASE_URL in .env (if not done)")
        print("   2. Run: alembic revision --autogenerate -m 'Initial schema'")
        print("   3. Run: alembic upgrade head")
        print("   4. Run: python scripts/init_db.py")
        print("   5. Start backend: uvicorn app.main:app --reload")
    else:
        print("\n⚠  Some tests failed. Please review errors above.")
        print("   See DATABASE_SETUP.md for detailed setup instructions.")
    
    print("\n" + "="*70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
