#!/usr/bin/env python3
"""
Database Implementation Verification Script
Validates that all Neon database components are correctly installed
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

print("=" * 70)
print("NEON POSTGRES DATABASE - IMPLEMENTATION VERIFICATION")
print("=" * 70)

# Test 1: Import database connection
print("\n[1/6] Testing database connection imports...")
try:
    from app.database import db, get_db, lifespan_manager, Base
    print("✓ Database connection manager imported")
except ImportError as e:
    print(f"✗ Failed to import database connection: {e}")
    sys.exit(1)

# Test 2: Import all models
print("\n[2/6] Testing model imports...")
try:
    from app.database.models import (
        User, Organization, UserProfile, Role, UserRole,
        Assessment, PipelineStage, BiomarkerValue,
        RetinalResult, SpeechResult, CardiologyResult,
        RadiologyResult, DermatologyResult, CognitiveResult,
        ChatThread, ChatMessage, AIExplanation,
        UploadedFile, AuditEvent
    )
    model_count = 17
    print(f"✓ All {model_count} models imported successfully")
except ImportError as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)

# Test 3: Import repository
print("\n[3/6] Testing repository imports...")
try:
    from app.database.repositories import AssessmentRepository
    print("✓ Assessment repository imported")
except ImportError as e:
    print(f"✗ Failed to import repository: {e}")
    sys.exit(1)

# Test 4: Check dependencies
print("\n[4/6] Checking dependencies...")
try:
    import alembic
    print(f"✓ Alembic {alembic.__version__} installed")
except ImportError:
    print("✗ Alembic not installed")
    sys.exit(1)

try:
    import asyncpg
    print(f"✓ asyncpg {asyncpg.__version__} installed")
except ImportError:
    print("✗ asyncpg not installed")
    sys.exit(1)

try:
    import tenacity
    print(f"✓ tenacity {tenacity.__version__} installed")
except ImportError:
    print("✗ tenacity not installed")
    sys.exit(1)

# Test 5: Check Alembic configuration
print("\n[5/6] Checking Alembic configuration...")
alembic_ini = backend_dir / "alembic.ini"
migrations_env = backend_dir / "migrations" / "env.py"

if alembic_ini.exists():
    print("✓ alembic.ini found")
else:
    print("✗ alembic.ini missing")
    sys.exit(1)

if migrations_env.exists():
    print("✓ migrations/env.py found")
else:
    print("✗ migrations/env.py missing")
    sys.exit(1)

# Test 6: Check environment configuration
print("\n[6/6] Checking environment configuration...")
env_example = backend_dir / ".env.example"
if env_example.exists():
    with open(env_example, 'r') as f:
        content = f.read()
        if "NEON_DATABASE_URL" in content:
            print("✓ .env.example contains NEON_DATABASE_URL")
        else:
            print("⚠  .env.example missing NEON_DATABASE_URL")
else:
    print("⚠  .env.example not found")

env_file = backend_dir / ".env"
if env_file.exists():
    print("✓ .env file exists")
    with open(env_file, 'r') as f:
        content = f.read()
        if "NEON_DATABASE_URL" in content:
            # Check if it's configured (not default)
            if "your-neon-host" not in content:
                print("✓ NEON_DATABASE_URL appears to be configured")
            else:
                print("⚠  NEON_DATABASE_URL needs configuration")
        else:
            print("⚠  .env missing NEON_DATABASE_URL")
else:
    print("⚠  .env file not found - copy from .env.example")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\n✓ All core components installed successfully!")
print("\n📋 Next Steps:")
print("   1. Create Neon database at https://neon.tech")
print("   2. Copy .env.example to .env")
print("   3. Add your NEON_DATABASE_URL to .env")
print("   4. Run: alembic revision --autogenerate -m 'Initial schema'")
print("   5. Run: alembic upgrade head")
print("   6. Run: python scripts/init_db.py")
print("\n📚 See DATABASE_SETUP.md for detailed instructions")
print("=" * 70)
