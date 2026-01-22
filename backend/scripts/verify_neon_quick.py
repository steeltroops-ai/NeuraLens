"""
Quick Neon Database Verification
Tests all core components without external dependencies
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

print("="*70)
print("NEON POSTGRES DATABASE - QUICK VERIFICATION")
print("="*70)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    from app.database import db, get_db, Base
    print("✓ Database connection manager")
    
    from app.database.models import (
        User, Assessment, BiomarkerValue,
        RetinalResult, SpeechResult, ChatThread
    )
    print("✓ Core models imported")
    
    from app.database.repositories import AssessmentRepository
    print("✓ Repository pattern")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Model validation
print("\n[2/5] Validating models...")
try:
    tables = Base.metadata.tables
    print(f"✓ {len(tables)} tables registered:")
    for name in sorted(tables.keys())[:10]:
        print(f"  - {name}")
    if len(tables) > 10:
        print(f"  ... and {len(tables) - 10} more")
except Exception as e:
    print(f"✗ Model validation failed: {e}")
    sys.exit(1)

# Test 3: Dependencies
print("\n[3/5] Checking dependencies...")
try:
    import alembic
    print(f"✓ Alembic {alembic.__version__}")
    
    import tenacity
    # tenacity doesn't have __version__, just check it imports
    print(f"✓ Tenacity installed")
    
    import sqlalchemy
    print(f"✓ SQLAlchemy {sqlalchemy.__version__}")
    
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Test 4: Files
print("\n[4/5] Checking configuration files...")
alembic_ini = backend_dir / "alembic.ini"
migrations_env = backend_dir / "migrations" / "env.py"
env_example = backend_dir / ".env.example"

files_ok = True
if alembic_ini.exists():
    print("✓ alembic.ini")
else:
    print("✗ alembic.ini missing")
    files_ok = False

if migrations_env.exists():
    print("✓ migrations/env.py")
else:
    print("✗ migrations/env.py missing")
    files_ok = False

if env_example.exists():
    with open(env_example) as f:
        if "NEON_DATABASE_URL" in f.read():
            print("✓ .env.example (with NEON_DATABASE_URL)")
        else:
            print("⚠ .env.example missing NEON_DATABASE_URL")
else:
    print("✗ .env.example missing")
    files_ok = False

# Test 5: Repository methods
print("\n[5/5] Verifying repository methods...")
methods = [
    'create_assessment',
    'get_user_assessments',
    'save_biomarkers',
    'get_user_biomarker_history',
    'update_assessment_status'
]

all_present = True
for method in methods:
    if hasattr(AssessmentRepository, method):
        print(f"✓ {method}")
    else:
        print(f"✗ {method} missing")
        all_present = False

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

if files_ok and all_present:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nDatabase implementation is complete and ready.")
    print("\n📋 Next Steps:")
    print("   1. Set NEON_DATABASE_URL in .env")
    print("   2. Run: alembic revision --autogenerate -m 'Initial schema'")
    print("   3. Run: alembic upgrade head  ")
    print("   4. Run: python scripts/init_db.py")
    print("\n📚 Documentation:")
    print("   - Setup: DATABASE_SETUP.md")
    print("   - Architecture: .gemini/DATABASE-ARCHITECTURE.md")
else:
    print("\n⚠ Some checks failed - review output above")

print("="*70)
