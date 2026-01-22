#!/usr/bin/env python3
"""
Database Initialization Script
Creates all tables and seeds initial data for MediLens
"""

import asyncio
import os
import sys
from pathlib import Path

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

from app.database import db
from app.database.models import Role


async def init_database():
    """Initialize database with tables and seed data"""
    print("=" * 60)
    print("MediLens Database Initialization")
    print("=" * 60)
    
    # Initialize database connection
    print("\n[1/3] Connecting to Neon Postgres...")
    db.initialize()
    print("✓ Connected")
    
    # Create all tables
    print("\n[2/3] Creating database schema...")
    await db.create_all_tables()
    print("✓ Schema created")
    
    # Seed initial data
    print("\n[3/3] Seeding initial data...")
    async with db.session_scope() as session:
        # Check if roles exist
        from sqlalchemy import select
        result = await session.execute(select(Role))
        existing_roles = result.scalars().all()
        
        if not existing_roles:
            # Create default roles
            roles = [
                Role(
                    name='patient',
                    description='Standard patient user',
                    permissions=['view_own_assessments', 'create_assessment', 'chat']
                ),
                Role(
                    name='clinician',
                    description='Healthcare provider',
                    permissions=[
                        'view_all_assessments', 'create_assessment', 
                        'review_assessments', 'export_data', 'chat'
                    ]
                ),
                Role(
                    name='admin',
                    description='System administrator',
                    permissions=[
                        'manage_users', 'manage_organizations', 
                        'view_audit_logs', 'manage_roles', 'all'
                    ]
                ),
                Role(
                    name='researcher',
                    description='Research access only',
                    permissions=['view_anonymized_data', 'export_data']
                ),
            ]
            
            session.add_all(roles)
            print(f"✓ Created {len(roles)} default roles")
        else:
            print(f"✓ Found {len(existing_roles)} existing roles")
    
    # Close connection
    await db.close()
    
    print("\n" + "=" * 60)
    print("✓ Database initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(init_database())
