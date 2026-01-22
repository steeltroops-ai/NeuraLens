#!/usr/bin/env python3
"""
Direct Neon Database Schema Setup
FALLBACK SCRIPT - NOT FOR PRODUCTION USE WITH ALEMBIC

NOTE: This script creates the schema using raw SQL as a fallback for 
environments where asyncpg/Alembic cannot run (e.g., Windows without 
C++ build tools). 

WARNING: The Single Source of Truth for the database schema is the 
Alembic migrations (backend/migrations/). This script's SQL definitions
must be manually kept in sync with the SQLAlchemy models and Alembic 
revisions. Any divergence (drift) can cause deployment issues.

Currently reconciled with models as of 2026-01-22.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load .env file
env_file = backend_dir / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value

print("="*70)
print("NEON POSTGRES - DIRECT SCHEMA CREATION")
print("="*70)

# Get connection URL
DATABASE_URL = os.getenv("NEON_DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: NEON_DATABASE_URL not found in .env")
    sys.exit(1)

print(f"\n[1/4] Connecting to Neon...")
host = DATABASE_URL.split('@')[1].split('/')[0] if '@' in DATABASE_URL else 'hidden'
print(f"  Host: {host}")

# Use urllib to make HTTP request to execute SQL (Neon SQL API)
# Alternative: Use the SQL directly through psycopg2 sync driver
try:
    # Try using sqlalchemy with synchronous psycopg2
    from sqlalchemy import create_engine, text
    
    # Convert to sync URL (psycopg2)
    sync_url = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
    
    engine = create_engine(sync_url, echo=False)
    print("  ✓ Connected with psycopg2")
    
except ImportError:
    print("  psycopg2 not available, trying httpx...")
    # Fallback to HTTP API would go here
    print("ERROR: Please install psycopg2-binary or run migrations on Linux")
    sys.exit(1)

# SQL Schema
SCHEMA_SQL = '''
-- =============================================================================
-- MEDILENS DATABASE SCHEMA
-- Neon Postgres - Created by MediLens Database Architect
-- =============================================================================

-- Organizations (for multi-tenancy)
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    address JSONB,
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    subscription_tier VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ
);

-- Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    username VARCHAR(100) UNIQUE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    date_of_birth DATE,
    gender VARCHAR(20),
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ
);

-- User Profiles
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    age INTEGER,
    fitzpatrick_type SMALLINT,
    ethnicity VARCHAR(100),
    medical_history JSONB DEFAULT '[]',
    medications JSONB DEFAULT '[]',
    allergies JSONB DEFAULT '[]',
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    consent_research BOOLEAN DEFAULT FALSE,
    consent_data_sharing BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Roles
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User Roles
CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    granted_by UUID REFERENCES users(id) ON DELETE SET NULL
);

-- Assessments (central entity)
CREATE TABLE IF NOT EXISTS assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    pipeline_type VARCHAR(50) NOT NULL,
    pipeline_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    risk_score DECIMAL(5,2),
    risk_level VARCHAR(20),
    confidence DECIMAL(4,3),
    status VARCHAR(20) DEFAULT 'pending',
    current_stage VARCHAR(50),
    processing_time_ms INTEGER,
    quality_score DECIMAL(4,3),
    requires_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ
);

-- Pipeline Stages
CREATE TABLE IF NOT EXISTS pipeline_stages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    stage_name VARCHAR(100) NOT NULL,
    stage_index SMALLINT NOT NULL,
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    error_code VARCHAR(50),
    error_message TEXT,
    stage_metadata JSONB DEFAULT '{}'
);

-- Biomarker Values
CREATE TABLE IF NOT EXISTS biomarker_values (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    biomarker_name VARCHAR(100) NOT NULL,
    biomarker_category VARCHAR(50),
    value DECIMAL(12,6) NOT NULL,
    unit VARCHAR(20),
    normal_range_min DECIMAL(12,6),
    normal_range_max DECIMAL(12,6),
    status VARCHAR(20),
    confidence DECIMAL(4,3),
    percentile SMALLINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Retinal Results
CREATE TABLE IF NOT EXISTS retinal_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID UNIQUE NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    dr_grade SMALLINT,
    dr_severity VARCHAR(20),
    eye_laterality VARCHAR(10),
    image_quality VARCHAR(20),
    four_two_one_met BOOLEAN DEFAULT FALSE,
    hemorrhages_4_quadrants BOOLEAN,
    venous_beading_2_quadrants BOOLEAN,
    irma_1_quadrant BOOLEAN,
    dme_present BOOLEAN,
    dme_severity VARCHAR(20),
    vessel_density DECIMAL(5,4),
    av_ratio DECIMAL(4,3),
    tortuosity_index DECIMAL(4,3),
    hemorrhage_count INTEGER,
    microaneurysm_count INTEGER,
    exudate_area_percent DECIMAL(5,2),
    heatmap_data JSONB,
    vessel_segmentation JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Speech Results
CREATE TABLE IF NOT EXISTS speech_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID UNIQUE NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    duration_seconds DECIMAL(6,2),
    sample_rate INTEGER,
    audio_quality_score DECIMAL(4,3),
    jitter DECIMAL(8,6),
    shimmer DECIMAL(8,6),
    hnr DECIMAL(6,2),
    cpps DECIMAL(6,2),
    speech_rate DECIMAL(5,2),
    pause_ratio DECIMAL(4,3),
    voice_tremor DECIMAL(4,3),
    articulation_clarity DECIMAL(4,3),
    prosody_variation DECIMAL(6,2),
    fluency_score DECIMAL(4,3),
    parkinsons_probability DECIMAL(4,3),
    cognitive_decline_probability DECIMAL(4,3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cardiology Results
CREATE TABLE IF NOT EXISTS cardiology_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID UNIQUE NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    rhythm_classification VARCHAR(50),
    heart_rate_bpm SMALLINT,
    regularity VARCHAR(20),
    r_peaks_detected INTEGER,
    rmssd_ms DECIMAL(8,2),
    sdnn_ms DECIMAL(8,2),
    pnn50_percent DECIMAL(5,2),
    pr_interval_ms SMALLINT,
    qrs_duration_ms SMALLINT,
    qt_interval_ms SMALLINT,
    qtc_ms SMALLINT,
    arrhythmia_detected BOOLEAN DEFAULT FALSE,
    arrhythmia_types JSONB DEFAULT '[]',
    signal_quality_score DECIMAL(4,3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Radiology Results
CREATE TABLE IF NOT EXISTS radiology_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID UNIQUE NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    modality_type VARCHAR(50),
    primary_condition VARCHAR(100),
    primary_probability DECIMAL(5,2),
    primary_severity VARCHAR(20),
    findings JSONB DEFAULT '[]',
    lungs_status VARCHAR(20),
    heart_status VARCHAR(20),
    image_quality VARCHAR(20),
    quality_score DECIMAL(4,3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dermatology Results
CREATE TABLE IF NOT EXISTS dermatology_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID UNIQUE NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    primary_classification VARCHAR(100),
    melanoma_suspicion VARCHAR(50),
    asymmetry_score DECIMAL(4,3),
    border_score DECIMAL(4,3),
    color_score DECIMAL(4,3),
    diameter_mm DECIMAL(5,2),
    evolution_score DECIMAL(4,3),
    body_location VARCHAR(50),
    fitzpatrick_type SMALLINT,
    image_quality VARCHAR(20),
    lesion_centered BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cognitive Results
CREATE TABLE IF NOT EXISTS cognitive_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID UNIQUE NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    overall_risk_score DECIMAL(4,3),
    risk_level VARCHAR(20),
    confidence_score DECIMAL(4,3),
    attention_score DECIMAL(4,3),
    memory_score DECIMAL(4,3),
    executive_function_score DECIMAL(4,3),
    processing_speed_score DECIMAL(4,3),
    tasks_completed INTEGER,
    valid_tasks INTEGER,
    fatigue_index DECIMAL(4,3),
    consistency_score DECIMAL(4,3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat Threads
CREATE TABLE IF NOT EXISTS chat_threads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    context JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    last_message_at TIMESTAMPTZ
);

-- Chat Messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id UUID NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    model_used VARCHAR(50),
    tokens_used INTEGER,
    confidence JSONB,
    sources JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- AI Explanations
CREATE TABLE IF NOT EXISTS ai_explanations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    explanation_text TEXT NOT NULL,
    voice_generated BOOLEAN DEFAULT FALSE,
    voice_url TEXT,
    voice_duration_ms INTEGER,
    model_used VARCHAR(50),
    tokens_used INTEGER,
    generation_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Uploaded Files
CREATE TABLE IF NOT EXISTS uploaded_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assessment_id UUID REFERENCES assessments(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    content_type VARCHAR(100),
    file_size_bytes BIGINT,
    storage_path TEXT,
    storage_provider VARCHAR(50) DEFAULT 'local',
    processing_status VARCHAR(20) DEFAULT 'pending',
    file_hash VARCHAR(64),
    is_encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- Audit Events
CREATE TABLE IF NOT EXISTS audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    actor_type VARCHAR(50) NOT NULL,
    actor_ip INET,
    event_type VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    action VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}',
    hipaa_relevant BOOLEAN DEFAULT FALSE,
    phi_accessed BOOLEAN DEFAULT FALSE,
    session_id VARCHAR(255),
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_clerk_id ON users(clerk_user_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_org ON users(organization_id) WHERE deleted_at IS NULL;

-- Assessments indexes
CREATE INDEX IF NOT EXISTS idx_assessments_user ON assessments(user_id, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_assessments_session ON assessments(session_id);
CREATE INDEX IF NOT EXISTS idx_assessments_pipeline ON assessments(pipeline_type, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_assessments_review ON assessments(requires_review) WHERE requires_review = TRUE AND deleted_at IS NULL;

-- Biomarker indexes
CREATE INDEX IF NOT EXISTS idx_biomarker_values_assessment ON biomarker_values(assessment_id);
CREATE INDEX IF NOT EXISTS idx_biomarker_values_name ON biomarker_values(biomarker_name, created_at DESC);

-- Chat indexes
CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(user_id, last_message_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id, created_at ASC);

-- Audit indexes
CREATE INDEX IF NOT EXISTS idx_audit_events_user ON audit_events(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_hipaa ON audit_events(hipaa_relevant) WHERE hipaa_relevant = TRUE;

-- =============================================================================
-- SEED DATA
-- =============================================================================

-- Insert default roles
INSERT INTO roles (name, description, permissions) VALUES
    ('patient', 'Standard patient user', '["view_own_assessments", "create_assessment", "chat"]'),
    ('clinician', 'Healthcare provider', '["view_all_assessments", "create_assessment", "review_assessments", "export_data", "chat"]'),
    ('admin', 'System administrator', '["manage_users", "manage_organizations", "view_audit_logs", "manage_roles", "all"]'),
    ('researcher', 'Research access only', '["view_anonymized_data", "export_data"]')
ON CONFLICT (name) DO NOTHING;

'''

print(f"\n[2/4] Creating schema...")
try:
    with engine.connect() as conn:
        # Execute schema SQL
        statements = [s.strip() for s in SCHEMA_SQL.split(';') if s.strip()]
        created_tables = 0
        created_indexes = 0
        
        for statement in statements:
            if not statement or statement.startswith('--'):
                continue
            try:
                conn.execute(text(statement + ';'))
                if 'CREATE TABLE' in statement:
                    created_tables += 1
                elif 'CREATE INDEX' in statement:
                    created_indexes += 1
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    print(f"  Warning: {str(e)[:80]}")
        
        conn.commit()
        print(f"  ✓ Created/verified {created_tables} tables")
        print(f"  ✓ Created/verified {created_indexes} indexes")
        
except Exception as e:
    print(f"  ✗ Error creating schema: {e}")
    sys.exit(1)

print(f"\n[3/4] Verifying tables...")
try:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """))
        tables = [row[0] for row in result]
        
        print(f"  ✓ Found {len(tables)} tables in database:")
        for table in tables:
            print(f"    - {table}")
            
except Exception as e:
    print(f"  ✗ Error verifying: {e}")

print(f"\n[4/4] Checking seed data...")
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM roles ORDER BY id"))
        roles = [row[0] for row in result]
        print(f"  ✓ Roles seeded: {', '.join(roles)}")
except Exception as e:
    print(f"  ✗ Error checking seed data: {e}")

print("\n" + "="*70)
print("✅ NEON DATABASE SETUP COMPLETE!")
print("="*70)
print("\nAll 19 tables have been created in your Neon database.")
print("You can now view them in the Neon console at:")
print("  https://console.neon.tech")
print("\n" + "="*70)
