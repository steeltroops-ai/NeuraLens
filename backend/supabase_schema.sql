-- NeuraLens Complete Database Schema for Supabase PostgreSQL
-- This schema supports the full multi-modal neurological assessment platform

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create custom types
CREATE TYPE user_role AS ENUM ('patient', 'clinician', 'admin', 'researcher');
CREATE TYPE assessment_status AS ENUM ('in_progress', 'completed', 'failed', 'cancelled');
CREATE TYPE risk_category AS ENUM ('low', 'moderate', 'high', 'very_high');
CREATE TYPE modality_type AS ENUM ('speech', 'retinal', 'motor', 'cognitive', 'nri');
CREATE TYPE study_status AS ENUM ('planning', 'active', 'completed', 'terminated');

-- ============================================================================
-- USERS AND AUTHENTICATION
-- ============================================================================

-- Main users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE,
    username VARCHAR(100) UNIQUE,
    
    -- Demographics for risk assessment
    age INTEGER CHECK (age >= 0 AND age <= 150),
    sex VARCHAR(10) CHECK (sex IN ('male', 'female', 'other')),
    education_years INTEGER CHECK (education_years >= 0 AND education_years <= 30),
    
    -- Medical history (JSONB for better performance)
    family_history JSONB DEFAULT '{}',
    medical_history JSONB DEFAULT '{}',
    medications JSONB DEFAULT '[]',
    lifestyle_factors JSONB DEFAULT '{}',
    
    -- Account metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_assessment TIMESTAMP WITH TIME ZONE,
    
    -- Privacy and consent
    consent_given BOOLEAN DEFAULT FALSE,
    privacy_settings JSONB DEFAULT '{}',
    
    -- Soft delete
    is_active BOOLEAN DEFAULT TRUE
);

-- Extended user profiles
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Extended demographics
    ethnicity VARCHAR(100),
    occupation VARCHAR(200),
    handedness VARCHAR(15) CHECK (handedness IN ('left', 'right', 'ambidextrous')),
    
    -- Health baselines
    baseline_cognitive_score FLOAT CHECK (baseline_cognitive_score >= 0 AND baseline_cognitive_score <= 1),
    baseline_motor_score FLOAT CHECK (baseline_motor_score >= 0 AND baseline_motor_score <= 1),
    baseline_speech_score FLOAT CHECK (baseline_speech_score >= 0 AND baseline_speech_score <= 1),
    baseline_retinal_score FLOAT CHECK (baseline_retinal_score >= 0 AND baseline_retinal_score <= 1),
    
    -- Risk factors
    genetic_risk_factors JSONB DEFAULT '{}',
    environmental_risk_factors JSONB DEFAULT '{}',
    
    -- Preferences
    language_preference VARCHAR(10) DEFAULT 'en',
    notification_preferences JSONB DEFAULT '{}',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id)
);

-- Assessment history tracking
CREATE TABLE assessment_history (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    assessment_id INTEGER, -- Will reference assessments table
    
    -- Summary scores
    nri_score FLOAT CHECK (nri_score >= 0 AND nri_score <= 100),
    overall_risk_category risk_category,
    
    -- Trend analysis
    score_change FLOAT,
    trend_direction VARCHAR(20) CHECK (trend_direction IN ('improving', 'stable', 'declining')),
    
    -- Metadata
    assessment_date TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- ASSESSMENTS AND RESULTS
-- ============================================================================

-- Main assessments table
CREATE TABLE assessments (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Assessment metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status assessment_status DEFAULT 'in_progress',
    
    -- Assessment configuration
    modalities JSONB DEFAULT '[]', -- List of modalities included
    assessment_type VARCHAR(100) DEFAULT 'full', -- full, quick, targeted
    
    -- Progress tracking
    completed_modalities JSONB DEFAULT '[]',
    total_processing_time FLOAT DEFAULT 0,
    
    -- Quality metrics
    overall_quality_score FLOAT CHECK (overall_quality_score >= 0 AND overall_quality_score <= 1)
);

-- Assessment results for each modality
CREATE TABLE assessment_results (
    id SERIAL PRIMARY KEY,
    assessment_id INTEGER REFERENCES assessments(id) ON DELETE CASCADE,
    
    -- Result metadata
    modality modality_type NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_time FLOAT CHECK (processing_time >= 0),
    
    -- Scores and metrics
    risk_score FLOAT CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    
    -- Detailed results (JSONB for flexibility and performance)
    biomarkers JSONB DEFAULT '{}',
    raw_data JSONB DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',
    
    -- File information for uploaded files
    file_info JSONB DEFAULT '{}',
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

-- NRI Fusion results (separate table for importance)
CREATE TABLE nri_results (
    id SERIAL PRIMARY KEY,
    assessment_id INTEGER REFERENCES assessments(id) ON DELETE CASCADE,
    
    -- NRI scores
    nri_score FLOAT NOT NULL CHECK (nri_score >= 0 AND nri_score <= 100),
    risk_category risk_category NOT NULL,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    uncertainty FLOAT CHECK (uncertainty >= 0 AND uncertainty <= 1),
    consistency_score FLOAT CHECK (consistency_score >= 0 AND consistency_score <= 1),
    
    -- Modality contributions
    modality_contributions JSONB NOT NULL DEFAULT '[]',
    fusion_method VARCHAR(50) DEFAULT 'bayesian',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_time FLOAT CHECK (processing_time >= 0),
    
    -- Clinical recommendations
    recommendations JSONB DEFAULT '[]',
    follow_up_actions JSONB DEFAULT '[]',
    
    -- Unique constraint to ensure one NRI result per assessment
    UNIQUE(assessment_id)
);

-- ============================================================================
-- VALIDATION AND CLINICAL STUDIES
-- ============================================================================

-- Clinical validation studies
CREATE TABLE validation_studies (
    id SERIAL PRIMARY KEY,
    study_id UUID UNIQUE DEFAULT gen_random_uuid(),
    
    -- Study metadata
    name VARCHAR(500) NOT NULL,
    description TEXT,
    principal_investigator VARCHAR(200),
    institution VARCHAR(300),
    
    -- Study parameters
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    target_participants INTEGER CHECK (target_participants > 0),
    actual_participants INTEGER DEFAULT 0 CHECK (actual_participants >= 0),
    study_sites INTEGER DEFAULT 1 CHECK (study_sites > 0),
    
    -- Study design
    study_type VARCHAR(100), -- prospective, retrospective, cross-sectional
    control_group BOOLEAN DEFAULT FALSE,
    blinded BOOLEAN DEFAULT FALSE,
    
    -- Status
    status study_status DEFAULT 'planning',
    
    -- Results summary
    overall_accuracy FLOAT CHECK (overall_accuracy >= 0 AND overall_accuracy <= 1),
    sensitivity FLOAT CHECK (sensitivity >= 0 AND sensitivity <= 1),
    specificity FLOAT CHECK (specificity >= 0 AND specificity <= 1),
    auc_score FLOAT CHECK (auc_score >= 0 AND auc_score <= 1),
    f1_score FLOAT CHECK (f1_score >= 0 AND f1_score <= 1),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Individual validation results
CREATE TABLE validation_results (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES validation_studies(id) ON DELETE CASCADE,
    participant_id VARCHAR(255) NOT NULL,
    
    -- Assessment details
    modality modality_type NOT NULL,
    assessment_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Predictions
    predicted_risk FLOAT CHECK (predicted_risk >= 0 AND predicted_risk <= 1),
    predicted_category risk_category,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Ground truth
    actual_diagnosis VARCHAR(200),
    actual_risk_category risk_category,
    clinical_scores JSONB DEFAULT '{}', -- MMSE, MoCA, UPDRS, etc.
    
    -- Performance metrics
    correct_prediction BOOLEAN,
    prediction_error FLOAT,
    
    -- Participant demographics
    age INTEGER CHECK (age >= 0 AND age <= 150),
    sex VARCHAR(10) CHECK (sex IN ('male', 'female', 'other')),
    education_years INTEGER CHECK (education_years >= 0 AND education_years <= 30),
    
    -- Additional data
    biomarkers JSONB DEFAULT '{}',
    processing_time FLOAT CHECK (processing_time >= 0),
    quality_metrics JSONB DEFAULT '{}',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- PERFORMANCE MONITORING
-- ============================================================================

-- System performance metrics
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    component VARCHAR(100), -- frontend, backend, ml_model
    modality modality_type,
    
    -- Metric values
    value FLOAT NOT NULL,
    unit VARCHAR(50), -- ms, seconds, percentage, etc.
    target_value FLOAT,
    threshold_warning FLOAT,
    threshold_critical FLOAT,
    
    -- Context
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    environment VARCHAR(50), -- development, staging, production
    version VARCHAR(50),
    user_agent VARCHAR(500),
    
    -- Additional context
    extra_data JSONB DEFAULT '{}'
);

-- System health monitoring
CREATE TABLE system_health (
    id SERIAL PRIMARY KEY,
    
    -- Health check details
    component VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('healthy', 'warning', 'critical', 'down')),
    
    -- Metrics
    response_time FLOAT CHECK (response_time >= 0),
    cpu_usage FLOAT CHECK (cpu_usage >= 0 AND cpu_usage <= 100),
    memory_usage FLOAT CHECK (memory_usage >= 0 AND memory_usage <= 100),
    disk_usage FLOAT CHECK (disk_usage >= 0 AND disk_usage <= 100),
    
    -- Error information
    error_count INTEGER DEFAULT 0 CHECK (error_count >= 0),
    last_error TEXT,
    
    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Additional data
    details JSONB DEFAULT '{}'
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Users table indexes
CREATE INDEX idx_users_email ON users(email) WHERE email IS NOT NULL;
CREATE INDEX idx_users_username ON users(username) WHERE username IS NOT NULL;
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;

-- Assessments table indexes
CREATE INDEX idx_assessments_user_id ON assessments(user_id);
CREATE INDEX idx_assessments_session_id ON assessments(session_id);
CREATE INDEX idx_assessments_status ON assessments(status);
CREATE INDEX idx_assessments_created_at ON assessments(created_at);

-- Assessment results indexes
CREATE INDEX idx_assessment_results_assessment_id ON assessment_results(assessment_id);
CREATE INDEX idx_assessment_results_modality ON assessment_results(modality);
CREATE INDEX idx_assessment_results_created_at ON assessment_results(created_at);

-- NRI results indexes
CREATE INDEX idx_nri_results_assessment_id ON nri_results(assessment_id);
CREATE INDEX idx_nri_results_risk_category ON nri_results(risk_category);
CREATE INDEX idx_nri_results_created_at ON nri_results(created_at);

-- Validation studies indexes
CREATE INDEX idx_validation_studies_status ON validation_studies(status);
CREATE INDEX idx_validation_studies_study_id ON validation_studies(study_id);

-- Performance monitoring indexes
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_performance_metrics_component ON performance_metrics(component);
CREATE INDEX idx_system_health_timestamp ON system_health(timestamp);
CREATE INDEX idx_system_health_component ON system_health(component);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_assessments_updated_at BEFORE UPDATE ON assessments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_validation_studies_updated_at BEFORE UPDATE ON validation_studies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
