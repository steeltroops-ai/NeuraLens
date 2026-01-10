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
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Clinical information
    primary_language VARCHAR(50) DEFAULT 'en',
    handedness VARCHAR(10) CHECK (handedness IN ('left', 'right', 'ambidextrous')),
    vision_corrected BOOLEAN DEFAULT FALSE,
    hearing_impaired BOOLEAN DEFAULT FALSE,
    
    -- Baseline measurements
    baseline_speech_metrics JSONB DEFAULT '{}',
    baseline_motor_metrics JSONB DEFAULT '{}',
    baseline_cognitive_metrics JSONB DEFAULT '{}',
    baseline_retinal_metrics JSONB DEFAULT '{}',
    
    -- Preferences
    ui_preferences JSONB DEFAULT '{}',
    notification_preferences JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- ASSESSMENT SESSIONS
-- ============================================================================

-- Main assessment sessions
CREATE TABLE assessment_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Session metadata
    session_type VARCHAR(50) NOT NULL, -- 'screening', 'monitoring', 'diagnostic'
    status assessment_status DEFAULT 'in_progress',
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    
    -- Environment and device info
    device_info JSONB DEFAULT '{}',
    environment_conditions JSONB DEFAULT '{}',
    
    -- Overall results
    overall_risk_score DECIMAL(5,4) CHECK (overall_risk_score >= 0 AND overall_risk_score <= 1),
    risk_category risk_category,
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Quality metrics
    data_quality_score DECIMAL(5,4) CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    completion_percentage DECIMAL(5,2) CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    
    -- Metadata
    notes TEXT,
    flags JSONB DEFAULT '[]',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- MODALITY-SPECIFIC ASSESSMENTS
-- ============================================================================

-- Speech assessments
CREATE TABLE speech_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES assessment_sessions(id) ON DELETE CASCADE,
    
    -- Audio file information
    audio_file_path VARCHAR(500),
    audio_duration_seconds DECIMAL(8,3),
    audio_quality_score DECIMAL(5,4),
    sample_rate INTEGER DEFAULT 44100,
    
    -- Processing metadata
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    preprocessing_applied JSONB DEFAULT '[]',
    
    -- Speech biomarkers
    fluency_score DECIMAL(5,4) CHECK (fluency_score >= 0 AND fluency_score <= 1),
    articulation_score DECIMAL(5,4) CHECK (articulation_score >= 0 AND articulation_score <= 1),
    prosody_score DECIMAL(5,4) CHECK (prosody_score >= 0 AND prosody_score <= 1),
    voice_quality_score DECIMAL(5,4) CHECK (voice_quality_score >= 0 AND voice_quality_score <= 1),
    
    -- Detailed metrics
    pause_patterns JSONB DEFAULT '{}',
    speech_rate_wpm DECIMAL(6,2),
    fundamental_frequency JSONB DEFAULT '{}',
    formant_analysis JSONB DEFAULT '{}',
    spectral_features JSONB DEFAULT '{}',
    
    -- Risk assessment
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Transcription and analysis
    transcription TEXT,
    language_detected VARCHAR(10),
    sentiment_analysis JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Retinal assessments
CREATE TABLE retinal_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES assessment_sessions(id) ON DELETE CASCADE,
    
    -- Image file information
    image_file_path VARCHAR(500),
    image_quality_score DECIMAL(5,4),
    image_resolution VARCHAR(20),
    eye_examined VARCHAR(10) CHECK (eye_examined IN ('left', 'right', 'both')),
    
    -- Processing metadata
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    preprocessing_applied JSONB DEFAULT '[]',
    
    -- Retinal biomarkers
    vessel_tortuosity DECIMAL(5,4) CHECK (vessel_tortuosity >= 0 AND vessel_tortuosity <= 1),
    av_ratio DECIMAL(4,3) CHECK (av_ratio >= 0 AND av_ratio <= 2),
    cup_disc_ratio DECIMAL(4,3) CHECK (cup_disc_ratio >= 0 AND cup_disc_ratio <= 1),
    vessel_density DECIMAL(5,4) CHECK (vessel_density >= 0 AND vessel_density <= 1),
    
    -- Detailed analysis
    vessel_analysis JSONB DEFAULT '{}',
    optic_disc_analysis JSONB DEFAULT '{}',
    macula_analysis JSONB DEFAULT '{}',
    hemorrhage_detection JSONB DEFAULT '{}',
    exudate_detection JSONB DEFAULT '{}',
    
    -- Risk assessment
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Clinical findings
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Motor assessments
CREATE TABLE motor_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES assessment_sessions(id) ON DELETE CASCADE,
    
    -- Test metadata
    test_type VARCHAR(50), -- 'finger_tapping', 'hand_movement', 'gait'
    test_duration_seconds INTEGER,
    hand_tested VARCHAR(10) CHECK (hand_tested IN ('left', 'right', 'both')),
    
    -- Processing metadata
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    
    -- Motor biomarkers
    tap_frequency DECIMAL(6,2),
    rhythm_consistency DECIMAL(5,4) CHECK (rhythm_consistency >= 0 AND rhythm_consistency <= 1),
    movement_amplitude DECIMAL(8,4),
    tremor_score DECIMAL(5,4) CHECK (tremor_score >= 0 AND tremor_score <= 1),
    coordination_score DECIMAL(5,4) CHECK (coordination_score >= 0 AND coordination_score <= 1),
    
    -- Detailed metrics
    tap_intervals JSONB DEFAULT '[]',
    movement_trajectory JSONB DEFAULT '{}',
    acceleration_patterns JSONB DEFAULT '{}',
    bradykinesia_indicators JSONB DEFAULT '{}',
    
    -- Risk assessment
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Raw data reference
    raw_data_path VARCHAR(500),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cognitive assessments
CREATE TABLE cognitive_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES assessment_sessions(id) ON DELETE CASCADE,
    
    -- Test metadata
    test_battery VARCHAR(100),
    total_duration_seconds INTEGER,
    
    -- Processing metadata
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    
    -- Cognitive domains
    attention_score DECIMAL(5,4) CHECK (attention_score >= 0 AND attention_score <= 1),
    memory_score DECIMAL(5,4) CHECK (memory_score >= 0 AND memory_score <= 1),
    executive_function_score DECIMAL(5,4) CHECK (executive_function_score >= 0 AND executive_function_score <= 1),
    language_score DECIMAL(5,4) CHECK (language_score >= 0 AND language_score <= 1),
    visuospatial_score DECIMAL(5,4) CHECK (visuospatial_score >= 0 AND visuospatial_score <= 1),
    
    -- Detailed metrics
    reaction_times JSONB DEFAULT '[]',
    accuracy_scores JSONB DEFAULT '{}',
    error_patterns JSONB DEFAULT '{}',
    learning_curves JSONB DEFAULT '{}',
    
    -- Risk assessment
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Individual test results
    test_results JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- NEURO-RISK INDEX (NRI) FUSION
-- ============================================================================

-- NRI calculations and fusion results
CREATE TABLE nri_calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES assessment_sessions(id) ON DELETE CASCADE,
    
    -- Input modality scores
    speech_risk_score DECIMAL(5,4),
    retinal_risk_score DECIMAL(5,4),
    motor_risk_score DECIMAL(5,4),
    cognitive_risk_score DECIMAL(5,4),
    
    -- Fusion methodology
    fusion_algorithm VARCHAR(50) DEFAULT 'weighted_ensemble',
    fusion_weights JSONB DEFAULT '{}',
    
    -- Final NRI
    nri_score DECIMAL(5,4) CHECK (nri_score >= 0 AND nri_score <= 1),
    nri_category risk_category,
    confidence_interval JSONB DEFAULT '{}',
    
    -- Uncertainty quantification
    epistemic_uncertainty DECIMAL(5,4),
    aleatoric_uncertainty DECIMAL(5,4),
    total_uncertainty DECIMAL(5,4),
    
    -- Explainability
    feature_importance JSONB DEFAULT '{}',
    risk_factors JSONB DEFAULT '[]',
    protective_factors JSONB DEFAULT '[]',
    
    -- Metadata
    calculation_time_ms INTEGER,
    model_versions JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- User indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);
CREATE INDEX idx_users_last_assessment ON users(last_assessment);

-- Session indexes
CREATE INDEX idx_sessions_user_id ON assessment_sessions(user_id);
CREATE INDEX idx_sessions_status ON assessment_sessions(status);
CREATE INDEX idx_sessions_started_at ON assessment_sessions(started_at);
CREATE INDEX idx_sessions_risk_category ON assessment_sessions(risk_category);

-- Assessment indexes
CREATE INDEX idx_speech_session_id ON speech_assessments(session_id);
CREATE INDEX idx_speech_created_at ON speech_assessments(created_at);
CREATE INDEX idx_retinal_session_id ON retinal_assessments(session_id);
CREATE INDEX idx_retinal_created_at ON retinal_assessments(created_at);
CREATE INDEX idx_motor_session_id ON motor_assessments(session_id);
CREATE INDEX idx_motor_created_at ON motor_assessments(created_at);
CREATE INDEX idx_cognitive_session_id ON cognitive_assessments(session_id);
CREATE INDEX idx_cognitive_created_at ON cognitive_assessments(created_at);

-- NRI indexes
CREATE INDEX idx_nri_session_id ON nri_calculations(session_id);
CREATE INDEX idx_nri_score ON nri_calculations(nri_score);
CREATE INDEX idx_nri_category ON nri_calculations(nri_category);

-- ============================================================================
-- TRIGGERS FOR UPDATED_AT
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_assessment_sessions_updated_at BEFORE UPDATE ON assessment_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
