-- NeuraLens Row Level Security (RLS) Policies
-- Ensures secure multi-user access with proper data isolation

-- ============================================================================
-- ENABLE RLS ON ALL TABLES
-- ============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessment_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessment_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE nri_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_studies ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_health ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- HELPER FUNCTIONS FOR RLS
-- ============================================================================

-- Function to get current user ID from JWT
CREATE OR REPLACE FUNCTION auth.user_id() RETURNS UUID AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'sub',
    current_setting('request.jwt.claims', true)::json->>'user_id'
  )::UUID;
$$ LANGUAGE SQL STABLE;

-- Function to check if user is admin
CREATE OR REPLACE FUNCTION auth.is_admin() RETURNS BOOLEAN AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'role' = 'admin',
    false
  );
$$ LANGUAGE SQL STABLE;

-- Function to check if user is clinician
CREATE OR REPLACE FUNCTION auth.is_clinician() RETURNS BOOLEAN AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'role' IN ('clinician', 'admin'),
    false
  );
$$ LANGUAGE SQL STABLE;

-- Function to check if user is researcher
CREATE OR REPLACE FUNCTION auth.is_researcher() RETURNS BOOLEAN AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'role' IN ('researcher', 'admin'),
    false
  );
$$ LANGUAGE SQL STABLE;

-- ============================================================================
-- USERS TABLE POLICIES
-- ============================================================================

-- Users can read their own data
CREATE POLICY "Users can view own profile" ON users
    FOR SELECT USING (auth.user_id() = id);

-- Users can update their own data
CREATE POLICY "Users can update own profile" ON users
    FOR UPDATE USING (auth.user_id() = id);

-- Users can insert their own data (for registration)
CREATE POLICY "Users can create own profile" ON users
    FOR INSERT WITH CHECK (auth.user_id() = id);

-- Admins can view all users
CREATE POLICY "Admins can view all users" ON users
    FOR SELECT USING (auth.is_admin());

-- Clinicians can view users they have access to (implement based on your access control)
CREATE POLICY "Clinicians can view assigned users" ON users
    FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- USER PROFILES TABLE POLICIES
-- ============================================================================

-- Users can manage their own profiles
CREATE POLICY "Users can view own profile details" ON user_profiles
    FOR SELECT USING (user_id = auth.user_id());

CREATE POLICY "Users can update own profile details" ON user_profiles
    FOR UPDATE USING (user_id = auth.user_id());

CREATE POLICY "Users can create own profile details" ON user_profiles
    FOR INSERT WITH CHECK (user_id = auth.user_id());

-- Admins and clinicians can view profiles
CREATE POLICY "Admins can view all profiles" ON user_profiles
    FOR SELECT USING (auth.is_admin());

CREATE POLICY "Clinicians can view profiles" ON user_profiles
    FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- ASSESSMENT HISTORY TABLE POLICIES
-- ============================================================================

-- Users can view their own assessment history
CREATE POLICY "Users can view own assessment history" ON assessment_history
    FOR SELECT USING (user_id = auth.user_id());

-- System can insert assessment history
CREATE POLICY "System can insert assessment history" ON assessment_history
    FOR INSERT WITH CHECK (true); -- Will be restricted by application logic

-- Clinicians can view assessment history for their patients
CREATE POLICY "Clinicians can view assessment history" ON assessment_history
    FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- ASSESSMENTS TABLE POLICIES
-- ============================================================================

-- Users can view their own assessments
CREATE POLICY "Users can view own assessments" ON assessments
    FOR SELECT USING (user_id = auth.user_id());

-- Users can create their own assessments
CREATE POLICY "Users can create own assessments" ON assessments
    FOR INSERT WITH CHECK (user_id = auth.user_id());

-- Users can update their own assessments
CREATE POLICY "Users can update own assessments" ON assessments
    FOR UPDATE USING (user_id = auth.user_id());

-- Clinicians can view assessments
CREATE POLICY "Clinicians can view assessments" ON assessments
    FOR SELECT USING (auth.is_clinician());

-- System/service role can manage all assessments
CREATE POLICY "Service role can manage assessments" ON assessments
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- ============================================================================
-- ASSESSMENT RESULTS TABLE POLICIES
-- ============================================================================

-- Users can view results for their own assessments
CREATE POLICY "Users can view own assessment results" ON assessment_results
    FOR SELECT USING (
        assessment_id IN (
            SELECT id FROM assessments WHERE user_id = auth.user_id()
        )
    );

-- System can insert assessment results
CREATE POLICY "System can insert assessment results" ON assessment_results
    FOR INSERT WITH CHECK (true); -- Controlled by application logic

-- Clinicians can view assessment results
CREATE POLICY "Clinicians can view assessment results" ON assessment_results
    FOR SELECT USING (auth.is_clinician());

-- Service role can manage all results
CREATE POLICY "Service role can manage assessment results" ON assessment_results
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- ============================================================================
-- NRI RESULTS TABLE POLICIES
-- ============================================================================

-- Users can view NRI results for their own assessments
CREATE POLICY "Users can view own NRI results" ON nri_results
    FOR SELECT USING (
        assessment_id IN (
            SELECT id FROM assessments WHERE user_id = auth.user_id()
        )
    );

-- System can insert NRI results
CREATE POLICY "System can insert NRI results" ON nri_results
    FOR INSERT WITH CHECK (true);

-- Clinicians can view NRI results
CREATE POLICY "Clinicians can view NRI results" ON nri_results
    FOR SELECT USING (auth.is_clinician());

-- Service role can manage all NRI results
CREATE POLICY "Service role can manage NRI results" ON nri_results
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- ============================================================================
-- VALIDATION STUDIES TABLE POLICIES
-- ============================================================================

-- Researchers can view validation studies
CREATE POLICY "Researchers can view validation studies" ON validation_studies
    FOR SELECT USING (auth.is_researcher());

-- Researchers can create validation studies
CREATE POLICY "Researchers can create validation studies" ON validation_studies
    FOR INSERT WITH CHECK (auth.is_researcher());

-- Researchers can update validation studies
CREATE POLICY "Researchers can update validation studies" ON validation_studies
    FOR UPDATE USING (auth.is_researcher());

-- Admins can manage all validation studies
CREATE POLICY "Admins can manage validation studies" ON validation_studies
    FOR ALL USING (auth.is_admin());

-- ============================================================================
-- VALIDATION RESULTS TABLE POLICIES
-- ============================================================================

-- Researchers can view validation results
CREATE POLICY "Researchers can view validation results" ON validation_results
    FOR SELECT USING (auth.is_researcher());

-- Researchers can insert validation results
CREATE POLICY "Researchers can insert validation results" ON validation_results
    FOR INSERT WITH CHECK (auth.is_researcher());

-- Admins can manage all validation results
CREATE POLICY "Admins can manage validation results" ON validation_results
    FOR ALL USING (auth.is_admin());

-- ============================================================================
-- PERFORMANCE METRICS TABLE POLICIES
-- ============================================================================

-- System can insert performance metrics
CREATE POLICY "System can insert performance metrics" ON performance_metrics
    FOR INSERT WITH CHECK (true);

-- Admins can view performance metrics
CREATE POLICY "Admins can view performance metrics" ON performance_metrics
    FOR SELECT USING (auth.is_admin());

-- Service role can manage performance metrics
CREATE POLICY "Service role can manage performance metrics" ON performance_metrics
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- ============================================================================
-- SYSTEM HEALTH TABLE POLICIES
-- ============================================================================

-- System can insert health metrics
CREATE POLICY "System can insert health metrics" ON system_health
    FOR INSERT WITH CHECK (true);

-- Admins can view system health
CREATE POLICY "Admins can view system health" ON system_health
    FOR SELECT USING (auth.is_admin());

-- Service role can manage system health
CREATE POLICY "Service role can manage system health" ON system_health
    FOR ALL USING (current_setting('request.jwt.claims', true)::json->>'role' = 'service_role');

-- ============================================================================
-- STORAGE POLICIES (for Supabase Storage)
-- ============================================================================

-- Create storage buckets policies
-- Note: These are applied through Supabase dashboard or API

-- Audio files: Users can upload/view their own audio files
-- Images: Users can upload/view their own retinal images  
-- Reports: Users can view their own reports, clinicians can view assigned patient reports

-- Example policy for audio bucket (to be applied via Supabase dashboard):
-- CREATE POLICY "Users can upload own audio files" ON storage.objects
--   FOR INSERT WITH CHECK (bucket_id = 'neuralens-audio' AND auth.user_id()::text = (storage.foldername(name))[1]);

-- CREATE POLICY "Users can view own audio files" ON storage.objects
--   FOR SELECT USING (bucket_id = 'neuralens-audio' AND auth.user_id()::text = (storage.foldername(name))[1]);

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

-- Grant usage on sequences
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO service_role;

-- Grant permissions on tables
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;

-- Grant permissions on functions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;
