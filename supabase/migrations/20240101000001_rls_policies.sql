-- NeuraLens Row Level Security (RLS) Policies
-- Ensures secure multi-user access with proper data isolation

-- ============================================================================
-- ENABLE RLS ON ALL TABLES
-- ============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessment_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE speech_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE retinal_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE motor_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE cognitive_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE nri_calculations ENABLE ROW LEVEL SECURITY;

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

-- Users can read their own profile
CREATE POLICY "Users can read own profile" ON users
  FOR SELECT USING (id = auth.user_id());

-- Users can update their own profile
CREATE POLICY "Users can update own profile" ON users
  FOR UPDATE USING (id = auth.user_id());

-- Admins can read all users
CREATE POLICY "Admins can read all users" ON users
  FOR SELECT USING (auth.is_admin());

-- Clinicians can read patient profiles
CREATE POLICY "Clinicians can read patient profiles" ON users
  FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- USER PROFILES TABLE POLICIES
-- ============================================================================

-- Users can read their own profile
CREATE POLICY "Users can read own user_profile" ON user_profiles
  FOR SELECT USING (user_id = auth.user_id());

-- Users can update their own profile
CREATE POLICY "Users can update own user_profile" ON user_profiles
  FOR UPDATE USING (user_id = auth.user_id());

-- Users can insert their own profile
CREATE POLICY "Users can insert own user_profile" ON user_profiles
  FOR INSERT WITH CHECK (user_id = auth.user_id());

-- Clinicians can read patient profiles
CREATE POLICY "Clinicians can read patient user_profiles" ON user_profiles
  FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- ASSESSMENT SESSIONS TABLE POLICIES
-- ============================================================================

-- Users can read their own assessment sessions
CREATE POLICY "Users can read own assessment_sessions" ON assessment_sessions
  FOR SELECT USING (user_id = auth.user_id());

-- Users can insert their own assessment sessions
CREATE POLICY "Users can insert own assessment_sessions" ON assessment_sessions
  FOR INSERT WITH CHECK (user_id = auth.user_id());

-- Users can update their own assessment sessions
CREATE POLICY "Users can update own assessment_sessions" ON assessment_sessions
  FOR UPDATE USING (user_id = auth.user_id());

-- Clinicians can read patient assessment sessions
CREATE POLICY "Clinicians can read patient assessment_sessions" ON assessment_sessions
  FOR SELECT USING (auth.is_clinician());

-- Researchers can read anonymized assessment sessions
CREATE POLICY "Researchers can read anonymized assessment_sessions" ON assessment_sessions
  FOR SELECT USING (auth.is_researcher());

-- ============================================================================
-- SPEECH ASSESSMENTS TABLE POLICIES
-- ============================================================================

-- Users can read their own speech assessments
CREATE POLICY "Users can read own speech_assessments" ON speech_assessments
  FOR SELECT USING (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Users can insert their own speech assessments
CREATE POLICY "Users can insert own speech_assessments" ON speech_assessments
  FOR INSERT WITH CHECK (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Clinicians can read patient speech assessments
CREATE POLICY "Clinicians can read patient speech_assessments" ON speech_assessments
  FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- RETINAL ASSESSMENTS TABLE POLICIES
-- ============================================================================

-- Users can read their own retinal assessments
CREATE POLICY "Users can read own retinal_assessments" ON retinal_assessments
  FOR SELECT USING (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Users can insert their own retinal assessments
CREATE POLICY "Users can insert own retinal_assessments" ON retinal_assessments
  FOR INSERT WITH CHECK (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Clinicians can read patient retinal assessments
CREATE POLICY "Clinicians can read patient retinal_assessments" ON retinal_assessments
  FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- MOTOR ASSESSMENTS TABLE POLICIES
-- ============================================================================

-- Users can read their own motor assessments
CREATE POLICY "Users can read own motor_assessments" ON motor_assessments
  FOR SELECT USING (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Users can insert their own motor assessments
CREATE POLICY "Users can insert own motor_assessments" ON motor_assessments
  FOR INSERT WITH CHECK (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Clinicians can read patient motor assessments
CREATE POLICY "Clinicians can read patient motor_assessments" ON motor_assessments
  FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- COGNITIVE ASSESSMENTS TABLE POLICIES
-- ============================================================================

-- Users can read their own cognitive assessments
CREATE POLICY "Users can read own cognitive_assessments" ON cognitive_assessments
  FOR SELECT USING (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Users can insert their own cognitive assessments
CREATE POLICY "Users can insert own cognitive_assessments" ON cognitive_assessments
  FOR INSERT WITH CHECK (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Clinicians can read patient cognitive assessments
CREATE POLICY "Clinicians can read patient cognitive_assessments" ON cognitive_assessments
  FOR SELECT USING (auth.is_clinician());

-- ============================================================================
-- NRI CALCULATIONS TABLE POLICIES
-- ============================================================================

-- Users can read their own NRI calculations
CREATE POLICY "Users can read own nri_calculations" ON nri_calculations
  FOR SELECT USING (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Users can insert their own NRI calculations
CREATE POLICY "Users can insert own nri_calculations" ON nri_calculations
  FOR INSERT WITH CHECK (
    session_id IN (
      SELECT id FROM assessment_sessions WHERE user_id = auth.user_id()
    )
  );

-- Clinicians can read patient NRI calculations
CREATE POLICY "Clinicians can read patient nri_calculations" ON nri_calculations
  FOR SELECT USING (auth.is_clinician());

-- Researchers can read anonymized NRI calculations
CREATE POLICY "Researchers can read anonymized nri_calculations" ON nri_calculations
  FOR SELECT USING (auth.is_researcher());

-- ============================================================================
-- STORAGE POLICIES
-- ============================================================================

-- Users can upload files to their own folder
CREATE POLICY "Users can upload to own folder" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'assessments' AND 
    (storage.foldername(name))[1] = auth.user_id()::text
  );

-- Users can read their own files
CREATE POLICY "Users can read own files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'assessments' AND 
    (storage.foldername(name))[1] = auth.user_id()::text
  );

-- Clinicians can read patient files
CREATE POLICY "Clinicians can read patient files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'assessments' AND 
    auth.is_clinician()
  );
