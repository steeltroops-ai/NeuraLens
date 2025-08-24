-- NeuraLens Supabase Storage Configuration
-- Sets up storage buckets and policies for file management

-- ============================================================================
-- STORAGE BUCKET CREATION
-- ============================================================================

-- Create storage buckets (these commands are typically run via Supabase API or dashboard)
-- But we'll document the configuration here

-- Audio files bucket for speech analysis
-- Bucket: neuralens-audio
-- Configuration:
-- - Private bucket (public = false)
-- - Max file size: 10MB
-- - Allowed MIME types: audio/wav, audio/mp3, audio/m4a, audio/webm, audio/ogg
-- - File size limit: 10485760 bytes (10MB)

-- Retinal images bucket
-- Bucket: neuralens-images  
-- Configuration:
-- - Private bucket (public = false)
-- - Max file size: 5MB
-- - Allowed MIME types: image/jpeg, image/png, image/jpg
-- - File size limit: 5242880 bytes (5MB)

-- Reports and documents bucket
-- Bucket: neuralens-reports
-- Configuration:
-- - Private bucket (public = false)
-- - Max file size: 2MB
-- - Allowed MIME types: application/pdf, text/plain, application/json
-- - File size limit: 2097152 bytes (2MB)

-- ============================================================================
-- STORAGE POLICIES
-- ============================================================================

-- Enable RLS on storage objects
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;
ALTER TABLE storage.buckets ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- AUDIO FILES POLICIES (neuralens-audio bucket)
-- ============================================================================

-- Users can upload their own audio files
CREATE POLICY "Users can upload own audio files" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'neuralens-audio' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Users can view their own audio files
CREATE POLICY "Users can view own audio files" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'neuralens-audio' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Users can update their own audio files
CREATE POLICY "Users can update own audio files" ON storage.objects
    FOR UPDATE USING (
        bucket_id = 'neuralens-audio' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Users can delete their own audio files
CREATE POLICY "Users can delete own audio files" ON storage.objects
    FOR DELETE USING (
        bucket_id = 'neuralens-audio' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Clinicians can view audio files for their patients
CREATE POLICY "Clinicians can view patient audio files" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'neuralens-audio' 
        AND (
            current_setting('request.jwt.claims', true)::json->>'role' IN ('clinician', 'admin')
        )
    );

-- Service role can manage all audio files
CREATE POLICY "Service role can manage audio files" ON storage.objects
    FOR ALL USING (
        bucket_id = 'neuralens-audio' 
        AND current_setting('request.jwt.claims', true)::json->>'role' = 'service_role'
    );

-- ============================================================================
-- IMAGE FILES POLICIES (neuralens-images bucket)
-- ============================================================================

-- Users can upload their own retinal images
CREATE POLICY "Users can upload own images" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'neuralens-images' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Users can view their own retinal images
CREATE POLICY "Users can view own images" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'neuralens-images' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Users can update their own retinal images
CREATE POLICY "Users can update own images" ON storage.objects
    FOR UPDATE USING (
        bucket_id = 'neuralens-images' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Users can delete their own retinal images
CREATE POLICY "Users can delete own images" ON storage.objects
    FOR DELETE USING (
        bucket_id = 'neuralens-images' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Clinicians can view retinal images for their patients
CREATE POLICY "Clinicians can view patient images" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'neuralens-images' 
        AND (
            current_setting('request.jwt.claims', true)::json->>'role' IN ('clinician', 'admin')
        )
    );

-- Service role can manage all retinal images
CREATE POLICY "Service role can manage images" ON storage.objects
    FOR ALL USING (
        bucket_id = 'neuralens-images' 
        AND current_setting('request.jwt.claims', true)::json->>'role' = 'service_role'
    );

-- ============================================================================
-- REPORTS POLICIES (neuralens-reports bucket)
-- ============================================================================

-- Users can view their own reports
CREATE POLICY "Users can view own reports" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'neuralens-reports' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- System can create reports for users
CREATE POLICY "System can create reports" ON storage.objects
    FOR INSERT WITH CHECK (
        bucket_id = 'neuralens-reports'
    );

-- Users can update their own reports
CREATE POLICY "Users can update own reports" ON storage.objects
    FOR UPDATE USING (
        bucket_id = 'neuralens-reports' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Users can delete their own reports
CREATE POLICY "Users can delete own reports" ON storage.objects
    FOR DELETE USING (
        bucket_id = 'neuralens-reports' 
        AND auth.user_id()::text = (storage.foldername(name))[1]
    );

-- Clinicians can view reports for their patients
CREATE POLICY "Clinicians can view patient reports" ON storage.objects
    FOR SELECT USING (
        bucket_id = 'neuralens-reports' 
        AND (
            current_setting('request.jwt.claims', true)::json->>'role' IN ('clinician', 'admin')
        )
    );

-- Service role can manage all reports
CREATE POLICY "Service role can manage reports" ON storage.objects
    FOR ALL USING (
        bucket_id = 'neuralens-reports' 
        AND current_setting('request.jwt.claims', true)::json->>'role' = 'service_role'
    );

-- ============================================================================
-- BUCKET POLICIES
-- ============================================================================

-- Allow authenticated users to view bucket information
CREATE POLICY "Authenticated users can view buckets" ON storage.buckets
    FOR SELECT USING (auth.role() = 'authenticated');

-- Service role can manage buckets
CREATE POLICY "Service role can manage buckets" ON storage.buckets
    FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- HELPER FUNCTIONS FOR FILE ORGANIZATION
-- ============================================================================

-- Function to generate organized file paths
CREATE OR REPLACE FUNCTION generate_file_path(
    user_id UUID,
    file_type TEXT,
    session_id TEXT,
    original_filename TEXT
) RETURNS TEXT AS $$
DECLARE
    file_extension TEXT;
    date_path TEXT;
    unique_filename TEXT;
BEGIN
    -- Extract file extension
    file_extension := LOWER(SUBSTRING(original_filename FROM '\.([^.]*)$'));
    
    -- Generate date-based path
    date_path := TO_CHAR(NOW(), 'YYYY/MM/DD');
    
    -- Generate unique filename
    unique_filename := session_id || '_' || gen_random_uuid()::text || '.' || file_extension;
    
    -- Return organized path
    RETURN user_id::text || '/' || file_type || '/' || date_path || '/' || unique_filename;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old files (to be called by scheduled job)
CREATE OR REPLACE FUNCTION cleanup_old_files(
    bucket_name TEXT,
    days_old INTEGER DEFAULT 30
) RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    file_record RECORD;
BEGIN
    -- Find files older than specified days
    FOR file_record IN 
        SELECT name, bucket_id 
        FROM storage.objects 
        WHERE bucket_id = bucket_name 
        AND created_at < NOW() - INTERVAL '1 day' * days_old
    LOOP
        -- Delete the file
        DELETE FROM storage.objects 
        WHERE bucket_id = file_record.bucket_id 
        AND name = file_record.name;
        
        deleted_count := deleted_count + 1;
    END LOOP;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FILE VALIDATION FUNCTIONS
-- ============================================================================

-- Function to validate audio file uploads
CREATE OR REPLACE FUNCTION validate_audio_file(
    filename TEXT,
    file_size BIGINT
) RETURNS BOOLEAN AS $$
DECLARE
    allowed_extensions TEXT[] := ARRAY['wav', 'mp3', 'm4a', 'webm', 'ogg'];
    file_extension TEXT;
    max_size BIGINT := 10485760; -- 10MB
BEGIN
    -- Extract file extension
    file_extension := LOWER(SUBSTRING(filename FROM '\.([^.]*)$'));
    
    -- Check file extension
    IF file_extension = ANY(allowed_extensions) AND file_size <= max_size THEN
        RETURN TRUE;
    ELSE
        RETURN FALSE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to validate image file uploads
CREATE OR REPLACE FUNCTION validate_image_file(
    filename TEXT,
    file_size BIGINT
) RETURNS BOOLEAN AS $$
DECLARE
    allowed_extensions TEXT[] := ARRAY['jpg', 'jpeg', 'png'];
    file_extension TEXT;
    max_size BIGINT := 5242880; -- 5MB
BEGIN
    -- Extract file extension
    file_extension := LOWER(SUBSTRING(filename FROM '\.([^.]*)$'));
    
    -- Check file extension
    IF file_extension = ANY(allowed_extensions) AND file_size <= max_size THEN
        RETURN TRUE;
    ELSE
        RETURN FALSE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STORAGE STATISTICS FUNCTIONS
-- ============================================================================

-- Function to get storage usage statistics
CREATE OR REPLACE FUNCTION get_storage_stats(bucket_name TEXT DEFAULT NULL)
RETURNS TABLE(
    bucket TEXT,
    file_count BIGINT,
    total_size BIGINT,
    avg_file_size NUMERIC,
    oldest_file TIMESTAMP WITH TIME ZONE,
    newest_file TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    IF bucket_name IS NULL THEN
        -- Return stats for all buckets
        RETURN QUERY
        SELECT 
            o.bucket_id::TEXT,
            COUNT(*)::BIGINT,
            COALESCE(SUM((o.metadata->>'size')::BIGINT), 0)::BIGINT,
            COALESCE(AVG((o.metadata->>'size')::BIGINT), 0)::NUMERIC,
            MIN(o.created_at),
            MAX(o.created_at)
        FROM storage.objects o
        GROUP BY o.bucket_id;
    ELSE
        -- Return stats for specific bucket
        RETURN QUERY
        SELECT 
            o.bucket_id::TEXT,
            COUNT(*)::BIGINT,
            COALESCE(SUM((o.metadata->>'size')::BIGINT), 0)::BIGINT,
            COALESCE(AVG((o.metadata->>'size')::BIGINT), 0)::NUMERIC,
            MIN(o.created_at),
            MAX(o.created_at)
        FROM storage.objects o
        WHERE o.bucket_id = bucket_name
        GROUP BY o.bucket_id;
    END IF;
END;
$$ LANGUAGE plpgsql;
