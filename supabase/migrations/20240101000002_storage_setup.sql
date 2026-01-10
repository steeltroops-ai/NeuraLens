-- NeuraLens Storage Setup for Supabase
-- Creates buckets and policies for file storage

-- ============================================================================
-- CREATE STORAGE BUCKETS
-- ============================================================================

-- Create assessments bucket for storing assessment files
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'assessments',
  'assessments',
  false,
  52428800, -- 50MB limit
  ARRAY[
    'audio/wav',
    'audio/mp3',
    'audio/mpeg',
    'audio/webm',
    'audio/ogg',
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/webp',
    'video/mp4',
    'video/webm',
    'video/mov',
    'application/json'
  ]
);

-- Create reports bucket for generated reports
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'reports',
  'reports',
  false,
  10485760, -- 10MB limit
  ARRAY[
    'application/pdf',
    'application/json',
    'text/csv',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  ]
);

-- Create models bucket for ML model files (admin only)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'models',
  'models',
  false,
  1073741824, -- 1GB limit
  ARRAY[
    'application/octet-stream',
    'application/x-onnx',
    'application/json'
  ]
);

-- ============================================================================
-- STORAGE POLICIES
-- ============================================================================

-- Enable RLS on storage objects
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- Assessments bucket policies
CREATE POLICY "Users can upload assessment files" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'assessments' AND 
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users can read own assessment files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'assessments' AND 
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users can update own assessment files" ON storage.objects
  FOR UPDATE USING (
    bucket_id = 'assessments' AND 
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users can delete own assessment files" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'assessments' AND 
    (storage.foldername(name))[1] = auth.uid()::text
  );

-- Clinicians can access patient assessment files
CREATE POLICY "Clinicians can read patient assessment files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'assessments' AND 
    auth.is_clinician()
  );

-- Reports bucket policies
CREATE POLICY "Users can upload report files" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'reports' AND 
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users can read own report files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'reports' AND 
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Clinicians can read patient report files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'reports' AND 
    auth.is_clinician()
  );

-- Models bucket policies (admin only)
CREATE POLICY "Admins can manage model files" ON storage.objects
  FOR ALL USING (
    bucket_id = 'models' AND 
    auth.is_admin()
  );

-- ============================================================================
-- HELPER FUNCTIONS FOR FILE MANAGEMENT
-- ============================================================================

-- Function to generate secure file paths
CREATE OR REPLACE FUNCTION generate_file_path(
  user_id UUID,
  assessment_type TEXT,
  file_extension TEXT
) RETURNS TEXT AS $$
BEGIN
  RETURN user_id::text || '/' || assessment_type || '/' || 
         extract(epoch from now())::bigint || '_' || 
         substr(md5(random()::text), 1, 8) || '.' || file_extension;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old files
CREATE OR REPLACE FUNCTION cleanup_old_files(days_old INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER := 0;
BEGIN
  -- Delete old assessment files
  DELETE FROM storage.objects 
  WHERE bucket_id = 'assessments' 
    AND created_at < NOW() - INTERVAL '1 day' * days_old;
  
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  
  -- Delete old report files
  DELETE FROM storage.objects 
  WHERE bucket_id = 'reports' 
    AND created_at < NOW() - INTERVAL '1 day' * days_old;
  
  GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
  
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- FILE VALIDATION FUNCTIONS
-- ============================================================================

-- Function to validate audio file metadata
CREATE OR REPLACE FUNCTION validate_audio_file(
  file_size BIGINT,
  mime_type TEXT,
  duration_seconds DECIMAL DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
  -- Check file size (max 50MB)
  IF file_size > 52428800 THEN
    RETURN FALSE;
  END IF;
  
  -- Check mime type
  IF mime_type NOT IN ('audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/webm', 'audio/ogg') THEN
    RETURN FALSE;
  END IF;
  
  -- Check duration if provided (max 10 minutes)
  IF duration_seconds IS NOT NULL AND duration_seconds > 600 THEN
    RETURN FALSE;
  END IF;
  
  RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to validate image file metadata
CREATE OR REPLACE FUNCTION validate_image_file(
  file_size BIGINT,
  mime_type TEXT,
  width INTEGER DEFAULT NULL,
  height INTEGER DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
  -- Check file size (max 10MB)
  IF file_size > 10485760 THEN
    RETURN FALSE;
  END IF;
  
  -- Check mime type
  IF mime_type NOT IN ('image/jpeg', 'image/jpg', 'image/png', 'image/webp') THEN
    RETURN FALSE;
  END IF;
  
  -- Check dimensions if provided (min 512x512, max 4096x4096)
  IF width IS NOT NULL AND height IS NOT NULL THEN
    IF width < 512 OR height < 512 OR width > 4096 OR height > 4096 THEN
      RETURN FALSE;
    END IF;
  END IF;
  
  RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STORAGE TRIGGERS
-- ============================================================================

-- Function to log file uploads
CREATE OR REPLACE FUNCTION log_file_upload()
RETURNS TRIGGER AS $$
BEGIN
  -- Log the file upload for audit purposes
  INSERT INTO storage.audit_log (
    bucket_id,
    object_name,
    operation,
    user_id,
    created_at
  ) VALUES (
    NEW.bucket_id,
    NEW.name,
    'INSERT',
    auth.uid(),
    NOW()
  );
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create audit log table if it doesn't exist
CREATE TABLE IF NOT EXISTS storage.audit_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  bucket_id TEXT NOT NULL,
  object_name TEXT NOT NULL,
  operation TEXT NOT NULL,
  user_id UUID,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create trigger for file upload logging
CREATE TRIGGER log_storage_uploads
  AFTER INSERT ON storage.objects
  FOR EACH ROW
  EXECUTE FUNCTION log_file_upload();

-- ============================================================================
-- INDEXES FOR STORAGE PERFORMANCE
-- ============================================================================

-- Index for faster file lookups
CREATE INDEX IF NOT EXISTS idx_storage_objects_bucket_user 
ON storage.objects (bucket_id, (metadata->>'user_id'));

CREATE INDEX IF NOT EXISTS idx_storage_objects_created_at 
ON storage.objects (created_at);

CREATE INDEX IF NOT EXISTS idx_storage_audit_log_user_id 
ON storage.audit_log (user_id);

CREATE INDEX IF NOT EXISTS idx_storage_audit_log_created_at 
ON storage.audit_log (created_at);
