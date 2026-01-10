-- NeuraLens Database Seed Data
-- Sample data for development and testing

-- ============================================================================
-- SAMPLE USERS
-- ============================================================================

-- Insert sample users (passwords will be handled by Supabase Auth)
INSERT INTO users (id, email, username, age, sex, education_years, consent_given, is_active) VALUES
  ('550e8400-e29b-41d4-a716-446655440001', 'patient1@example.com', 'patient1', 65, 'male', 16, true, true),
  ('550e8400-e29b-41d4-a716-446655440002', 'patient2@example.com', 'patient2', 72, 'female', 12, true, true),
  ('550e8400-e29b-41d4-a716-446655440003', 'clinician1@example.com', 'dr_smith', 45, 'female', 20, true, true),
  ('550e8400-e29b-41d4-a716-446655440004', 'researcher1@example.com', 'researcher1', 38, 'male', 22, true, true);

-- Insert user profiles
INSERT INTO user_profiles (user_id, primary_language, handedness, vision_corrected, hearing_impaired) VALUES
  ('550e8400-e29b-41d4-a716-446655440001', 'en', 'right', true, false),
  ('550e8400-e29b-41d4-a716-446655440002', 'en', 'right', false, false),
  ('550e8400-e29b-41d4-a716-446655440003', 'en', 'right', false, false),
  ('550e8400-e29b-41d4-a716-446655440004', 'en', 'left', true, false);

-- ============================================================================
-- SAMPLE ASSESSMENT SESSIONS
-- ============================================================================

-- Insert sample assessment sessions
INSERT INTO assessment_sessions (
  id, 
  user_id, 
  session_type, 
  status, 
  started_at, 
  completed_at, 
  duration_seconds,
  overall_risk_score,
  risk_category,
  confidence_score,
  data_quality_score,
  completion_percentage
) VALUES
  (
    '660e8400-e29b-41d4-a716-446655440001',
    '550e8400-e29b-41d4-a716-446655440001',
    'screening',
    'completed',
    '2024-01-15 10:30:00+00',
    '2024-01-15 10:45:00+00',
    900,
    0.25,
    'low',
    0.92,
    0.95,
    100.0
  ),
  (
    '660e8400-e29b-41d4-a716-446655440002',
    '550e8400-e29b-41d4-a716-446655440002',
    'monitoring',
    'completed',
    '2024-01-14 14:15:00+00',
    '2024-01-14 14:35:00+00',
    1200,
    0.45,
    'moderate',
    0.88,
    0.91,
    100.0
  ),
  (
    '660e8400-e29b-41d4-a716-446655440003',
    '550e8400-e29b-41d4-a716-446655440001',
    'diagnostic',
    'completed',
    '2024-01-13 09:45:00+00',
    '2024-01-13 10:15:00+00',
    1800,
    0.15,
    'low',
    0.95,
    0.98,
    100.0
  );

-- ============================================================================
-- SAMPLE SPEECH ASSESSMENTS
-- ============================================================================

INSERT INTO speech_assessments (
  session_id,
  audio_file_path,
  audio_duration_seconds,
  audio_quality_score,
  processing_time_ms,
  model_version,
  fluency_score,
  articulation_score,
  prosody_score,
  voice_quality_score,
  speech_rate_wpm,
  risk_score,
  confidence,
  transcription,
  language_detected
) VALUES
  (
    '660e8400-e29b-41d4-a716-446655440001',
    '550e8400-e29b-41d4-a716-446655440001/speech/1705312200_abc123.wav',
    30.5,
    0.92,
    87,
    'whisper-tiny-v1.0',
    0.85,
    0.90,
    0.88,
    0.87,
    145.2,
    0.15,
    0.92,
    'The quick brown fox jumps over the lazy dog. This is a test of speech clarity and fluency.',
    'en'
  ),
  (
    '660e8400-e29b-41d4-a716-446655440002',
    '550e8400-e29b-41d4-a716-446655440002/speech/1705226100_def456.wav',
    28.3,
    0.89,
    92,
    'whisper-tiny-v1.0',
    0.72,
    0.78,
    0.75,
    0.80,
    132.8,
    0.35,
    0.88,
    'The quick brown fox... jumps over the... lazy dog. This is a test of speech.',
    'en'
  );

-- ============================================================================
-- SAMPLE RETINAL ASSESSMENTS
-- ============================================================================

INSERT INTO retinal_assessments (
  session_id,
  image_file_path,
  image_quality_score,
  image_resolution,
  eye_examined,
  processing_time_ms,
  model_version,
  vessel_tortuosity,
  av_ratio,
  cup_disc_ratio,
  vessel_density,
  risk_score,
  confidence
) VALUES
  (
    '660e8400-e29b-41d4-a716-446655440001',
    '550e8400-e29b-41d4-a716-446655440001/retinal/1705312200_ghi789.jpg',
    0.94,
    '1024x1024',
    'right',
    156,
    'efficientnet-b0-v1.0',
    0.35,
    0.72,
    0.28,
    0.65,
    0.25,
    0.91
  ),
  (
    '660e8400-e29b-41d4-a716-446655440002',
    '550e8400-e29b-41d4-a716-446655440002/retinal/1705226100_jkl012.jpg',
    0.87,
    '1024x1024',
    'left',
    168,
    'efficientnet-b0-v1.0',
    0.52,
    0.68,
    0.35,
    0.58,
    0.48,
    0.85
  );

-- ============================================================================
-- SAMPLE MOTOR ASSESSMENTS
-- ============================================================================

INSERT INTO motor_assessments (
  session_id,
  test_type,
  test_duration_seconds,
  hand_tested,
  processing_time_ms,
  model_version,
  tap_frequency,
  rhythm_consistency,
  movement_amplitude,
  tremor_score,
  coordination_score,
  risk_score,
  confidence
) VALUES
  (
    '660e8400-e29b-41d4-a716-446655440001',
    'finger_tapping',
    15,
    'right',
    134,
    'motor-analysis-v1.0',
    4.2,
    0.91,
    0.85,
    0.08,
    0.92,
    0.12,
    0.95
  ),
  (
    '660e8400-e29b-41d4-a716-446655440002',
    'finger_tapping',
    15,
    'right',
    142,
    'motor-analysis-v1.0',
    3.1,
    0.76,
    0.72,
    0.25,
    0.78,
    0.38,
    0.89
  );

-- ============================================================================
-- SAMPLE COGNITIVE ASSESSMENTS
-- ============================================================================

INSERT INTO cognitive_assessments (
  session_id,
  test_battery,
  total_duration_seconds,
  processing_time_ms,
  model_version,
  attention_score,
  memory_score,
  executive_function_score,
  language_score,
  visuospatial_score,
  risk_score,
  confidence
) VALUES
  (
    '660e8400-e29b-41d4-a716-446655440003',
    'MoCA-Digital',
    1200,
    89,
    'cognitive-analysis-v1.0',
    0.88,
    0.85,
    0.90,
    0.92,
    0.87,
    0.15,
    0.93
  );

-- ============================================================================
-- SAMPLE NRI CALCULATIONS
-- ============================================================================

INSERT INTO nri_calculations (
  session_id,
  speech_risk_score,
  retinal_risk_score,
  motor_risk_score,
  cognitive_risk_score,
  fusion_algorithm,
  nri_score,
  nri_category,
  epistemic_uncertainty,
  aleatoric_uncertainty,
  total_uncertainty,
  calculation_time_ms
) VALUES
  (
    '660e8400-e29b-41d4-a716-446655440001',
    0.15,
    0.25,
    0.12,
    NULL,
    'weighted_ensemble',
    0.18,
    'low',
    0.05,
    0.03,
    0.08,
    45
  ),
  (
    '660e8400-e29b-41d4-a716-446655440002',
    0.35,
    0.48,
    0.38,
    NULL,
    'weighted_ensemble',
    0.42,
    'moderate',
    0.08,
    0.06,
    0.14,
    52
  ),
  (
    '660e8400-e29b-41d4-a716-446655440003',
    NULL,
    NULL,
    NULL,
    0.15,
    'single_modality',
    0.15,
    'low',
    0.03,
    0.02,
    0.05,
    23
  );

-- ============================================================================
-- UPDATE USER LAST ASSESSMENT TIMESTAMPS
-- ============================================================================

UPDATE users SET last_assessment = '2024-01-15 10:45:00+00' 
WHERE id = '550e8400-e29b-41d4-a716-446655440001';

UPDATE users SET last_assessment = '2024-01-14 14:35:00+00' 
WHERE id = '550e8400-e29b-41d4-a716-446655440002';

-- ============================================================================
-- SAMPLE JSONB DATA
-- ============================================================================

-- Update some records with sample JSONB data
UPDATE speech_assessments SET 
  pause_patterns = '{"total_pauses": 12, "avg_pause_duration": 0.8, "pause_locations": ["word_boundary", "phrase_boundary"]}',
  fundamental_frequency = '{"mean_f0": 145.2, "std_f0": 12.8, "f0_range": [120.5, 180.3]}',
  spectral_features = '{"mfcc": [1.2, -0.8, 0.5], "spectral_centroid": 2500.0, "spectral_rolloff": 4200.0}'
WHERE session_id = '660e8400-e29b-41d4-a716-446655440001';

UPDATE retinal_assessments SET
  vessel_analysis = '{"artery_count": 8, "vein_count": 12, "bifurcation_points": 15, "crossing_points": 6}',
  optic_disc_analysis = '{"disc_area": 2.1, "cup_area": 0.6, "rim_area": 1.5, "disc_diameter": 1.8}',
  findings = '["mild vessel tortuosity", "normal cup-disc ratio", "no hemorrhages detected"]'
WHERE session_id = '660e8400-e29b-41d4-a716-446655440001';

UPDATE motor_assessments SET
  tap_intervals = '[250, 245, 252, 248, 251, 249, 253, 247, 250, 246]',
  movement_trajectory = '{"amplitude_variation": 0.15, "frequency_stability": 0.91, "coordination_index": 0.88}',
  bradykinesia_indicators = '{"progressive_slowing": false, "amplitude_decrement": 0.08, "hesitation_episodes": 1}'
WHERE session_id = '660e8400-e29b-41d4-a716-446655440001';

UPDATE cognitive_assessments SET
  reaction_times = '[450, 520, 480, 510, 465, 495, 475, 505, 485, 490]',
  accuracy_scores = '{"attention_tasks": 0.95, "memory_tasks": 0.88, "executive_tasks": 0.92}',
  test_results = '{"trail_making_a": 28, "trail_making_b": 65, "digit_span_forward": 7, "digit_span_backward": 5}'
WHERE session_id = '660e8400-e29b-41d4-a716-446655440003';

UPDATE nri_calculations SET
  fusion_weights = '{"speech": 0.3, "retinal": 0.4, "motor": 0.3}',
  feature_importance = '{"vessel_tortuosity": 0.25, "speech_fluency": 0.20, "tap_frequency": 0.18}',
  risk_factors = '["age > 65", "mild vessel changes", "slight speech hesitation"]',
  protective_factors = '["regular exercise", "good cognitive performance", "stable motor function"]'
WHERE session_id = '660e8400-e29b-41d4-a716-446655440001';
