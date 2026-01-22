/**
 * TypeScript interfaces matching backend Pydantic schemas
 * Ensures type safety between frontend and backend
 */

// Patient Types
export interface Patient {
  id: string;
  full_name: string;
  phone_number: string;
  email?: string;
  date_of_birth: string | null;
  gender?: string;
  address?: string;
  medical_notes?: string;
  created_at: string;
}

// Base Assessment Types
export interface BaseAssessmentResponse {
  session_id: string;
  processing_time: number;
  timestamp: string;
  confidence: number;
  risk_score: number;
  recommendations: string[];
}

// Speech Analysis Types
export interface SpeechBiomarkers {
  fluency_score: number;
  pause_pattern: number;
  voice_tremor: number;
  articulation_clarity: number;
  prosody_variation: number;
  speaking_rate: number;
  pause_frequency: number;
}

export interface SpeechAnalysisResponse extends BaseAssessmentResponse {
  biomarkers: SpeechBiomarkers;
  quality_score: number;
  file_info: {
    duration: number;
    sample_rate: number;
    channels: number;
  };
}

export interface SpeechAnalysisRequest {
  session_id: string;
  patient_id?: string;
  audio_file: File;
  quality_threshold?: number;
}

// Retinal Analysis Types
export interface RetinalBiomarkers {
  vessel_density: number;
  vessel_tortuosity: number;
  cup_disc_ratio: number;
  av_ratio: number;
  hemorrhage_count: number;
  exudate_area: number;
  microaneurysm_count: number;
}

export interface RetinalAnalysisResponse extends BaseAssessmentResponse {
  biomarkers: RetinalBiomarkers;
  quality_score: number;
  detected_conditions: string[];
  image_info: {
    width: number;
    height: number;
    format: string;
    file_size: number;
  };
}

export interface RetinalAnalysisRequest {
  session_id: string;
  patient_id?: string;
  image_file: File;
  quality_threshold?: number;
}

// Motor Assessment Types
export interface MotorBiomarkers {
  movement_frequency: number;
  amplitude_variation: number;
  coordination_index: number;
  tremor_severity: number;
  fatigue_index: number;
  asymmetry_score: number;
}

export interface MotorAssessmentResponse extends BaseAssessmentResponse {
  biomarkers: MotorBiomarkers;
  assessment_type: string;
  movement_quality: string;
}

export interface MotorAssessmentRequest {
  session_id: string;
  patient_id?: string;
  sensor_data: {
    accelerometer?: Array<{ x: number; y: number; z: number }>;
    gyroscope?: Array<{ x: number; y: number; z: number }>;
    position?: Array<{ x: number; y: number }>;
  };
  assessment_type: "tremor" | "finger_tapping" | "gait" | "balance";
}

// Cognitive Assessment Types
export interface CognitiveBiomarkers {
  memory_score: number;
  attention_score: number;
  executive_score: number;
  language_score: number;
  processing_speed: number;
  cognitive_flexibility: number;
}

export interface CognitiveAssessmentResponse extends BaseAssessmentResponse {
  biomarkers: CognitiveBiomarkers;
  overall_score: number;
  test_battery: string[];
  domain_scores: Record<string, number>;
}

export interface CognitiveAssessmentRequest {
  session_id: string;
  patient_id?: string;
  test_results: {
    response_times?: number[];
    accuracy?: number[];
    memory?: Record<string, number>;
    attention?: Record<string, number>;
    executive?: Record<string, number>;
    task_switching?: {
      repeat_trials: number[];
      switch_trials: number[];
      switch_accuracy: number;
    };
  };
  test_battery: string[];
  difficulty_level: "easy" | "standard" | "hard";
}

// NRI Fusion Types
export interface ModalityContribution {
  modality: string;
  weight: number;
  confidence: number;
  risk_score: number;
}

export interface NRIFusionResponse {
  session_id: string;
  nri_score: number;
  risk_category: "low" | "moderate" | "high";
  confidence: number;
  uncertainty: number;
  consistency_score: number;
  modality_contributions: ModalityContribution[];
  processing_time: number;
  timestamp: string;
  recommendations: string[];
  follow_up_actions: string[];
}

export interface NRIFusionRequest {
  session_id: string;
  modality_results: {
    speech?: SpeechAnalysisResponse;
    retinal?: RetinalAnalysisResponse;
    motor?: MotorAssessmentResponse;
    cognitive?: CognitiveAssessmentResponse;
  };
  fusion_method?: "bayesian" | "weighted_average" | "ensemble";
  uncertainty_quantification?: boolean;
}

// Validation Types
export interface ValidationMetrics {
  accuracy: number;
  sensitivity: number;
  specificity: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_score: number;
  confusion_matrix: number[][];
}

export interface ValidationResponse {
  study_id: string;
  modality: string;
  metrics: ValidationMetrics;
  sample_size: number;
  validation_date: string;
  confidence_interval: {
    lower: number;
    upper: number;
  };
}

// Health Check Types
export interface HealthCheckResponse {
  status: "healthy" | "unhealthy" | "warning";
  service: string;
  version: string;
  environment: string;
  timestamp?: string;
  components?: Record<string, string>;
}

// API Status Types
export interface ApiStatusResponse {
  status: "operational" | "degraded" | "down";
  version: string;
  endpoints: Record<string, string>;
  features: Record<string, boolean>;
}

// Error Types
export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: string;
  request_id?: string;
}

// Service Info Types
export interface ServiceInfo {
  service: string;
  version: string;
  description: string;
  capabilities?: Record<string, any>;
  requirements?: Record<string, any>;
  performance?: {
    target_latency: string;
    accuracy: number;
  };
}

// Assessment Progress Types
export interface AssessmentProgress {
  session_id: string;
  current_step: string;
  completed_steps: string[];
  total_steps: number;
  progress_percentage: number;
  estimated_time_remaining?: number;
}

// Complete Assessment Result
export interface CompleteAssessmentResult {
  session_id: string;
  speech_result?: SpeechAnalysisResponse;
  retinal_result?: RetinalAnalysisResponse;
  motor_result?: MotorAssessmentResponse;
  cognitive_result?: CognitiveAssessmentResponse;
  nri_result: NRIFusionResponse;
  overall_risk_category: "low" | "moderate" | "high";
  completion_time: string;
  total_processing_time: number;
}

// Request/Response wrapper types
export interface StandardRequest<T = any> {
  data: T;
  metadata?: {
    client_version?: string;
    user_agent?: string;
    timestamp?: string;
  };
}

export interface StandardResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiErrorResponse["error"];
  metadata: {
    timestamp: string;
    processing_time?: number;
    request_id?: string;
    api_version?: string;
  };
}

// Export utility types
export type AssessmentType = "speech" | "retinal" | "motor" | "cognitive";
export type RiskCategory = "low" | "moderate" | "high";
export type AssessmentStatus =
  | "pending"
  | "processing"
  | "completed"
  | "failed";

// Type guards
export const isSpeechAnalysisResponse = (
  obj: any,
): obj is SpeechAnalysisResponse => {
  return (
    obj &&
    typeof obj.biomarkers === "object" &&
    "fluency_score" in obj.biomarkers
  );
};

export const isRetinalAnalysisResponse = (
  obj: any,
): obj is RetinalAnalysisResponse => {
  return (
    obj &&
    typeof obj.biomarkers === "object" &&
    "vessel_density" in obj.biomarkers
  );
};

export const isMotorAssessmentResponse = (
  obj: any,
): obj is MotorAssessmentResponse => {
  return (
    obj &&
    typeof obj.biomarkers === "object" &&
    "movement_frequency" in obj.biomarkers
  );
};

export const isCognitiveAssessmentResponse = (
  obj: any,
): obj is CognitiveAssessmentResponse => {
  return (
    obj &&
    typeof obj.biomarkers === "object" &&
    "memory_score" in obj.biomarkers
  );
};

export const isNRIFusionResponse = (obj: any): obj is NRIFusionResponse => {
  return obj && typeof obj.nri_score === "number" && "risk_category" in obj;
};
