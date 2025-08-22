/**
 * NeuroLens-X TypeScript Type Definitions
 * Comprehensive type system for multi-modal neurological assessment
 */

// Base Assessment Types
export interface BaseAssessment {
  sessionId: string;
  timestamp: Date;
  userId?: string;
  metadata?: Record<string, unknown>;
}

export interface BaseAssessmentResponse {
  sessionId: string;
  processingTime: number;
  timestamp: Date;
  confidence: number;
  status: "completed" | "processing" | "error";
  errorMessage?: string;
}

// Speech Analysis Types
export interface SpeechAnalysisRequest extends BaseAssessment {
  audioFormat?: string;
  duration?: number;
  sampleRate?: number;
  language: string;
}

export interface SpeechAnalysisResponse extends BaseAssessmentResponse {
  riskScore: number;
  features: {
    pausePatterns: number;
    speechRate: number;
    voiceQuality: number;
    articulation: number;
    prosody: number;
  };
  biomarkers: {
    tremor: number;
    breathiness: number;
    roughness: number;
    nasality: number;
  };
  recommendations: string[];
}

// Retinal Analysis Types
export interface RetinalAnalysisRequest extends BaseAssessment {
  imageFormat: string;
  imageQuality: number;
  eyeType: "left" | "right" | "both";
}

export interface RetinalAnalysisResponse extends BaseAssessmentResponse {
  riskScore: number;
  features: {
    vesselTortuosity: number;
    opticDiscRatio: number;
    maculaHealth: number;
    retinalThickness: number;
  };
  biomarkers: {
    microaneurysms: number;
    hemorrhages: number;
    exudates: number;
    cottonWoolSpots: number;
  };
  recommendations: string[];
}

// Motor Assessment Types
export interface MotorAssessmentRequest extends BaseAssessment {
  testType:
    | "finger_tapping"
    | "hand_movement"
    | "pronation"
    | "leg_agility"
    | "gait";
  duration: number;
  deviceType: "smartphone" | "tablet" | "webcam";
}

export interface MotorAssessmentResponse extends BaseAssessmentResponse {
  riskScore: number;
  features: {
    frequency: number;
    amplitude: number;
    rhythm: number;
    coordination: number;
    fatigue: number;
  };
  biomarkers: {
    bradykinesia: number;
    tremor: number;
    rigidity: number;
    dyskinesia: number;
  };
  recommendations: string[];
}

// Cognitive Assessment Types
export interface CognitiveAssessmentRequest extends BaseAssessment {
  testType: "memory" | "attention" | "executive" | "language" | "visuospatial";
  difficulty: "easy" | "medium" | "hard";
  timeLimit: number;
}

export interface CognitiveAssessmentResponse extends BaseAssessmentResponse {
  riskScore: number;
  features: {
    reactionTime: number;
    accuracy: number;
    consistency: number;
    processingSpeed: number;
  };
  biomarkers: {
    memoryImpairment: number;
    attentionDeficit: number;
    executiveFunction: number;
    languageImpairment: number;
  };
  recommendations: string[];
}

// NRI Fusion Types
export interface ModalityContribution {
  modality: string;
  score: number;
  confidence: number;
  weight: number;
  reliability: number;
}

export interface NRIFusionRequest {
  sessionId: string;
  modalities: string[];
  modalityScores: Record<string, number>;
  modalityConfidences?: Record<string, number>;
  userProfile?: Record<string, unknown>;
  fusionMethod: "bayesian" | "weighted" | "ensemble";
}

export interface NRIFusionResponse {
  sessionId: string;
  nriScore: number;
  confidence: number;
  riskCategory: "low" | "moderate" | "high" | "very_high";
  modalityContributions: ModalityContribution[];
  consistencyScore: number;
  uncertainty: number;
  processingTime: number;
  timestamp: Date;
  recommendations: string[];
  followUpActions: string[];
}

// Validation Types
export interface ValidationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  calibrationError: number;
  fairnessMetrics: Record<string, number>;
  reliabilityScore: number;
}

// UI Component Types
export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  variant?: "default" | "glass" | "neural";
  elevation?: 1 | 2 | 3 | 4 | 5;
  interactive?: boolean;
}

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "outline" | "ghost";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  asChild?: boolean;
}

export interface ProgressProps {
  value: number;
  max?: number;
  variant?: "default" | "neural" | "success" | "warning" | "error";
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}

// Assessment Flow Types
export interface AssessmentStep {
  id: string;
  title: string;
  description: string;
  component: React.ComponentType;
  estimatedTime: number;
  required: boolean;
  completed: boolean;
}

export interface AssessmentFlow {
  id: string;
  title: string;
  description: string;
  steps: AssessmentStep[];
  currentStep: number;
  totalSteps: number;
  progress: number;
  estimatedTotalTime: number;
}

// Dashboard Types
export interface DashboardStats {
  totalAssessments: number;
  averageRiskScore: number;
  lastAssessmentDate: Date;
  riskTrend: "improving" | "stable" | "declining";
  modalityScores: Record<string, number>;
}

export interface TestOption {
  id: string;
  title: string;
  description: string;
  icon: string;
  estimatedTime: number;
  processingTime: number;
  difficulty: "easy" | "medium" | "hard";
  available: boolean;
  route: string;
}

// API Response Types
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: Date;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: Date;
}

// Theme Types
export interface ThemeConfig {
  colors: {
    primary: Record<string, string>;
    secondary: Record<string, string>;
    neutral: Record<string, string>;
    glass: Record<string, string>;
    neural: Record<string, string>;
  };
  spacing: Record<string, string>;
  typography: {
    fontFamily: Record<string, string[]>;
    fontSize: Record<string, [string, { lineHeight: string }]>;
  };
  borderRadius: Record<string, string>;
  boxShadow: Record<string, string>;
}

// Performance Monitoring Types
export interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  interactionTime: number;
  coreWebVitals: {
    lcp: number; // Largest Contentful Paint
    fid: number; // First Input Delay
    cls: number; // Cumulative Layout Shift
  };
  memoryUsage: number;
  networkLatency: number;
}
