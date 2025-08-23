/**
 * Speech Analysis Types for Neuralens
 * 
 * This file defines TypeScript interfaces for the Speech Analysis ML model
 * implementation, focusing on neurological disorder detection through voice patterns.
 * 
 * Key Features:
 * - Fluency detection for Parkinson's speech patterns
 * - Vocal tremor analysis for neurological indicators
 * - Pause pattern recognition for cognitive decline assessment
 * - Real-time processing with <100ms latency target
 * - Integration with Neuro-Risk Index (NRI) calculation
 */

// Core speech analysis result structure
export interface SpeechResult {
  /** 
   * Normalized fluency score (0-1, where 1 is perfect fluency)
   * Calculated from pause patterns, hesitations, and speech rhythm
   */
  fluencyScore: number;
  
  /** 
   * Model confidence level (0-1) indicating prediction certainty
   * Higher values indicate more reliable results
   */
  confidence: number;
  
  /** 
   * Detailed biomarkers extracted from speech analysis
   * Used for clinical interpretation and explainability
   */
  biomarkers: SpeechBiomarkers;
  
  /** 
   * Processing metadata for performance monitoring
   */
  metadata: ProcessingMetadata;
}

// Detailed biomarkers for clinical analysis
export interface SpeechBiomarkers {
  /** Average pause duration in milliseconds */
  pauseDuration: number;
  
  /** Number of pauses per minute */
  pauseFrequency: number;
  
  /** Vocal tremor frequency in Hz (0 if no tremor detected) */
  tremorFrequency: number;
  
  /** Speech rate in words per minute */
  speechRate: number;
  
  /** Pitch variation coefficient (higher = more unstable) */
  pitchVariation: number;
  
  /** Voice quality metrics */
  voiceQuality: {
    /** Jitter (pitch perturbation) percentage */
    jitter: number;
    /** Shimmer (amplitude perturbation) percentage */
    shimmer: number;
    /** Harmonics-to-noise ratio in dB */
    hnr: number;
  };
  
  /** MFCC feature vector (13 coefficients) */
  mfccFeatures: number[];
}

// Processing performance metadata
export interface ProcessingMetadata {
  /** Processing time in milliseconds */
  processingTime: number;
  
  /** Audio sample duration in seconds */
  audioDuration: number;
  
  /** Sample rate of processed audio */
  sampleRate: number;
  
  /** Model version used for inference */
  modelVersion: string;
  
  /** Timestamp of analysis */
  timestamp: Date;
}

// Audio input configuration
export interface AudioConfig {
  /** Target sample rate for processing (default: 16000 Hz) */
  sampleRate: number;
  
  /** Audio duration in seconds (default: 30s) */
  duration: number;
  
  /** Minimum audio level threshold for valid recording */
  minAudioLevel: number;
  
  /** Enable noise reduction preprocessing */
  noiseReduction: boolean;
}

// Speech processor configuration
export interface SpeechProcessorConfig {
  /** Path to ONNX model file */
  modelPath: string;
  
  /** Audio processing configuration */
  audioConfig: AudioConfig;
  
  /** Enable debug logging */
  debug: boolean;
  
  /** Maximum processing timeout in milliseconds */
  timeout: number;
}

// Recording state management
export interface RecordingState {
  /** Current recording status */
  status: 'idle' | 'recording' | 'processing' | 'complete' | 'error';
  
  /** Recording progress (0-1) */
  progress: number;
  
  /** Current audio level for visual feedback */
  audioLevel: number;
  
  /** Error message if status is 'error' */
  error?: string;
  
  /** Recorded audio duration in seconds */
  recordedDuration: number;
}

// API request/response types for backend integration
export interface SpeechAnalysisRequest {
  /** Base64 encoded audio data */
  audioData: string;
  
  /** Audio format (wav, mp3, etc.) */
  format: string;
  
  /** Processing configuration */
  config: Partial<SpeechProcessorConfig>;
  
  /** User session ID for caching */
  sessionId?: string;
}

export interface SpeechAnalysisResponse {
  /** Analysis results */
  result: SpeechResult;
  
  /** Processing success status */
  success: boolean;
  
  /** Error message if processing failed */
  error?: string;
  
  /** Cache key for result storage */
  cacheKey?: string;
}

// Validation and testing types
export interface ValidationMetrics {
  /** Model accuracy on test dataset */
  accuracy: number;
  
  /** Precision score */
  precision: number;
  
  /** Recall score */
  recall: number;
  
  /** F1 score */
  f1Score: number;
  
  /** Average processing latency in ms */
  averageLatency: number;
  
  /** Bias metrics across demographics */
  biasMetrics: {
    ageGroups: Record<string, number>;
    genderGroups: Record<string, number>;
  };
}

// Demo and testing data structure
export interface DemoAudioSample {
  /** Sample identifier */
  id: string;
  
  /** Audio file path */
  filePath: string;
  
  /** Expected NRI score (0-100) */
  expectedNRI: number;
  
  /** Ground truth labels for validation */
  groundTruth: {
    fluencyScore: number;
    hasNeurologicalIndicators: boolean;
    condition?: 'healthy' | 'parkinsons' | 'alzheimers' | 'other';
  };
  
  /** Sample metadata */
  metadata: {
    duration: number;
    sampleRate: number;
    speakerAge: number;
    speakerGender: 'male' | 'female' | 'other';
  };
}

// Error types for robust error handling
export class SpeechAnalysisError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message);
    this.name = 'SpeechAnalysisError';
  }
}

// Constants for speech analysis
export const SPEECH_ANALYSIS_CONSTANTS = {
  // Model configuration
  MODEL_INPUT_SIZE: 13, // MFCC coefficients
  TARGET_SAMPLE_RATE: 16000, // Hz
  DEFAULT_DURATION: 30, // seconds
  
  // Performance targets
  MAX_LATENCY: 100, // milliseconds
  MIN_ACCURACY: 0.90, // 90%
  MIN_CONFIDENCE: 0.70, // 70%
  
  // Audio quality thresholds
  MIN_AUDIO_LEVEL: 0.01,
  MAX_NOISE_LEVEL: 0.05,
  
  // Biomarker normal ranges
  NORMAL_SPEECH_RATE: { min: 150, max: 200 }, // words per minute
  NORMAL_PAUSE_DURATION: { min: 200, max: 800 }, // milliseconds
  NORMAL_PITCH_VARIATION: { min: 0.02, max: 0.08 }, // coefficient
} as const;

// Export utility type for component props
export type SpeechAnalysisProps = {
  onResult: (result: SpeechResult) => void;
  onError: (error: SpeechAnalysisError) => void;
  onStateChange: (state: RecordingState) => void;
  config?: Partial<SpeechProcessorConfig>;
  className?: string;
};
