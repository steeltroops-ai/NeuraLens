/**
 * Retinal Analysis Types for Neuralens
 * 
 * This file defines TypeScript interfaces for the Retinal Analysis ML model
 * implementation, focusing on neurological disorder detection through retinal
 * image analysis including vascular changes and cup-disc ratio assessment.
 * 
 * Key Features:
 * - Vascular change detection for Alzheimer's indicators
 * - Cup-disc ratio analysis for stroke risk assessment
 * - Real-time processing with <150ms latency target
 * - Integration with Neuro-Risk Index (NRI) calculation
 * - EfficientNet-B0 model with ONNX optimization
 */

// Core retinal analysis result structure
export interface RetinalResult {
  /** 
   * Normalized vascular score (0-1, where 1 indicates severe vascular changes)
   * Calculated from vessel tortuosity, narrowing, and density analysis
   */
  vascularScore: number;
  
  /** 
   * Cup-disc ratio (0-1) representing optic disc cupping severity
   * Higher values indicate increased neurological risk
   */
  cupDiscRatio: number;
  
  /** 
   * Model confidence level (0-1) indicating prediction certainty
   * Higher values indicate more reliable results
   */
  confidence: number;
  
  /** 
   * Detailed risk features extracted from retinal analysis
   * Used for clinical interpretation and explainability
   */
  riskFeatures: RetinalRiskFeatures;
  
  /** 
   * Processing metadata for performance monitoring
   */
  metadata: RetinalProcessingMetadata;
}

// Detailed risk features for clinical analysis
export interface RetinalRiskFeatures {
  /** Vessel density measurement (vessels per unit area) */
  vesselDensity: number;
  
  /** Tortuosity index (0-1, higher = more tortuous vessels) */
  tortuosityIndex: number;
  
  /** Average vessel width in pixels */
  averageVesselWidth: number;
  
  /** Arteriovenous ratio (normal ~0.67) */
  arteriovenousRatio: number;
  
  /** Optic disc area in pixels */
  opticDiscArea: number;
  
  /** Optic cup area in pixels */
  opticCupArea: number;
  
  /** Retinal hemorrhage count */
  hemorrhageCount: number;
  
  /** Microaneurysm count */
  microaneurysmCount: number;
  
  /** Hard exudate area percentage */
  hardExudateArea: number;
  
  /** Soft exudate (cotton wool spots) count */
  softExudateCount: number;
  
  /** Overall image quality score (0-1) */
  imageQuality: number;
  
  /** Spatial feature vector from EfficientNet-B0 */
  spatialFeatures: number[];
}

// Processing performance metadata
export interface RetinalProcessingMetadata {
  /** Processing time in milliseconds */
  processingTime: number;
  
  /** Image dimensions (width x height) */
  imageDimensions: {
    width: number;
    height: number;
  };
  
  /** Original image size in bytes */
  imageSize: number;
  
  /** Model version used for inference */
  modelVersion: string;
  
  /** Preprocessing steps applied */
  preprocessingSteps: string[];
  
  /** Timestamp of analysis */
  timestamp: Date;
  
  /** GPU acceleration used */
  gpuAccelerated: boolean;
}

// Image input configuration
export interface RetinalImageConfig {
  /** Target image size for model input (default: 224x224) */
  targetSize: {
    width: number;
    height: number;
  };
  
  /** Image normalization parameters */
  normalization: {
    mean: [number, number, number];
    std: [number, number, number];
  };
  
  /** Supported image formats */
  supportedFormats: string[];
  
  /** Maximum file size in bytes */
  maxFileSize: number;
  
  /** Minimum image resolution */
  minResolution: {
    width: number;
    height: number;
  };
}

// Retinal processor configuration
export interface RetinalProcessorConfig {
  /** Path to ONNX model file */
  modelPath: string;
  
  /** Image processing configuration */
  imageConfig: RetinalImageConfig;
  
  /** Enable debug logging */
  debug: boolean;
  
  /** Maximum processing timeout in milliseconds */
  timeout: number;
  
  /** Enable GPU acceleration if available */
  useGPU: boolean;
  
  /** Batch processing size */
  batchSize: number;
}

// Image upload and processing state
export interface RetinalProcessingState {
  /** Current processing status */
  status: 'idle' | 'uploading' | 'preprocessing' | 'analyzing' | 'complete' | 'error';
  
  /** Processing progress (0-1) */
  progress: number;
  
  /** Current processing step description */
  currentStep: string;
  
  /** Error message if status is 'error' */
  error?: string;
  
  /** Uploaded image data URL */
  imageDataUrl?: string;
  
  /** Image file information */
  imageInfo?: {
    name: string;
    size: number;
    type: string;
    lastModified: number;
  };
}

// API request/response types for backend integration
export interface RetinalAnalysisRequest {
  /** Base64 encoded image data */
  imageData: string;
  
  /** Image format (jpeg, png, etc.) */
  format: string;
  
  /** Processing configuration */
  config: Partial<RetinalProcessorConfig>;
  
  /** User session ID for caching */
  sessionId?: string;
  
  /** Additional metadata */
  metadata?: {
    patientId?: string;
    captureDate?: string;
    deviceInfo?: string;
  };
}

export interface RetinalAnalysisResponse {
  /** Analysis results */
  result: RetinalResult;
  
  /** Processing success status */
  success: boolean;
  
  /** Error message if processing failed */
  error?: string;
  
  /** Cache key for result storage */
  cacheKey?: string;
  
  /** NRI contribution score */
  nriContribution?: number;
}

// Validation and testing types
export interface RetinalValidationMetrics {
  /** Model precision on test dataset */
  precision: number;
  
  /** Model recall */
  recall: number;
  
  /** F1 score */
  f1Score: number;
  
  /** Area under ROC curve */
  aucScore: number;
  
  /** Average processing latency in ms */
  averageLatency: number;
  
  /** Bias metrics across demographics */
  biasMetrics: {
    ageGroups: Record<string, number>;
    ethnicityGroups: Record<string, number>;
    genderGroups: Record<string, number>;
  };
  
  /** Confusion matrix */
  confusionMatrix: number[][];
}

// Demo and testing data structure
export interface RetinalDemoSample {
  /** Sample identifier */
  id: string;
  
  /** Image file path */
  imagePath: string;
  
  /** Expected NRI score (0-100) */
  expectedNRI: number;
  
  /** Ground truth labels for validation */
  groundTruth: {
    vascularScore: number;
    cupDiscRatio: number;
    hasNeurologicalIndicators: boolean;
    condition?: 'healthy' | 'alzheimers' | 'stroke_risk' | 'diabetic_retinopathy' | 'other';
    severity?: 'mild' | 'moderate' | 'severe';
  };
  
  /** Sample metadata */
  metadata: {
    patientAge: number;
    patientGender: 'male' | 'female' | 'other';
    ethnicity?: string;
    imageQuality: number;
    captureDevice: string;
  };
}

// Error types for robust error handling
export class RetinalAnalysisError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message);
    this.name = 'RetinalAnalysisError';
  }
}

// Constants for retinal analysis
export const RETINAL_ANALYSIS_CONSTANTS = {
  // Model configuration
  MODEL_INPUT_SIZE: { width: 224, height: 224 },
  SPATIAL_FEATURES_SIZE: 1280, // EfficientNet-B0 feature size
  TARGET_SAMPLE_RATE: 16000, // For consistency with speech analysis
  
  // Performance targets
  MAX_LATENCY: 150, // milliseconds
  MIN_PRECISION: 0.85, // 85%
  MIN_CONFIDENCE: 0.70, // 70%
  
  // Image quality thresholds
  MIN_IMAGE_SIZE: { width: 224, height: 224 },
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  SUPPORTED_FORMATS: ['image/jpeg', 'image/png', 'image/bmp'],
  
  // Clinical normal ranges
  NORMAL_CUP_DISC_RATIO: { min: 0.1, max: 0.4 },
  NORMAL_AV_RATIO: { min: 0.6, max: 0.8 },
  NORMAL_VESSEL_DENSITY: { min: 0.15, max: 0.25 },
  
  // Risk thresholds
  HIGH_RISK_VASCULAR_SCORE: 0.7,
  HIGH_RISK_CUP_DISC_RATIO: 0.6,
  CRITICAL_RISK_THRESHOLD: 0.8,
} as const;

// Export utility type for component props
export type RetinalAnalysisProps = {
  onResult: (result: RetinalResult) => void;
  onError: (error: RetinalAnalysisError) => void;
  onStateChange: (state: RetinalProcessingState) => void;
  config?: Partial<RetinalProcessorConfig>;
  className?: string;
  allowCamera?: boolean;
  allowUpload?: boolean;
};

// Image processing utility types
export interface ImageProcessingUtils {
  resizeImage: (imageData: ImageData, targetWidth: number, targetHeight: number) => ImageData;
  normalizePixels: (imageData: ImageData, mean: number[], std: number[]) => Float32Array;
  validateImageFormat: (file: File) => boolean;
  extractImageFeatures: (imageData: ImageData) => Promise<number[]>;
}

// Camera capture configuration
export interface CameraConfig {
  /** Preferred camera resolution */
  resolution: {
    width: number;
    height: number;
  };
  
  /** Camera facing mode */
  facingMode: 'user' | 'environment';
  
  /** Enable flash if available */
  enableFlash: boolean;
  
  /** Auto-focus mode */
  autoFocus: boolean;
  
  /** Image capture format */
  captureFormat: 'jpeg' | 'png';
  
  /** JPEG quality (0-1) */
  jpegQuality: number;
}

export default RetinalAnalysisError;
