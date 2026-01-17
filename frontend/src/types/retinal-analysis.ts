/**
 * TypeScript Type Definitions for Retinal Analysis Pipeline
 * 
 * Comprehensive types for:
 * - Biomarker data structures
 * - Risk assessment
 * - API request/response types
 * - Visualization types
 * - Patient history and trends
 * - NRI integration types
 * 
 * @module types/retinal-analysis
 */

// ============================================================================
// Risk Category Types
// ============================================================================

/**
 * Risk category levels for neurological assessment
 */
export type RiskCategory = 'minimal' | 'low' | 'moderate' | 'elevated' | 'high' | 'critical';

/**
 * Risk category color mapping for UI display
 */
export const RISK_CATEGORY_COLORS: Record<RiskCategory, string> = {
  minimal: '#22c55e',   // Green
  low: '#84cc16',       // Lime
  moderate: '#eab308',  // Yellow
  elevated: '#f97316',  // Orange
  high: '#ef4444',      // Red
  critical: '#991b1b',  // Dark red
};

/**
 * Risk category descriptions for patient-facing content
 */
export const RISK_CATEGORY_DESCRIPTIONS: Record<RiskCategory, string> = {
  minimal: 'Your neurological risk indicators are within normal limits.',
  low: 'Minor variations detected, generally not concerning.',
  moderate: 'Some patterns may benefit from monitoring.',
  elevated: 'Patterns warrant attention. Consider specialist consultation.',
  high: 'Significant patterns require prompt medical attention.',
  critical: 'Patterns of immediate concern. Urgent consultation recommended.',
};

// ============================================================================
// Biomarker Interfaces
// ============================================================================

/**
 * Vessel biomarker measurements from retinal image analysis
 */
export interface VesselBiomarkers {
  /** Percentage of image area covered by blood vessels */
  density_percentage: number;
  /** Index measuring vessel curvature/twisting */
  tortuosity_index: number;
  /** Artery to vein width ratio */
  avr_ratio: number;
  /** Measure of vessel branching complexity */
  branching_coefficient: number;
  /** Model confidence score (0-1) */
  confidence: number;
}

/**
 * Optic disc measurements
 */
export interface OpticDiscBiomarkers {
  /** Ratio of cup to disc diameter */
  cup_to_disc_ratio: number;
  /** Optic disc area in square millimeters */
  disc_area_mm2: number;
  /** Neuroretinal rim area in square millimeters */
  rim_area_mm2: number;
  /** Model confidence score (0-1) */
  confidence: number;
}

/**
 * Macular region measurements
 */
export interface MacularBiomarkers {
  /** Central macular thickness in micrometers */
  thickness_um: number;
  /** Macular volume in cubic millimeters */
  volume_mm3: number;
  /** Model confidence score (0-1) */
  confidence: number;
}

/**
 * Amyloid-beta deposit indicators
 */
export interface AmyloidBetaIndicators {
  /** Probability score for amyloid-beta presence (0-1) */
  presence_score: number;
  /** Pattern classification of deposits */
  distribution_pattern: 'normal' | 'focal' | 'diffuse' | 'perivascular';
  /** Model confidence score (0-1) */
  confidence: number;
}

/**
 * Complete biomarker set from retinal analysis
 */
export interface RetinalBiomarkers {
  vessels: VesselBiomarkers;
  optic_disc: OpticDiscBiomarkers;
  macula: MacularBiomarkers;
  amyloid_beta: AmyloidBetaIndicators;
}

// ============================================================================
// Risk Assessment Types
// ============================================================================

/**
 * Complete risk assessment with contributing factors
 */
export interface RiskAssessment {
  /** Overall risk score (0-100) */
  risk_score: number;
  /** Categorized risk level */
  risk_category: RiskCategory;
  /** 95% confidence interval [lower, upper] */
  confidence_interval: [number, number];
  /** Breakdown of factors contributing to risk score */
  contributing_factors: Record<string, number>;
}

// ============================================================================
// Analysis Result Types
// ============================================================================

/**
 * Complete result from retinal analysis endpoint
 */
export interface RetinalAnalysisResult {
  /** Unique assessment identifier */
  assessment_id: string;
  /** Patient identifier */
  patient_id: string;
  /** All extracted biomarkers */
  biomarkers: RetinalBiomarkers;
  /** Risk assessment with category and factors */
  risk_assessment: RiskAssessment;
  /** Image quality score (0-100) */
  quality_score: number;
  /** URL to attention heatmap visualization */
  heatmap_url: string;
  /** URL to vessel segmentation overlay */
  segmentation_url: string;
  /** Analysis timestamp */
  created_at: string;
  /** ML model version used */
  model_version: string;
  /** Processing time in milliseconds */
  processing_time_ms: number;
}

/**
 * Image validation result before full analysis
 */
export interface ImageValidationResult {
  /** Whether image passed validation */
  is_valid: boolean;
  /** Quality score (0-100) */
  quality_score: number;
  /** List of validation issues */
  issues: string[];
  /** Recommendations for better image quality */
  recommendations: string[];
  /** Signal-to-noise ratio in decibels */
  snr_db: number;
  /** Whether optic disc was detected */
  has_optic_disc: boolean;
  /** Whether macula was detected */
  has_macula: boolean;
  /** Focus quality score (0-1) */
  focus_score?: number;
  /** Glare percentage (0-1) */
  glare_percentage?: number;
}

// ============================================================================
// Request Types
// ============================================================================

/**
 * Options for retinal analysis request
 */
export interface RetinalAnalysisOptions {
  /** Include visualization generation */
  include_visualizations?: boolean;
  /** Priority level (0-10) */
  priority?: number;
  /** Custom patient metadata */
  metadata?: Record<string, string>;
}

/**
 * Options for image validation request
 */
export interface ImageValidationOptions {
  /** Run detailed quality analysis */
  detailed?: boolean;
  /** Check for anatomical features */
  check_anatomy?: boolean;
}

// ============================================================================
// Patient History Types
// ============================================================================

/**
 * Single assessment item in patient history
 */
export interface PatientHistoryItem {
  /** Assessment identifier */
  assessment_id: string;
  /** When the assessment was created */
  created_at: string;
  /** Risk score at time of assessment */
  risk_score: number;
  /** Risk category at time of assessment */
  risk_category: RiskCategory;
  /** Image quality score */
  quality_score: number;
}

/**
 * Patient assessment history response
 */
export interface PatientHistoryResponse {
  /** Patient identifier */
  patient_id: string;
  /** List of historical assessments */
  assessments: PatientHistoryItem[];
  /** Total number of assessments */
  total_count: number;
  /** Whether there are more results */
  has_more: boolean;
}

// ============================================================================
// Trend Analysis Types
// ============================================================================

/**
 * Single data point in a trend
 */
export interface TrendDataPoint {
  /** Date of measurement */
  date: string;
  /** Value at that date */
  value: number;
}

/**
 * Trend direction indicator
 */
export type TrendDirection = 'improving' | 'stable' | 'declining';

/**
 * Biomarker trend analysis response
 */
export interface TrendAnalysisResponse {
  /** Patient identifier */
  patient_id: string;
  /** Biomarker being tracked */
  biomarker: string;
  /** Historical data points */
  data_points: TrendDataPoint[];
  /** Overall trend direction */
  trend_direction: TrendDirection;
}

// ============================================================================
// Visualization Types
// ============================================================================

/**
 * Available visualization types
 */
export type VisualizationType = 'heatmap' | 'segmentation' | 'gauge' | 'measurements';

/**
 * Visualization request options
 */
export interface VisualizationOptions {
  /** Type of visualization */
  type: VisualizationType;
  /** Image width */
  width?: number;
  /** Image height */
  height?: number;
  /** Overlay opacity (0-1) */
  opacity?: number;
}

// ============================================================================
// NRI Integration Types
// ============================================================================

/**
 * NRI contribution status
 */
export type NRIStatus = 'success' | 'pending' | 'failed' | 'partial' | 'standalone';

/**
 * NRI contribution data from retinal analysis
 */
export interface NRIContribution {
  /** Base weight percentage (30%) */
  weight_percentage: number;
  /** Actual contribution score */
  contribution_score: number;
  /** Confidence in contribution */
  confidence: number;
  /** Integration status */
  status: NRIStatus;
}

/**
 * NRI dashboard data with retinal contribution
 */
export interface NRIDashboardData {
  /** Retinal analysis summary */
  retinal_analysis: {
    assessment_id: string;
    risk_score: number;
    risk_category: RiskCategory;
    quality_score: number;
    created_at: string;
  };
  /** NRI contribution details */
  nri_contribution: NRIContribution;
  /** Biomarker summary for quick display */
  biomarker_summary: Record<string, {
    value: number;
    status: 'normal' | 'low' | 'high';
  }>;
  /** Total NRI score if available */
  nri_total?: {
    score: number;
    category: RiskCategory;
    retinal_contribution_percentage: number;
  };
}

// ============================================================================
// Report Types
// ============================================================================

/**
 * Report generation options
 */
export interface ReportOptions {
  /** Patient full name for report */
  patient_name?: string;
  /** Patient date of birth */
  patient_dob?: string;
  /** Healthcare provider name */
  provider_name?: string;
  /** Provider NPI number */
  provider_npi?: string;
}

// ============================================================================
// Queue Status Types (for high-load scenarios)
// ============================================================================

/**
 * Request queue status
 */
export interface QueueStatus {
  /** Current status */
  status: 'queued' | 'processing' | 'completed' | 'failed';
  /** Position in queue (if queued) */
  position?: number;
  /** Estimated wait time in seconds */
  estimated_wait?: number;
  /** When processing started */
  started_at?: string;
  /** When processing completed */
  completed_at?: string;
  /** Processing time if completed */
  processing_time_ms?: number;
}

// ============================================================================
// API Error Types
// ============================================================================

/**
 * Structured API error response
 */
export interface APIError {
  /** Error message */
  message: string;
  /** Error code */
  code?: string;
  /** Detailed validation issues */
  issues?: string[];
  /** Recommendations to fix the error */
  recommendations?: string[];
}

// ============================================================================
// Component Props Types
// ============================================================================

/**
 * Props for image upload components
 */
export interface ImageUploadProps {
  /** Called when file is selected */
  onFileSelect: (file: File) => void;
  /** Called on validation result */
  onValidationResult?: (result: ImageValidationResult) => void;
  /** Accepted file types */
  acceptedTypes?: string[];
  /** Maximum file size in bytes */
  maxSizeBytes?: number;
  /** Whether upload is disabled */
  disabled?: boolean;
  /** Whether currently uploading */
  isUploading?: boolean;
  /** Upload progress (0-100) */
  progress?: number;
}

/**
 * Props for results display components
 */
export interface ResultsDisplayProps {
  /** Analysis result to display */
  result: RetinalAnalysisResult;
  /** Whether to show visualizations */
  showVisualizations?: boolean;
  /** Whether to enable report download */
  enableReport?: boolean;
  /** Called when report is downloaded */
  onReportDownload?: (assessmentId: string) => void;
}

/**
 * Props for biomarker card components
 */
export interface BiomarkerCardProps {
  /** Biomarker name */
  name: string;
  /** Current value */
  value: number;
  /** Unit of measurement */
  unit: string;
  /** Normal range [min, max] */
  normalRange?: [number, number];
  /** Confidence score */
  confidence?: number;
  /** Whether value is within normal range */
  isNormal?: boolean;
}

/**
 * Props for risk gauge components
 */
export interface RiskGaugeProps {
  /** Risk score (0-100) */
  score: number;
  /** Risk category */
  category: RiskCategory;
  /** Confidence interval */
  confidenceInterval?: [number, number];
  /** Size of gauge */
  size?: 'sm' | 'md' | 'lg';
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Reference ranges for biomarkers
 */
export const BIOMARKER_REFERENCE_RANGES: Record<string, { min: number; max: number; unit: string }> = {
  vessel_density: { min: 4.0, max: 7.0, unit: '%' },
  tortuosity_index: { min: 0.8, max: 1.3, unit: '' },
  avr_ratio: { min: 0.6, max: 0.8, unit: '' },
  cup_to_disc_ratio: { min: 0.3, max: 0.5, unit: '' },
  disc_area: { min: 2.0, max: 3.5, unit: 'mm²' },
  macular_thickness: { min: 250, max: 320, unit: 'μm' },
  amyloid_presence: { min: 0.0, max: 0.2, unit: '' },
};

/**
 * Check if a biomarker value is within normal range
 */
export function isWithinNormalRange(
  biomarker: keyof typeof BIOMARKER_REFERENCE_RANGES,
  value: number
): boolean {
  const range = BIOMARKER_REFERENCE_RANGES[biomarker];
  if (!range) return true;
  return value >= range.min && value <= range.max;
}

/**
 * Get risk category color for styling
 */
export function getRiskCategoryColor(category: RiskCategory): string {
  return RISK_CATEGORY_COLORS[category] || '#6b7280';
}

/**
 * Format confidence as percentage string
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(0)}%`;
}

// ============================================================================
// Legacy/API Route Types (for backwards compatibility)
// ============================================================================

/**
 * Risk features extracted from retinal image analysis
 */
export interface RetinalRiskFeatures {
  vesselDensity: number;
  tortuosityIndex: number;
  averageVesselWidth: number;
  arteriovenousRatio: number;
  opticDiscArea: number;
  opticCupArea: number;
  hemorrhageCount: number;
  microaneurysmCount: number;
  hardExudateArea: number;
  softExudateCount: number;
  imageQuality: number;
  spatialFeatures: number[];
}

/**
 * Metadata for retinal analysis result
 */
export interface RetinalResultMetadata {
  processingTime: number;
  imageDimensions: { width: number; height: number };
  imageSize: number;
  modelVersion: string;
  preprocessingSteps: string[];
  timestamp: Date;
  gpuAccelerated: boolean;
}

/**
 * Complete retinal analysis result (legacy format)
 */
export interface RetinalResult {
  vascularScore: number;
  cupDiscRatio: number;
  confidence: number;
  riskFeatures: RetinalRiskFeatures;
  metadata: RetinalResultMetadata;
}

/**
 * API response for retinal analysis
 */
export interface RetinalAnalysisResponse {
  result: RetinalResult;
  success: boolean;
  cacheKey: string;
  nriContribution: number;
}
