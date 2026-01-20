/**
 * Radiology Analysis Types
 *
 * TypeScript interfaces matching the backend response contract.
 */

// Primary finding from analysis
export interface PrimaryFinding {
  condition: string;
  probability: number;
  severity: "normal" | "minimal" | "low" | "moderate" | "high" | "critical";
  description?: string;
}

// Individual finding with clinical details
export interface Finding {
  id?: string;
  condition: string;
  probability: number;
  severity: string;
  confidence?: number;
  location?: string;
  description: string;
  radiological_features?: string[];
  urgency?: string;
  is_critical: boolean;
}

// Quality assessment metrics
export interface QualityMetrics {
  overall_quality: "good" | "adequate" | "poor";
  quality_score: number;
  resolution?: string;
  resolution_adequate: boolean;
  positioning?: string;
  exposure?: string;
  contrast?: number;
  issues: string[];
  usable: boolean;
}

// Pipeline stage result
export interface StageResult {
  stage: string;
  status: "success" | "failed";
  time_ms: number;
  error_code?: string;
}

// Receipt confirmation
export interface Receipt {
  acknowledged: boolean;
  modality_received: string;
  body_region?: string;
  is_volumetric: boolean;
  file_hash?: string;
  file_size_mb: number;
}

// Main analysis response
export interface RadiologyAnalysisResponse {
  success: boolean;
  request_id?: string;
  timestamp: string;
  processing_time_ms: number;

  // Receipt
  receipt?: Receipt;

  // Stage tracking
  stages_completed: StageResult[];

  // Clinical results
  primary_finding?: PrimaryFinding;
  all_predictions?: Record<string, number>;
  findings: Finding[];

  // Risk assessment
  risk_level?: "normal" | "low" | "moderate" | "high" | "critical";
  risk_score?: number;

  // Visualizations
  heatmap_base64?: string;

  // Quality
  quality?: QualityMetrics;

  // Recommendations
  recommendations: string[];

  // Error info (for failed requests)
  error?: {
    code: string;
    message: string;
    stage: string;
    user_message?: {
      title: string;
      explanation: string;
      action: string;
    };
    recoverable: boolean;
  };
}

// Condition information
export interface ConditionInfo {
  name: string;
  description: string;
  category: string;
  urgency: string;
  accuracy: number;
}

// Conditions list response
export interface ConditionsResponse {
  conditions: ConditionInfo[];
  total: number;
}

// Health check response
export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  module: string;
  version: string;
  model: string;
  torchxrayvision_available: boolean;
  gradcam_available: boolean;
  pathologies_count: number;
}

// Module info response
export interface ModuleInfoResponse {
  name: string;
  version: string;
  description: string;
  model: string;
  pathologies: number;
  datasets: string[];
  supported_formats: string[];
  max_file_size: string;
  recommended_resolution: string;
}

// Severity color mapping helper
export function getSeverityColor(severity: string): {
  text: string;
  bg: string;
  border: string;
  fill: string;
} {
  switch (severity) {
    case "normal":
      return {
        text: "text-green-600",
        bg: "bg-green-50",
        border: "border-green-200",
        fill: "bg-green-500",
      };
    case "minimal":
    case "low":
      return {
        text: "text-blue-600",
        bg: "bg-blue-50",
        border: "border-blue-200",
        fill: "bg-blue-500",
      };
    case "moderate":
    case "possible":
      return {
        text: "text-yellow-600",
        bg: "bg-yellow-50",
        border: "border-yellow-200",
        fill: "bg-yellow-500",
      };
    case "high":
    case "likely":
      return {
        text: "text-orange-600",
        bg: "bg-orange-50",
        border: "border-orange-200",
        fill: "bg-orange-500",
      };
    case "critical":
      return {
        text: "text-red-700",
        bg: "bg-red-100",
        border: "border-red-300",
        fill: "bg-red-600",
      };
    default:
      return {
        text: "text-zinc-600",
        bg: "bg-zinc-50",
        border: "border-zinc-200",
        fill: "bg-zinc-500",
      };
  }
}

// Risk level color mapping
export function getRiskColor(riskLevel: string): {
  text: string;
  bg: string;
  border: string;
  glow: string;
} {
  switch (riskLevel) {
    case "normal":
    case "low":
      return {
        text: "text-green-600",
        bg: "bg-green-500",
        border: "border-green-200",
        glow: "shadow-green-500/20",
      };
    case "moderate":
      return {
        text: "text-yellow-600",
        bg: "bg-yellow-500",
        border: "border-yellow-200",
        glow: "shadow-yellow-500/20",
      };
    case "high":
      return {
        text: "text-orange-600",
        bg: "bg-orange-500",
        border: "border-orange-200",
        glow: "shadow-orange-500/20",
      };
    case "critical":
      return {
        text: "text-red-600",
        bg: "bg-red-600",
        border: "border-red-200",
        glow: "shadow-red-500/30",
      };
    default:
      return {
        text: "text-zinc-600",
        bg: "bg-zinc-500",
        border: "border-zinc-200",
        glow: "shadow-zinc-500/10",
      };
  }
}
