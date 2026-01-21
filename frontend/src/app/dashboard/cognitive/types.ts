/**
 * Cognitive Pipeline Frontend Types - Production Grade
 * Aligned with backend schemas.py v2.0.0
 */

// =============================================================================
// ENUMS
// =============================================================================

export type PipelineStage =
  | "pending"
  | "validating"
  | "extracting"
  | "scoring"
  | "complete"
  | "failed";

export type RiskLevel = "low" | "moderate" | "high" | "critical";

export type TaskCompletionStatus =
  | "complete"
  | "incomplete"
  | "invalid"
  | "unknown";

// =============================================================================
// INPUT TYPES
// =============================================================================

export interface TaskEvent {
  timestamp: number;
  event_type: string;
  payload: Record<string, unknown>;
}

export interface TaskResult {
  task_id: string;
  start_time: string; // ISO datetime
  end_time: string;
  events: TaskEvent[];
  metadata: Record<string, unknown>;
}

export interface CognitiveSessionInput {
  session_id: string;
  patient_id?: string;
  tasks: TaskResult[];
  user_metadata: Record<string, unknown>;
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

export interface StageProgress {
  stage: PipelineStage;
  stage_index: number;
  total_stages: number;
  message: string;
  started_at?: string;
  completed_at?: string;
  duration_ms?: number;
  error?: string;
}

export interface DomainRiskDetail {
  score: number;
  risk_level: RiskLevel;
  percentile?: number;
  confidence: number;
  contributing_factors: string[];
}

export interface CognitiveRiskAssessment {
  overall_risk_score: number;
  risk_level: RiskLevel;
  confidence_score: number;
  confidence_interval: [number, number];
  domain_risks: Record<string, DomainRiskDetail>;
}

export interface TaskMetrics {
  task_id: string;
  completion_status: TaskCompletionStatus;
  performance_score: number;
  parameters: Record<string, number>;
  validity_flag: boolean;
  quality_warnings: string[];
}

export interface CognitiveFeatures {
  domain_scores: Record<string, number>;
  raw_metrics: TaskMetrics[];
  fatigue_index: number;
  consistency_score: number;
  valid_task_count: number;
  total_task_count: number;
}

export interface ClinicalRecommendation {
  category: "clinical" | "lifestyle" | "routine" | "specific";
  description: string;
  priority: "low" | "medium" | "high" | "critical";
  action_url?: string;
}

export interface ExplainabilityArtifact {
  summary: string;
  key_factors: string[];
  domain_contributions: Record<string, number>;
  methodology_note: string;
}

export interface CognitiveResponse {
  session_id: string;
  pipeline_version: string;
  timestamp: string;
  processing_time_ms: number;
  status: "success" | "partial" | "failed";
  stages: StageProgress[];
  risk_assessment: CognitiveRiskAssessment | null;
  features: CognitiveFeatures | null;
  recommendations: ClinicalRecommendation[];
  explainability: ExplainabilityArtifact | null;
  error_code: string | null;
  error_message: string | null;
  recoverable: boolean;
  retry_after_ms?: number;
}

// =============================================================================
// UI STATE TYPES
// =============================================================================

export type SessionState =
  | "idle"
  | "testing"
  | "submitting"
  | "polling"
  | "success"
  | "partial"
  | "error";

export interface CognitiveSessionState {
  state: SessionState;
  sessionId: string | null;
  tasks: TaskResult[];
  response: CognitiveResponse | null;
  error: string | null;
  retryCount: number;
}

// =============================================================================
// API HELPERS
// =============================================================================

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export async function submitCognitiveSession(
  input: CognitiveSessionInput,
): Promise<CognitiveResponse> {
  const res = await fetch(`${API_BASE}/api/cognitive/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error_message || `Request failed: ${res.status}`);
  }

  return res.json();
}

export async function validateCognitiveSession(
  input: CognitiveSessionInput,
): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
  const res = await fetch(`${API_BASE}/api/cognitive/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!res.ok) {
    throw new Error("Validation request failed");
  }

  return res.json();
}

export async function checkCognitiveHealth(): Promise<{
  status: string;
  version: string;
}> {
  const res = await fetch(`${API_BASE}/api/cognitive/health`);
  return res.json();
}
