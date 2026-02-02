/**
 * Enhanced Speech Analysis Types v4.0
 * Matches backend EnhancedSpeechAnalysisResponse schema
 * Includes research-grade biomarkers and condition risks
 *
 * v4.0 Features:
 * - Uncertainty quantification (95% CI)
 * - SHAP-style risk explanations
 * - Age/sex-adjusted normative comparisons
 * - Real-time streaming support
 */

// Individual biomarker with clinical metadata
export interface BiomarkerResult {
  value: number;
  unit: string;
  normal_range: [number, number];
  is_estimated: boolean;
  confidence: number | null;
  status?: "normal" | "borderline" | "abnormal";
  z_score?: number;
  percentile?: number;
}

// Primary 10 biomarkers from backend
export interface EnhancedBiomarkers {
  jitter: BiomarkerResult;
  shimmer: BiomarkerResult;
  hnr: BiomarkerResult;
  speech_rate: BiomarkerResult;
  pause_ratio: BiomarkerResult;
  fluency_score: BiomarkerResult;
  voice_tremor: BiomarkerResult;
  articulation_clarity: BiomarkerResult;
  prosody_variation: BiomarkerResult;
  cpps?: BiomarkerResult; // Optional CPPS
}

// Research-grade extended biomarkers
export interface ExtendedBiomarkers {
  mean_f0?: BiomarkerResult; // Mean fundamental frequency
  f0_range?: BiomarkerResult; // Pitch range
  nii?: BiomarkerResult; // Neuromotor Instability Index
  vfmt?: BiomarkerResult; // Vocal Fold Micro-Tremor
  ace?: BiomarkerResult; // Articulatory Coordination Entropy
  rpcs?: BiomarkerResult; // Respiratory-Phonatory Coupling Score
}

// Condition-specific risk assessment
export interface ConditionRisk {
  condition: string;
  probability: number; // 0-1
  confidence: number;
  confidence_interval: [number, number];
  risk_level: "low" | "moderate" | "high" | "critical";
  contributing_factors: string[];
}

// Baseline comparison for tracking changes
export interface BaselineComparison {
  biomarker_name: string;
  current_value: number;
  baseline_value: number;
  delta: number;
  delta_percent: number;
  direction: "improved" | "worsened" | "stable";
}

// File information
export interface FileInfo {
  filename?: string;
  size?: number;
  content_type?: string;
  duration?: number;
  sample_rate?: number;
  resampled?: boolean;
}

// Enhanced speech analysis response from backend
export interface EnhancedSpeechAnalysisResponse {
  session_id: string;
  processing_time: number;
  timestamp: string;
  confidence: number;
  risk_score: number; // 0-1
  quality_score: number;
  biomarkers: EnhancedBiomarkers;
  file_info?: FileInfo;
  recommendations: string[];
  baseline_comparisons?: BaselineComparison[];
  status: "completed" | "partial" | "error";
  error_message?: string;

  // Research-grade extensions
  extended_biomarkers?: ExtendedBiomarkers;
  condition_risks?: ConditionRisk[];
  confidence_interval?: [number, number];
  clinical_notes?: string;
  requires_review?: boolean;
  review_reason?: string;
}

// Biomarker display configuration
export interface BiomarkerDisplayConfig {
  key: keyof EnhancedBiomarkers | keyof ExtendedBiomarkers;
  label: string;
  description: string;
  icon: string;
  higherIsBetter: boolean;
  formatValue: (value: number, unit: string) => string;
  isResearch?: boolean;
  clinicalMeaning?: string;
}

// Primary biomarker display configurations (always shown)
export const PRIMARY_BIOMARKER_CONFIGS: BiomarkerDisplayConfig[] = [
  {
    key: "jitter",
    label: "Jitter",
    description: "Voice pitch stability",
    clinicalMeaning: "Elevated in Parkinson's, laryngeal pathology",
    icon: "zap",
    higherIsBetter: false,
    formatValue: (v) => `${v.toFixed(2)}%`,
  },
  {
    key: "shimmer",
    label: "Shimmer",
    description: "Voice amplitude stability",
    clinicalMeaning: "Elevated in vocal fold disorders",
    icon: "bar-chart-2",
    higherIsBetter: false,
    formatValue: (v) => `${v.toFixed(2)}%`,
  },
  {
    key: "hnr",
    label: "HNR",
    description: "Harmonics-to-Noise Ratio",
    clinicalMeaning: "Low = breathy, strained voice",
    icon: "volume-2",
    higherIsBetter: true,
    formatValue: (v, u) => `${v.toFixed(1)} ${u}`,
  },
  {
    key: "cpps",
    label: "CPPS",
    description: "Cepstral Peak Prominence (gold standard)",
    clinicalMeaning: "Low = dysphonia, voice disorder",
    icon: "audio-waveform",
    higherIsBetter: true,
    formatValue: (v, u) => `${v.toFixed(1)} ${u}`,
  },
  {
    key: "speech_rate",
    label: "Speech Rate",
    description: "Speaking speed in syllables/second",
    clinicalMeaning: "Low = cognitive/motor impairment",
    icon: "gauge",
    higherIsBetter: false,
    formatValue: (v, u) => `${v.toFixed(1)} ${u}`,
  },
  {
    key: "voice_tremor",
    label: "Voice Tremor",
    description: "Tremor intensity in voice",
    clinicalMeaning: "Elevated in PD, essential tremor",
    icon: "activity",
    higherIsBetter: false,
    formatValue: (v) => `${(v * 100).toFixed(1)}%`,
  },
];

// Secondary biomarker configs (shown on expand)
export const SECONDARY_BIOMARKER_CONFIGS: BiomarkerDisplayConfig[] = [
  {
    key: "pause_ratio",
    label: "Pause Ratio",
    description: "Proportion of silence in speech",
    clinicalMeaning: "High = cognitive decline, word-finding difficulty",
    icon: "pause",
    higherIsBetter: false,
    formatValue: (v) => `${(v * 100).toFixed(0)}%`,
  },
  {
    key: "articulation_clarity",
    label: "Articulation",
    description: "Clarity of speech articulation (FCR)",
    clinicalMeaning: "High deviation = dysarthria",
    icon: "message-circle",
    higherIsBetter: false,
    formatValue: (v) => v.toFixed(2),
  },
  {
    key: "prosody_variation",
    label: "Prosody",
    description: "Prosodic richness (F0 variation)",
    clinicalMeaning: "Low = monotone (depression, PD)",
    icon: "music",
    higherIsBetter: true,
    formatValue: (v, u) => `${v.toFixed(1)} ${u}`,
  },
  {
    key: "fluency_score",
    label: "Fluency",
    description: "Overall speech fluency",
    icon: "waves",
    higherIsBetter: true,
    formatValue: (v) => `${(v * 100).toFixed(0)}%`,
  },
];

// Research-grade biomarker configs
export const RESEARCH_BIOMARKER_CONFIGS: BiomarkerDisplayConfig[] = [
  {
    key: "nii",
    label: "Neuromotor Instability",
    description: "Composite index (tremor + jitter + shimmer)",
    clinicalMeaning: "Novel research metric for motor disorders",
    icon: "brain",
    higherIsBetter: false,
    formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    isResearch: true,
  },
  {
    key: "vfmt",
    label: "Vocal Fold Micro-Tremor",
    description: "Subclinical tremor detection",
    clinicalMeaning: "Detects tremor before visible symptoms",
    icon: "activity",
    higherIsBetter: false,
    formatValue: (v) => v.toFixed(3),
    isResearch: true,
  },
  {
    key: "ace",
    label: "Articulatory Entropy",
    description: "Articulatory coordination measure",
    clinicalMeaning: "High = uncoordinated articulation",
    icon: "git-branch",
    higherIsBetter: false,
    formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    isResearch: true,
  },
  {
    key: "rpcs",
    label: "Respiratory-Phonatory Coupling",
    description: "Breathing-voice synchronization",
    clinicalMeaning: "Low = motor disorder indicator",
    icon: "wind",
    higherIsBetter: true,
    formatValue: (v) => `${(v * 100).toFixed(0)}%`,
    isResearch: true,
  },
  {
    key: "mean_f0",
    label: "Fundamental Frequency",
    description: "Average voice pitch",
    icon: "music-2",
    higherIsBetter: false,
    formatValue: (v, u) => `${v.toFixed(0)} ${u}`,
    isResearch: true,
  },
  {
    key: "f0_range",
    label: "Pitch Range",
    description: "Vocal pitch range",
    icon: "maximize-2",
    higherIsBetter: false,
    formatValue: (v, u) => `${v.toFixed(0)} ${u}`,
    isResearch: true,
  },
];

// All biomarker configs combined
export const BIOMARKER_CONFIGS: BiomarkerDisplayConfig[] = [
  ...PRIMARY_BIOMARKER_CONFIGS,
  ...SECONDARY_BIOMARKER_CONFIGS,
];

// Helper to check if value is within normal range
export function isWithinNormalRange(biomarker: BiomarkerResult): boolean {
  const [min, max] = biomarker.normal_range;
  return biomarker.value >= min && biomarker.value <= max;
}

// Helper to get status based on value and normal range
export function getBiomarkerStatus(
  biomarker: BiomarkerResult,
  higherIsBetter: boolean,
): "normal" | "warning" | "abnormal" {
  const [min, max] = biomarker.normal_range;
  const { value } = biomarker;

  if (value >= min && value <= max) return "normal";

  // Check how far outside the range
  const deviation = value < min ? (min - value) / min : (value - max) / max;

  if (deviation > 0.5) return "abnormal";
  return "warning";
}

// Helper to get risk level color
export function getRiskLevelColor(level: string): string {
  switch (level) {
    case "low":
      return "#22c55e";
    case "moderate":
      return "#f59e0b";
    case "high":
      return "#ef4444";
    case "critical":
      return "#7f1d1d";
    default:
      return "#64748b";
  }
}

// Helper to format condition name
export function formatConditionName(condition: string): string {
  const names: Record<string, string> = {
    parkinsons: "Parkinson's Disease",
    cognitive_decline: "Cognitive Decline",
    depression: "Depression",
    dysarthria: "Dysarthria",
  };
  return (
    names[condition] ||
    condition.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  );
}

// Condition explanations for AI
export const CONDITION_EXPLANATIONS: Record<
  string,
  {
    description: string;
    indicators: string[];
    action: string;
  }
> = {
  parkinsons: {
    description: "Voice patterns sometimes associated with Parkinson's disease",
    indicators: [
      "Voice tremor",
      "Reduced pitch variation",
      "Slower speech rate",
      "Elevated jitter",
    ],
    action: "Consider neurological evaluation if other symptoms present",
  },
  cognitive_decline: {
    description: "Patterns that may indicate cognitive changes",
    indicators: [
      "Increased pauses",
      "Slower speech rate",
      "Word-finding hesitations",
    ],
    action:
      "Discuss with healthcare provider, especially if concerned about memory",
  },
  depression: {
    description: "Voice characteristics associated with mood changes",
    indicators: [
      "Monotone speech",
      "Reduced prosodic variation",
      "Slower rate",
    ],
    action: "Consider mental health evaluation if experiencing mood changes",
  },
  dysarthria: {
    description: "Motor speech disorder affecting articulation",
    indicators: [
      "Reduced articulation clarity",
      "Voice quality changes",
      "Irregular speech",
    ],
    action: "Speech-language pathology evaluation recommended",
  },
};
