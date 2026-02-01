"use client";

import React, { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Eye,
  Target,
  AlertCircle,
  Loader2,
  Upload,
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  RefreshCw,
  Layers,
  Droplet,
  FileText,
  HeartPulse,
  Brain,
  Gauge,
  Clock,
  ImageIcon,
  Stethoscope,
  TrendingUp,
  Info,
  Shield,
  Zap,
} from "lucide-react";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";
import { usePipelineStatus } from "@/components/pipeline";
import { usePatient } from "@/context/PatientContext";

// ============================================================================
// Types matching backend v4.0 structure
// ============================================================================

export interface PipelineError {
  stage: string;
  error_type: string;
  message: string;
  timestamp: string;
}

export interface PipelineState {
  session_id: string;
  current_stage: string;
  stages_completed: string[];
  stages_timing_ms: Record<string, number>;
  errors: PipelineError[];
  warnings: string[];
  started_at: string;
  completed_at?: string;
}

export interface BiomarkerValue {
  value: number;
  normal_range?: number[];
  threshold?: number;
  status: "normal" | "abnormal" | "borderline";
  measurement_confidence: number;
  clinical_significance?: string;
  source?: string;
}

export interface VesselBiomarkers {
  tortuosity_index: BiomarkerValue;
  av_ratio: BiomarkerValue;
  vessel_density: BiomarkerValue;
  fractal_dimension: BiomarkerValue;
  branching_coefficient: BiomarkerValue;
}

export interface OpticDiscBiomarkers {
  cup_disc_ratio: BiomarkerValue;
  disc_area_mm2: BiomarkerValue;
  rim_area_mm2: BiomarkerValue;
  rnfl_thickness: BiomarkerValue;
  notching_detected: boolean;
}

export interface MacularBiomarkers {
  thickness: BiomarkerValue;
  volume: BiomarkerValue;
}

export interface LesionBiomarkers {
  hemorrhage_count: BiomarkerValue;
  microaneurysm_count: BiomarkerValue;
  exudate_area_percent: BiomarkerValue;
  cotton_wool_spots: number;
  neovascularization_detected: boolean;
  venous_beading_detected: boolean;
  irma_detected: boolean;
}

export interface CompleteBiomarkers {
  vessels: VesselBiomarkers;
  optic_disc: OpticDiscBiomarkers;
  macula: MacularBiomarkers;
  lesions: LesionBiomarkers;
}

export interface DiabeticRetinopathyResult {
  grade: number;
  grade_name: string;
  probability: number;
  probabilities_all_grades: Record<string, number>;
  referral_urgency: string;
  clinical_action: string;
  macular_edema_present: boolean;
  clinically_significant_macular_edema: boolean;
}

export interface RiskAssessment {
  overall_score: number;
  category: string;
  confidence: number;
  confidence_interval_95: [number, number];
  primary_finding: string;
  contributing_factors: Record<string, number>;
  systemic_risk_indicators: Record<string, string>;
}

export interface ClinicalFinding {
  finding_type: string;
  anatomical_location: string;
  severity: string;
  description: string;
  clinical_relevance: string;
  icd10_code?: string;
  requires_referral: boolean;
  confidence: number;
}

export interface DifferentialDiagnosis {
  diagnosis: string;
  probability: number;
  supporting_evidence: string[];
  icd10_code: string;
}

export interface ImageQuality {
  overall_score: number;
  gradability: string;
  is_gradable: boolean;
  issues: string[];
  snr_db: number;
  focus_score: number;
  illumination_score: number;
  contrast_score: number;
  optic_disc_visible: boolean;
  macula_visible: boolean;
  vessel_arcades_visible: boolean;
  resolution: [number, number];
  file_size_mb: number;
  field_of_view: string;
}

export interface FourTwoOneRule {
  hemorrhages_4_quadrants: boolean;
  venous_beading_2_quadrants: boolean;
  irma_1_quadrant: boolean;
}

export interface DiabeticMacularEdema {
  present: boolean;
  csme: boolean;
  central_involvement: boolean;
  severity: "none" | "mild" | "moderate" | "severe";
}

export interface RetinalAnalysisResponse {
  success: boolean;
  session_id: string;
  patient_id: string;
  pipeline_state: PipelineState;
  timestamp: string;
  total_processing_time_ms: number;
  model_version: string;
  image_quality: ImageQuality;
  biomarkers: CompleteBiomarkers;
  diabetic_retinopathy: DiabeticRetinopathyResult & {
    four_two_one_rule?: FourTwoOneRule;
  };
  diabetic_macular_edema?: DiabeticMacularEdema;
  risk_assessment: RiskAssessment;
  findings: ClinicalFinding[];
  differential_diagnoses: DifferentialDiagnosis[];
  recommendations: string[];
  clinical_summary: string;
  heatmap_base64?: string;
  segmentation_base64?: string;
  eye: string;
  analysis_type: string;
}

type AnalysisState = "idle" | "uploading" | "processing" | "complete" | "error";

// ============================================================================
// Biomarker Display Card - Dark Theme
// ============================================================================

function BiomarkerCard({
  label,
  biomarker,
  icon: Icon,
}: {
  label: string;
  biomarker: BiomarkerValue;
  icon: React.ElementType;
}) {
  const statusConfig = {
    normal: {
      bg: "bg-emerald-500/15",
      text: "text-emerald-400",
      dot: "bg-emerald-500",
    },
    borderline: {
      bg: "bg-amber-500/15",
      text: "text-amber-400",
      dot: "bg-amber-500",
    },
    abnormal: {
      bg: "bg-red-500/15",
      text: "text-red-400",
      dot: "bg-red-500",
    },
  }[biomarker.status] || {
    bg: "bg-zinc-500/15",
    text: "text-zinc-400",
    dot: "bg-zinc-500",
  };

  const formatValue = (v: number) => {
    if (v === Math.floor(v)) return v.toString();
    return v.toFixed(3);
  };

  return (
    <div className="bg-zinc-800/50 rounded-lg border border-zinc-700/50 p-3 hover:border-zinc-600 transition-colors">
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-1.5 rounded-md ${statusConfig.bg}`}>
          <Icon className={`h-3 w-3 ${statusConfig.text}`} />
        </div>
        <span className="text-[11px] font-medium text-zinc-400 truncate flex-1">
          {label}
        </span>
        <span className={`w-2 h-2 rounded-full ${statusConfig.dot}`} />
      </div>
      <div className="text-[18px] font-bold text-zinc-100 leading-tight">
        {formatValue(biomarker.value)}
      </div>
      <div className="flex items-center justify-between mt-1.5 pt-1.5 border-t border-zinc-700/50">
        <span
          className={`text-[10px] font-semibold ${statusConfig.text} capitalize`}
        >
          {biomarker.status}
        </span>
        <span className="text-[9px] text-zinc-500 font-medium">
          {(biomarker.measurement_confidence * 100).toFixed(0)}% conf
        </span>
      </div>
      {biomarker.clinical_significance && (
        <div className="text-[9px] text-zinc-500 mt-1.5 leading-snug line-clamp-2">
          {biomarker.clinical_significance}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Image Quality Panel - Dark Theme
// ============================================================================

function ImageQualityPanel({ quality }: { quality: ImageQuality }) {
  const gradabilityConfig: Record<string, { color: string; bg: string }> = {
    excellent: { color: "text-emerald-400", bg: "bg-emerald-500" },
    good: { color: "text-green-400", bg: "bg-green-500" },
    fair: { color: "text-amber-400", bg: "bg-amber-500" },
    poor: { color: "text-orange-400", bg: "bg-orange-500" },
    ungradable: { color: "text-red-400", bg: "bg-red-500" },
  };
  const config = gradabilityConfig[quality.gradability] || {
    color: "text-zinc-400",
    bg: "bg-zinc-500",
  };

  const metrics = [
    { label: "Focus", value: quality.focus_score, icon: Target },
    { label: "Illumination", value: quality.illumination_score, icon: Zap },
    { label: "Contrast", value: quality.contrast_score, icon: Layers },
    {
      label: "SNR",
      value: quality.snr_db,
      icon: Activity,
      unit: "dB",
      isRaw: true,
    },
  ];

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/15">
            <ImageIcon className="h-4 w-4 text-cyan-400" />
          </div>
          <span className="text-[13px] font-semibold text-zinc-100">
            Image Quality
          </span>
        </div>
        <span
          className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-full text-white ${config.bg}`}
        >
          {quality.gradability}
        </span>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Overall Score Bar */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex-1">
            <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${config.bg}`}
                style={{ width: `${quality.overall_score * 100}%` }}
              />
            </div>
          </div>
          <span className="text-[14px] font-bold text-zinc-100">
            {(quality.overall_score * 100).toFixed(0)}%
          </span>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-2">
          {metrics.map(({ label, value, icon: Icon, unit, isRaw }) => (
            <div
              key={label}
              className="flex items-center gap-2 p-2 bg-zinc-800/50 rounded-lg border border-zinc-700/50"
            >
              <div className="p-1 bg-zinc-700/50 rounded">
                <Icon className="h-3 w-3 text-zinc-400" />
              </div>
              <div className="flex-1">
                <div className="text-[9px] text-zinc-500 uppercase tracking-wide">
                  {label}
                </div>
                <div className="text-[12px] font-semibold text-zinc-200">
                  {isRaw ? value.toFixed(1) : `${(value * 100).toFixed(0)}%`}
                  {unit && (
                    <span className="text-[9px] text-zinc-500 ml-0.5">
                      {unit}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Visibility Indicators */}
        <div className="flex items-center gap-4 mt-3 pt-3 border-t border-zinc-700/50">
          <div className="flex items-center gap-1.5">
            {quality.optic_disc_visible ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            ) : (
              <XCircle className="h-3.5 w-3.5 text-red-400" />
            )}
            <span className="text-[10px] text-zinc-400">Disc</span>
          </div>
          <div className="flex items-center gap-1.5">
            {quality.macula_visible ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            ) : (
              <XCircle className="h-3.5 w-3.5 text-red-400" />
            )}
            <span className="text-[10px] text-zinc-400">Macula</span>
          </div>
          <div className="flex items-center gap-1.5">
            {quality.vessel_arcades_visible ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            ) : (
              <XCircle className="h-3.5 w-3.5 text-red-400" />
            )}
            <span className="text-[10px] text-zinc-400">Vessels</span>
          </div>
        </div>

        {/* Issues */}
        {quality.issues.length > 0 && (
          <div className="mt-3 pt-3 border-t border-zinc-700/50 space-y-1">
            {quality.issues.map((issue, i) => (
              <div
                key={i}
                className="flex items-center gap-2 text-[10px] text-amber-400 bg-amber-500/10 px-2 py-1 rounded border border-amber-500/20"
              >
                <AlertTriangle className="h-3 w-3" />
                {issue}
              </div>
            ))}
          </div>
        )}

        {/* Meta Info */}
        <div className="flex items-center justify-between mt-3 pt-2 border-t border-zinc-700/50 text-[9px] text-zinc-500">
          <span>
            {quality.resolution[0]} x {quality.resolution[1]}
          </span>
          <span>{quality.file_size_mb.toFixed(2)} MB</span>
          <span>{quality.field_of_view || "standard"}</span>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// DME Card - Dark Theme
// ============================================================================

function DMECard({ dme }: { dme: DiabeticMacularEdema }) {
  const severityConfig = {
    none: { bg: "bg-emerald-500/15", text: "text-emerald-400" },
    mild: { bg: "bg-amber-500/15", text: "text-amber-400" },
    moderate: { bg: "bg-orange-500/15", text: "text-orange-400" },
    severe: { bg: "bg-red-500/15", text: "text-red-400" },
  };
  const config = severityConfig[dme.severity] || {
    bg: "bg-zinc-500/15",
    text: "text-zinc-400",
  };

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-violet-500/15">
            <Eye className="h-4 w-4 text-violet-400" />
          </div>
          <span className="text-[13px] font-semibold text-zinc-100">
            Diabetic Macular Edema
          </span>
        </div>
        <span
          className={`text-[9px] font-bold uppercase px-2 py-0.5 rounded-full ${config.bg} ${config.text}`}
        >
          {dme.severity}
        </span>
      </div>

      {/* Content */}
      <div className="p-4">
        <div className="grid grid-cols-3 gap-2">
          <div className="text-center p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <div className="text-[9px] text-zinc-500 uppercase tracking-wide mb-1">
              Present
            </div>
            {dme.present ? (
              <AlertCircle className="h-5 w-5 text-amber-400 mx-auto" />
            ) : (
              <CheckCircle2 className="h-5 w-5 text-emerald-400 mx-auto" />
            )}
          </div>
          <div className="text-center p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <div className="text-[9px] text-zinc-500 uppercase tracking-wide mb-1">
              CSME
            </div>
            {dme.csme ? (
              <AlertCircle className="h-5 w-5 text-red-400 mx-auto" />
            ) : (
              <CheckCircle2 className="h-5 w-5 text-emerald-400 mx-auto" />
            )}
          </div>
          <div className="text-center p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
            <div className="text-[9px] text-zinc-500 uppercase tracking-wide mb-1">
              Central
            </div>
            {dme.central_involvement ? (
              <AlertCircle className="h-5 w-5 text-red-400 mx-auto" />
            ) : (
              <CheckCircle2 className="h-5 w-5 text-emerald-400 mx-auto" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// 4-2-1 Rule Display - Dark Theme
// ============================================================================

function FourTwoOneRuleDisplay({ rule }: { rule: FourTwoOneRule }) {
  const criteria = [
    { label: "Hemorrhages in 4 quadrants", met: rule.hemorrhages_4_quadrants },
    {
      label: "Venous beading in 2+ quadrants",
      met: rule.venous_beading_2_quadrants,
    },
    { label: "IRMA in 1+ quadrant", met: rule.irma_1_quadrant },
  ];
  const anyMet =
    rule.hemorrhages_4_quadrants ||
    rule.venous_beading_2_quadrants ||
    rule.irma_1_quadrant;

  return (
    <div
      className={`rounded-lg p-3 border ${anyMet ? "bg-red-500/10 border-red-500/30" : "bg-zinc-800/50 border-zinc-700/50"}`}
    >
      <div className="text-[9px] font-semibold text-zinc-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
        <Shield className="h-3 w-3" />
        4-2-1 Rule (Severe NPDR)
      </div>
      <div className="space-y-1.5">
        {criteria.map(({ label, met }) => (
          <div key={label} className="flex items-center gap-2">
            {met ? (
              <div className="w-4 h-4 rounded-full bg-red-500/20 flex items-center justify-center">
                <AlertCircle className="h-2.5 w-2.5 text-red-400" />
              </div>
            ) : (
              <div className="w-4 h-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <CheckCircle2 className="h-2.5 w-2.5 text-emerald-400" />
              </div>
            )}
            <span
              className={`text-[10px] ${met ? "text-red-400 font-medium" : "text-zinc-500"}`}
            >
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Systemic Risk Indicators Panel - Dark Theme
// ============================================================================

function SystemicRiskPanel({
  indicators,
}: {
  indicators: Record<string, string>;
}) {
  if (!indicators || Object.keys(indicators).length === 0) return null;

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800">
      {/* Header */}
      <div className="flex items-center gap-2 p-4 border-b border-zinc-800">
        <div className="p-1.5 rounded-lg bg-amber-500/15">
          <HeartPulse className="h-4 w-4 text-amber-400" />
        </div>
        <span className="text-[13px] font-semibold text-zinc-100">
          Systemic Risk Indicators
        </span>
      </div>

      {/* Content */}
      <div className="p-4 space-y-2">
        {Object.entries(indicators).map(([key, value]) => (
          <div
            key={key}
            className="flex items-start gap-2 text-[11px] p-2 bg-amber-500/10 rounded-lg border border-amber-500/20"
          >
            <TrendingUp className="h-3.5 w-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
            <div>
              <span className="font-semibold text-amber-300">{key}:</span>
              <span className="text-amber-200/80 ml-1">{value}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Pipeline Progress Display - Dark Theme
// ============================================================================

function PipelineProgress({ pipelineState }: { pipelineState: PipelineState }) {
  const stages = [
    "input_validation",
    "quality_assessment",
    "vessel_analysis",
    "optic_disc_analysis",
    "macular_analysis",
    "lesion_detection",
    "dr_grading",
    "risk_calculation",
    "clinical_assessment",
    "output_formatting",
  ];
  const completedSet = new Set(pipelineState.stages_completed);
  const totalTime = Object.values(pipelineState.stages_timing_ms).reduce(
    (a, b) => a + b,
    0,
  );

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/15">
            <Activity className="h-4 w-4 text-emerald-400" />
          </div>
          <span className="text-[13px] font-semibold text-zinc-100">
            Pipeline Execution
          </span>
        </div>
        <span className="text-[10px] font-medium text-zinc-400 bg-zinc-800 px-2 py-0.5 rounded">
          {totalTime < 1000
            ? `${totalTime.toFixed(0)}ms`
            : `${(totalTime / 1000).toFixed(1)}s`}
        </span>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Progress Bar */}
        <div className="flex gap-1 mb-3">
          {stages.map((stage) => {
            const isComplete = completedSet.has(stage);
            const isCurrent = pipelineState.current_stage === stage;
            return (
              <div
                key={stage}
                className={`flex-1 h-1.5 rounded-full transition-all ${
                  isComplete
                    ? "bg-emerald-500"
                    : isCurrent
                      ? "bg-cyan-400 animate-pulse"
                      : "bg-zinc-700"
                }`}
                title={stage.replace(/_/g, " ")}
              />
            );
          })}
        </div>

        {/* Completed Count */}
        <div className="text-[10px] text-zinc-500 text-center">
          {pipelineState.stages_completed.length} / {stages.length} stages
          complete
        </div>

        {/* Warnings */}
        {pipelineState.warnings.length > 0 && (
          <div className="mt-3 pt-3 border-t border-zinc-700/50 space-y-1">
            {pipelineState.warnings.map((w, i) => (
              <div
                key={i}
                className="flex items-center gap-1.5 text-[10px] text-amber-400 bg-amber-500/10 px-2 py-1 rounded border border-amber-500/20"
              >
                <AlertTriangle className="h-3 w-3" />
                {w}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Contributing Factors Panel - Dark Theme
// ============================================================================

function ContributingFactorsPanel({
  factors,
}: {
  factors: Record<string, number>;
}) {
  if (!factors || Object.keys(factors).length === 0) return null;

  const sortedFactors = Object.entries(factors)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="mt-3 pt-3 border-t border-zinc-700/50">
      <div className="text-[9px] font-semibold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1">
        <Gauge className="h-3 w-3" />
        Contributing Factors
      </div>
      <div className="space-y-1.5">
        {sortedFactors.map(([factor, weight]) => (
          <div key={factor} className="flex items-center gap-2">
            <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                style={{ width: `${weight}%` }}
              />
            </div>
            <span className="text-[9px] text-zinc-500 w-20 text-right truncate">
              {factor}
            </span>
            <span className="text-[9px] font-semibold text-zinc-300 w-8">
              {weight.toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface RetinalAssessmentProps {
  onProcessingChange?: (isProcessing: boolean) => void;
}

export default function RetinalAssessment({
  onProcessingChange,
}: RetinalAssessmentProps) {
  const [state, setState] = useState<AnalysisState>("idle");
  const [results, setResults] = useState<RetinalAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showSegmentation, setShowSegmentation] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { activePatient } = usePatient();

  // Pipeline status bar integration
  const { startPipeline, updatePipeline, completePipeline } =
    usePipelineStatus();

  const addLog = useCallback((message: string) => {
    const timestamp =
      new Date().toISOString().split("T")[1]?.split(".")[0] ?? "";
    const logEntry = `[${timestamp}] ${message}`;
    console.log(`[RETINAL] ${message}`);
    setLogs((prev) => [...prev.slice(-20), logEntry]);
  }, []);

  const analyzeImage = useCallback(
    async (imageFile: File) => {
      setState("processing");
      setError(null);
      onProcessingChange?.(true);
      addLog(
        `Starting analysis for: ${imageFile.name} (${(imageFile.size / 1024 / 1024).toFixed(2)}MB)`,
      );

      // Start pipeline in status bar
      startPipeline("retinal", [
        "input",
        "quality",
        "vessels",
        "disc",
        "lesions",
        "dr_grade",
        "risk",
        "output",
      ]);
      updatePipeline("retinal", { currentStage: "Uploading..." });

      try {
        const formData = new FormData();
        formData.append("image", imageFile);
        formData.append("patient_id", activePatient?.id || "ANONYMOUS");

        addLog("Sending to backend API...");
        updatePipeline("retinal", { currentStage: "Analyzing..." });

        const response = await fetch("/api/retinal/analyze", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(`API error ${response.status}: ${errText}`);
        }

        const data: RetinalAnalysisResponse = await response.json();
        addLog(`Response received: success=${data.success}`);

        if (data.pipeline_state) {
          addLog(
            `Stages completed: ${data.pipeline_state.stages_completed.length}`,
          );
          for (const stage of data.pipeline_state.stages_completed) {
            const timing = data.pipeline_state.stages_timing_ms[stage];
            addLog(
              `  [OK] ${stage}${timing ? ` (${timing.toFixed(1)}ms)` : ""}`,
            );
          }
        }

        if (data.diabetic_retinopathy) {
          addLog(
            `DR Grade: ${data.diabetic_retinopathy.grade} - ${data.diabetic_retinopathy.grade_name}`,
          );
        }

        if (data.risk_assessment) {
          addLog(
            `Risk Score: ${data.risk_assessment.overall_score.toFixed(1)} (${data.risk_assessment.category})`,
          );
        }

        setResults(data);
        setState(data.success ? "complete" : "error");

        // Update status bar with completion
        if (data.success) {
          completePipeline(
            "retinal",
            true,
            data.diabetic_retinopathy?.grade_name || "Complete",
          );
        } else {
          completePipeline("retinal", false, "Analysis failed");
        }

        if (!data.success) {
          setError("Analysis failed - check pipeline errors");
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        addLog(`ERROR: ${msg}`);
        setError(msg);
        setState("error");
        completePipeline("retinal", false, "Error");
      } finally {
        onProcessingChange?.(false);
      }
    },
    [
      addLog,
      startPipeline,
      updatePipeline,
      completePipeline,
      onProcessingChange,
    ],
  );

  const handleFileSelect = useCallback(
    (file: File) => {
      addLog(`File selected: ${file.name}`);

      if (!file.type.startsWith("image/")) {
        addLog("ERROR: Invalid file type");
        setError("Please select an image file");
        return;
      }
      if (file.size > 15 * 1024 * 1024) {
        addLog("ERROR: File too large");
        setError("File must be less than 15MB");
        return;
      }
      setPreviewUrl(URL.createObjectURL(file));
      analyzeImage(file);
    },
    [analyzeImage, addLog],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect],
  );

  const handleReset = useCallback(() => {
    addLog("Resetting...");
    setState("idle");
    setResults(null);
    setError(null);
    setShowHeatmap(false);
    setShowSegmentation(false);
    setLogs([]);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  }, [previewUrl, addLog]);

  const isProcessing = state === "processing" || state === "uploading";

  return (
    <div className="space-y-4">
      {/* Main Content */}
      <AnimatePresence mode="wait">
        {state === "complete" && results?.success ? (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          >
            {/* Summary */}
            {results.clinical_summary && (
              <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
                <div className="flex items-start gap-4">
                  <div className="p-2 bg-cyan-500/15 rounded-lg">
                    <FileText className="h-5 w-5 text-cyan-400" />
                  </div>
                  <div>
                    <div className="text-[14px] font-bold text-zinc-100 mb-1">
                      Clinical Assessment Summary
                    </div>
                    <div className="text-[13px] leading-relaxed text-zinc-400">
                      {results.clinical_summary}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Results Grid - Speech Analysis Style Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Results Panel (Left 2/3) */}
              <div className="lg:col-span-2 space-y-6">
                {/* Top Section: Image & Key Diagnosis */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Fundus Image */}
                  {previewUrl && (
                    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4 h-full flex flex-col">
                      <div className="flex justify-between items-center mb-3">
                        <div className="flex items-center gap-2">
                          <span className="text-[13px] font-semibold text-zinc-100">
                            Fundus Examination
                          </span>
                          {results.eye && results.eye !== "unknown" && (
                            <span className="px-1.5 py-0.5 bg-cyan-500/15 text-cyan-400 text-[10px] font-bold rounded">
                              {results.eye}
                            </span>
                          )}
                        </div>
                        <div className="flex gap-1.5">
                          <button
                            onClick={() => {
                              setShowHeatmap(!showHeatmap);
                              setShowSegmentation(false);
                            }}
                            className={`px-2 py-0.5 text-[10px] font-medium rounded transition-colors ${
                              showHeatmap
                                ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                            }`}
                          >
                            Heatmap
                          </button>
                          {results.segmentation_base64 && (
                            <button
                              onClick={() => {
                                setShowSegmentation(!showSegmentation);
                                setShowHeatmap(false);
                              }}
                              className={`px-2 py-0.5 text-[10px] font-medium rounded transition-colors ${
                                showSegmentation
                                  ? "bg-violet-500/20 text-violet-400 border border-violet-500/30"
                                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                              }`}
                            >
                              Vessels
                            </button>
                          )}
                        </div>
                      </div>
                      <div className="relative flex-1 bg-zinc-950 rounded-lg overflow-hidden min-h-[240px] group">
                        <img
                          src={
                            showHeatmap && results.heatmap_base64
                              ? `data:image/png;base64,${results.heatmap_base64}`
                              : showSegmentation && results.segmentation_base64
                                ? `data:image/png;base64,${results.segmentation_base64}`
                                : previewUrl
                          }
                          alt="Fundus"
                          className="absolute inset-0 w-full h-full object-contain"
                        />
                      </div>
                    </div>
                  )}

                  {/* Diagnosis & Risk Cards */}
                  <div className="space-y-4">
                    {/* DR Grade Card */}
                    {results.diabetic_retinopathy && (
                      <div
                        className={`rounded-lg border-l-4 p-5 bg-zinc-900 border border-zinc-800 ${
                          results.diabetic_retinopathy.grade === 0
                            ? "border-l-emerald-500"
                            : results.diabetic_retinopathy.grade <= 2
                              ? "border-l-amber-500"
                              : "border-l-red-500"
                        }`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <span className="text-[12px] font-semibold text-zinc-500 uppercase tracking-wider">
                              Diagnosis
                            </span>
                            <div className="text-[18px] font-bold text-zinc-100 mt-1 leading-tight">
                              {results.diabetic_retinopathy.grade_name}
                            </div>
                          </div>
                          <span
                            className={`px-2.5 py-1 rounded-full text-[11px] font-bold ${
                              results.diabetic_retinopathy.grade === 0
                                ? "bg-emerald-500/15 text-emerald-400"
                                : results.diabetic_retinopathy.grade <= 2
                                  ? "bg-amber-500/15 text-amber-400"
                                  : "bg-red-500/15 text-red-400"
                            }`}
                          >
                            Grade {results.diabetic_retinopathy.grade}
                          </span>
                        </div>

                        <div className="flex items-center gap-3 mt-4">
                          <div className="flex-1 bg-zinc-800 rounded-full h-2 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                results.diabetic_retinopathy.grade === 0
                                  ? "bg-emerald-500"
                                  : results.diabetic_retinopathy.grade <= 2
                                    ? "bg-amber-500"
                                    : "bg-red-500"
                              }`}
                              style={{
                                width: `${results.diabetic_retinopathy.probability * 100}%`,
                              }}
                            />
                          </div>
                          <span className="text-[11px] font-medium text-zinc-400">
                            {(
                              results.diabetic_retinopathy.probability * 100
                            ).toFixed(0)}
                            % Conf.
                          </span>
                        </div>

                        <div className="mt-3 text-[11px] font-medium text-zinc-300 bg-zinc-800/50 p-2 rounded-lg border border-zinc-700/50">
                          Action: {results.diabetic_retinopathy.clinical_action}
                        </div>

                        {/* 4-2-1 Rule */}
                        {results.diabetic_retinopathy.four_two_one_rule && (
                          <div className="mt-3">
                            <FourTwoOneRuleDisplay
                              rule={
                                results.diabetic_retinopathy.four_two_one_rule
                              }
                            />
                          </div>
                        )}
                      </div>
                    )}

                    {/* DME Card */}
                    {results.diabetic_macular_edema && (
                      <DMECard dme={results.diabetic_macular_edema} />
                    )}

                    {/* Risk Assessment Card */}
                    {results.risk_assessment && (
                      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <div className="text-[12px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">
                              Risk Analysis
                            </div>
                            <div className="flex items-baseline gap-2">
                              <span
                                className={`text-[24px] font-bold ${
                                  results.risk_assessment.category ===
                                    "minimal" ||
                                  results.risk_assessment.category === "low"
                                    ? "text-emerald-400"
                                    : results.risk_assessment.category ===
                                        "moderate"
                                      ? "text-amber-400"
                                      : "text-red-400"
                                }`}
                              >
                                {results.risk_assessment.overall_score.toFixed(
                                  0,
                                )}
                              </span>
                              <span className="text-[13px] font-medium text-zinc-400 capitalize">
                                {results.risk_assessment.category} Risk
                              </span>
                            </div>
                            {/* Confidence Interval */}
                            <div className="text-[9px] text-zinc-500 mt-1">
                              95% CI: [
                              {results.risk_assessment.confidence_interval_95[0].toFixed(
                                1,
                              )}
                              ,{" "}
                              {results.risk_assessment.confidence_interval_95[1].toFixed(
                                1,
                              )}
                              ]
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-[10px] text-zinc-500 mb-1">
                              Primary Indicator
                            </div>
                            <div className="text-[11px] font-medium text-zinc-300 max-w-[120px] leading-tight">
                              {results.risk_assessment.primary_finding}
                            </div>
                          </div>
                        </div>
                        {/* Contributing Factors */}
                        <ContributingFactorsPanel
                          factors={results.risk_assessment.contributing_factors}
                        />
                      </div>
                    )}

                    {/* Systemic Risk Indicators */}
                    {results.risk_assessment?.systemic_risk_indicators && (
                      <SystemicRiskPanel
                        indicators={
                          results.risk_assessment.systemic_risk_indicators
                        }
                      />
                    )}
                  </div>
                </div>

                {/* Biomarkers Grid */}
                {results.biomarkers && (
                  <div className="bg-zinc-900 rounded-lg border border-zinc-800">
                    <div className="px-5 py-4 border-b border-zinc-800 flex items-center gap-2">
                      <Activity className="h-4 w-4 text-cyan-400" />
                      <h3 className="text-[13px] font-semibold text-zinc-100">
                        Quantitative Biomarkers
                      </h3>
                    </div>
                    <div className="p-5">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-6">
                        {/* Vessels */}
                        <div className="space-y-3">
                          <div className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
                            Retinal Vessels
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <BiomarkerCard
                              label="Tortuosity"
                              biomarker={
                                results.biomarkers.vessels.tortuosity_index
                              }
                              icon={Activity}
                            />
                            <BiomarkerCard
                              label="AV Ratio"
                              biomarker={results.biomarkers.vessels.av_ratio}
                              icon={Target}
                            />
                            <BiomarkerCard
                              label="Density"
                              biomarker={
                                results.biomarkers.vessels.vessel_density
                              }
                              icon={Droplet}
                            />
                            <BiomarkerCard
                              label="Fractal D"
                              biomarker={
                                results.biomarkers.vessels.fractal_dimension
                              }
                              icon={Layers}
                            />
                          </div>
                        </div>

                        {/* Structural */}
                        <div className="space-y-3">
                          <div className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
                            Structural & Lesions
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <BiomarkerCard
                              label="CDR"
                              biomarker={
                                results.biomarkers.optic_disc.cup_disc_ratio
                              }
                              icon={Eye}
                            />
                            <BiomarkerCard
                              label="RNFL"
                              biomarker={
                                results.biomarkers.optic_disc.rnfl_thickness
                              }
                              icon={Brain}
                            />
                            <BiomarkerCard
                              label="Hem/MA"
                              biomarker={
                                results.biomarkers.lesions.hemorrhage_count
                              }
                              icon={XCircle}
                            />
                            <BiomarkerCard
                              label="Macula"
                              biomarker={results.biomarkers.macula.thickness}
                              icon={HeartPulse}
                            />
                          </div>
                        </div>
                      </div>

                      {/* Pathology Flags */}
                      <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-zinc-700/50">
                        {results.biomarkers.lesions
                          .neovascularization_detected && (
                          <span className="px-2 py-1 bg-red-500/15 text-red-400 text-[10px] font-medium rounded">
                            Neovascularization
                          </span>
                        )}
                        {results.biomarkers.lesions.venous_beading_detected && (
                          <span className="px-2 py-1 bg-orange-500/15 text-orange-400 text-[10px] font-medium rounded">
                            Venous Beading
                          </span>
                        )}
                        {results.biomarkers.lesions.irma_detected && (
                          <span className="px-2 py-1 bg-amber-500/15 text-amber-400 text-[10px] font-medium rounded">
                            IRMA
                          </span>
                        )}
                        {results.biomarkers.optic_disc.notching_detected && (
                          <span className="px-2 py-1 bg-violet-500/15 text-violet-400 text-[10px] font-medium rounded">
                            Disc Notching
                          </span>
                        )}
                        {results.biomarkers.lesions.cotton_wool_spots > 0 && (
                          <span className="px-2 py-1 bg-zinc-700 text-zinc-300 text-[10px] font-medium rounded">
                            CWS: {results.biomarkers.lesions.cotton_wool_spots}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Clinical Findings */}
                {results.findings.length > 0 && (
                  <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
                    <div className="text-[13px] font-semibold text-zinc-100 mb-4 flex items-center gap-2">
                      <FileText className="h-4 w-4 text-zinc-400" />
                      Clinical Findings
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {results.findings.map((f, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-3 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50"
                        >
                          {f.severity === "normal" ? (
                            <CheckCircle2 className="h-4 w-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                          ) : f.severity === "mild" ? (
                            <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
                          ) : (
                            <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0 mt-0.5" />
                          )}
                          <div>
                            <div className="text-[12px] font-semibold text-zinc-200">
                              {f.finding_type}
                            </div>
                            <div className="text-[11px] text-zinc-500 mt-0.5">
                              {f.description}
                            </div>
                            {f.icd10_code && (
                              <div className="text-[10px] font-mono text-zinc-600 mt-1">
                                ICD-10: {f.icd10_code}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Sidebar (Right 1/3) */}
              <div className="lg:col-span-1 space-y-6">
                {/* Image Quality Panel */}
                <ImageQualityPanel quality={results.image_quality} />

                {/* Pipeline Progress */}
                <PipelineProgress pipelineState={results.pipeline_state} />

                {/* AI Explanation Panel */}
                <ExplanationPanel
                  pipeline="retinal"
                  results={{
                    diabetic_retinopathy: results.diabetic_retinopathy,
                    risk_assessment: results.risk_assessment,
                    biomarkers: results.biomarkers,
                    findings: results.findings,
                    recommendations: results.recommendations,
                  }}
                />

                {/* Recommendations */}
                {results.recommendations.length > 0 && (
                  <div className="bg-zinc-900 rounded-lg border border-cyan-500/30 p-5">
                    <div className="text-[12px] font-semibold text-zinc-100 mb-3 flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-cyan-400" />
                      Recommendations
                    </div>
                    <ul className="space-y-2.5">
                      {results.recommendations.map((rec, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2.5 text-[12px] text-zinc-400 leading-relaxed"
                        >
                          <span className="block w-1 h-1 rounded-full bg-cyan-400 mt-2 flex-shrink-0" />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Differential Diagnoses */}
                {results.differential_diagnoses &&
                  results.differential_diagnoses.length > 0 && (
                    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5">
                      <div className="text-[12px] font-semibold text-zinc-100 mb-3">
                        Differential Diagnoses
                      </div>
                      <div className="divide-y divide-zinc-700/50">
                        {results.differential_diagnoses.map((dx, i) => (
                          <div key={i} className="py-2 first:pt-0 last:pb-0">
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-[12px] font-medium text-zinc-200">
                                {dx.diagnosis}
                              </span>
                              <span className="text-[11px] font-bold text-zinc-400 bg-zinc-800 px-1.5 py-0.5 rounded">
                                {(dx.probability * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="text-[10px] font-mono text-zinc-500">
                              {dx.icd10_code}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            </div>

            {/* Reset Button */}
            <div className="flex justify-center mt-6">
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 bg-zinc-800 text-zinc-300 rounded-lg hover:bg-zinc-700 border border-zinc-700 transition font-medium text-[13px]"
              >
                <RefreshCw className="h-4 w-4" />
                Analyze Another Image
              </button>
            </div>
          </motion.div>
        ) : (
          /* Input State - Two Column Layout with Info Panel */
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* Left: Upload Card */}
            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6 flex flex-col h-full">
              <h2 className="text-[14px] font-semibold text-zinc-100 mb-4">
                Upload Fundus Image
              </h2>

              <div className="mb-6 space-y-3">
                <div className="flex items-start gap-3 text-[13px] text-zinc-400">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-cyan-500/15 text-cyan-400 flex items-center justify-center text-[10px] font-bold">
                    1
                  </span>
                  <p>Use a fundus camera to capture a clear retinal image.</p>
                </div>
                <div className="flex items-start gap-3 text-[13px] text-zinc-400">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-cyan-500/15 text-cyan-400 flex items-center justify-center text-[10px] font-bold">
                    2
                  </span>
                  <p>Ensure macular and optic disc are visible.</p>
                </div>
                <div className="flex items-start gap-3 text-[13px] text-zinc-400">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-cyan-500/15 text-cyan-400 flex items-center justify-center text-[10px] font-bold">
                    3
                  </span>
                  <p>Supported formats: JPEG, PNG, TIFF (Max 15MB).</p>
                </div>
              </div>

              {/* Upload Zone */}
              <div
                className={`flex-1 border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all min-h-[280px] ${
                  isDragging
                    ? "border-cyan-500 bg-cyan-500/10"
                    : error
                      ? "border-red-500/50 bg-red-500/10"
                      : "border-zinc-700 hover:border-cyan-500/50 hover:bg-zinc-800/50"
                }`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) =>
                    e.target.files?.[0] && handleFileSelect(e.target.files[0])
                  }
                />

                {isProcessing ? (
                  <div className="py-4">
                    <Loader2 className="h-10 w-10 text-cyan-400 mx-auto animate-spin mb-3" />
                    <div className="text-[14px] font-medium text-zinc-100">
                      Processing Image...
                    </div>
                    <div className="text-[12px] text-zinc-400 mt-1 max-w-[200px] mx-auto">
                      Running multi-layer neural network analysis
                    </div>
                  </div>
                ) : error ? (
                  <div className="py-4">
                    <XCircle className="h-10 w-10 text-red-400 mx-auto mb-3" />
                    <div className="text-[13px] font-medium text-red-400">
                      {error}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReset();
                      }}
                      className="mt-2 text-[11px] font-medium text-red-400 hover:text-red-300"
                    >
                      Try again
                    </button>
                  </div>
                ) : (
                  <div className="py-4">
                    <div className="w-12 h-12 bg-zinc-800 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Upload className="h-6 w-6 text-zinc-400" />
                    </div>
                    <div className="text-[14px] font-medium text-zinc-200">
                      {isDragging
                        ? "Drop your fundus image here"
                        : "Click to Browse or Drag File"}
                    </div>
                    <div className="text-[11px] text-zinc-500 mt-2">
                      Secure HIPAA-Compliant Upload
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Capabilities & Info Card */}
            <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
              <h3 className="text-[13px] font-semibold text-zinc-100 mb-4 flex items-center gap-2">
                <Eye className="h-4 w-4 text-cyan-400" />
                Analysis Capabilities
              </h3>

              <div className="space-y-4 mb-6">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-emerald-500/15">
                    <Target className="h-4 w-4 text-emerald-400" />
                  </div>
                  <div>
                    <div className="text-[12px] font-medium text-zinc-200">
                      DR Grading (ETDRS)
                    </div>
                    <div className="text-[11px] text-zinc-500 mt-0.5">
                      5-level diabetic retinopathy classification
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-cyan-500/15">
                    <Activity className="h-4 w-4 text-cyan-400" />
                  </div>
                  <div>
                    <div className="text-[12px] font-medium text-zinc-200">
                      Vessel Analysis
                    </div>
                    <div className="text-[11px] text-zinc-500 mt-0.5">
                      Tortuosity, AV ratio, density & fractal analysis
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-violet-500/15">
                    <Brain className="h-4 w-4 text-violet-400" />
                  </div>
                  <div>
                    <div className="text-[12px] font-medium text-zinc-200">
                      Lesion Detection
                    </div>
                    <div className="text-[11px] text-zinc-500 mt-0.5">
                      Microaneurysms, hemorrhages, exudates & more
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-amber-500/15">
                    <ImageIcon className="h-4 w-4 text-amber-400" />
                  </div>
                  <div>
                    <div className="text-[12px] font-medium text-zinc-200">
                      Explainability
                    </div>
                    <div className="text-[11px] text-zinc-500 mt-0.5">
                      Attention heatmaps & vessel segmentation
                    </div>
                  </div>
                </div>
              </div>

              {/* Model Info */}
              <div className="p-4 bg-zinc-800 rounded-lg border border-zinc-700">
                <div className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider mb-2">
                  AI Model
                </div>
                <div className="text-[13px] font-semibold text-zinc-100">
                  RetinaScan v4.0 Pipeline
                </div>
                <div className="text-[11px] text-zinc-500 mt-1">
                  Multi-stage deep learning pipeline for comprehensive retinal
                  analysis with biomarker extraction
                </div>
                <div className="flex items-center gap-4 mt-3">
                  <div className="text-[10px]">
                    <span className="text-zinc-500">DR Accuracy: </span>
                    <span className="font-medium text-emerald-400">93%</span>
                  </div>
                  <div className="text-[10px]">
                    <span className="text-zinc-500">Processing: </span>
                    <span className="font-medium text-cyan-400">&lt;2s</span>
                  </div>
                </div>
              </div>

              {/* Clinical Note */}
              <div className="mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
                <div className="flex items-start gap-2">
                  <Info className="h-3.5 w-3.5 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div className="text-[10px] text-blue-400 leading-relaxed">
                    <span className="font-medium">Clinical Note:</span> This
                    analysis follows ETDRS standards and should be reviewed by
                    an ophthalmologist. Not for primary diagnostic use.
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
