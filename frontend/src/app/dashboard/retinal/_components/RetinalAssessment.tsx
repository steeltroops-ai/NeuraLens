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
} from "lucide-react";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";
import { usePipelineStatus } from "@/components/pipeline";

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
  resolution: [number, number];
  file_size_mb: number;
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
  diabetic_retinopathy: DiabeticRetinopathyResult;
  risk_assessment: RiskAssessment;
  findings: ClinicalFinding[];
  differential_diagnoses: DifferentialDiagnosis[];
  recommendations: string[];
  clinical_summary: string;
  heatmap_base64?: string;
  segmentation_base64?: string;
}

type AnalysisState = "idle" | "uploading" | "processing" | "complete" | "error";

// ============================================================================
// Biomarker Display Card
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
  const colors = {
    normal: {
      bg: "bg-green-50",
      border: "border-green-200",
      text: "text-green-700",
      badge: "bg-green-100",
    },
    borderline: {
      bg: "bg-yellow-50",
      border: "border-yellow-200",
      text: "text-yellow-700",
      badge: "bg-yellow-100",
    },
    abnormal: {
      bg: "bg-red-50",
      border: "border-red-200",
      text: "text-red-700",
      badge: "bg-red-100",
    },
  }[biomarker.status] || {
    bg: "bg-zinc-50",
    border: "border-zinc-200",
    text: "text-zinc-700",
    badge: "bg-zinc-100",
  };

  const formatValue = (v: number) => {
    if (v === Math.floor(v)) return v.toString();
    return v.toFixed(3);
  };

  return (
    <div className={`${colors.bg} ${colors.border} border rounded-lg p-3`}>
      <div className="flex items-center gap-1.5 mb-1.5">
        <Icon className={`h-3.5 w-3.5 ${colors.text}`} />
        <span className="text-[12px] font-medium text-zinc-700 truncate">
          {label}
        </span>
      </div>
      <div className="text-[16px] font-bold text-zinc-900 leading-tight">
        {formatValue(biomarker.value)}
      </div>
      <div className="flex items-center justify-between mt-1">
        <span className={`text-[11px] font-semibold ${colors.text} capitalize`}>
          {biomarker.status}
        </span>
        <span className="text-[10px] text-zinc-500 font-medium">
          {(biomarker.measurement_confidence * 100).toFixed(0)}%
        </span>
      </div>
      {biomarker.clinical_significance && (
        <div className="text-[10px] text-zinc-600 mt-1.5 leading-snug line-clamp-2">
          {biomarker.clinical_significance}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function RetinalAssessment() {
  const [state, setState] = useState<AnalysisState>("idle");
  const [results, setResults] = useState<RetinalAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
        formData.append("patient_id", "ANONYMOUS");

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
      }
    },
    [addLog, startPipeline, updatePipeline, completePipeline],
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
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-5 shadow-sm">
                <div className="flex items-start gap-4">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <FileText className="h-5 w-5 text-blue-700" />
                  </div>
                  <div>
                    <div className="text-[14px] font-bold text-blue-900 mb-1">
                      Clinical Assessment Summary
                    </div>
                    <div className="text-[13px] leading-relaxed text-blue-800">
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
                    <div className="bg-white rounded-xl border border-zinc-200 p-4 shadow-sm h-full flex flex-col">
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-[13px] font-semibold text-zinc-800">
                          Fundus Examination
                        </span>
                        <button
                          onClick={() => setShowHeatmap(!showHeatmap)}
                          className={`px-2.5 py-1 text-[11px] font-medium rounded-md transition-colors ${
                            showHeatmap
                              ? "bg-cyan-100 text-cyan-700 border border-cyan-200"
                              : "bg-zinc-100 text-zinc-600 border border-zinc-200 hover:bg-zinc-200"
                          }`}
                        >
                          {showHeatmap ? "Heatmap Active" : "Show Heatmap"}
                        </button>
                      </div>
                      <div className="relative flex-1 bg-zinc-950 rounded-lg overflow-hidden min-h-[240px] group">
                        <img
                          src={
                            showHeatmap && results.heatmap_base64
                              ? `data:image/png;base64,${results.heatmap_base64}`
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
                        className={`rounded-xl border p-5 shadow-sm ${
                          results.diabetic_retinopathy.grade === 0
                            ? "bg-green-50 border-green-200"
                            : results.diabetic_retinopathy.grade <= 2
                              ? "bg-yellow-50 border-yellow-200"
                              : "bg-red-50 border-red-200"
                        }`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <span className="text-[12px] font-semibold text-zinc-600 uppercase tracking-wider">
                              Diagnosis
                            </span>
                            <div className="text-[18px] font-bold text-zinc-900 mt-1 leading-tight">
                              {results.diabetic_retinopathy.grade_name}
                            </div>
                          </div>
                          <span
                            className={`px-2.5 py-1 rounded-full text-[11px] font-bold border ${
                              results.diabetic_retinopathy.grade === 0
                                ? "bg-white border-green-100 text-green-700"
                                : results.diabetic_retinopathy.grade <= 2
                                  ? "bg-white border-yellow-100 text-yellow-700"
                                  : "bg-white border-red-100 text-red-700"
                            }`}
                          >
                            Grade {results.diabetic_retinopathy.grade}
                          </span>
                        </div>

                        <div className="flex items-center gap-3 mt-4">
                          <div className="flex-1 bg-white/50 rounded-full h-2 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                results.diabetic_retinopathy.grade === 0
                                  ? "bg-green-500"
                                  : results.diabetic_retinopathy.grade <= 2
                                    ? "bg-yellow-500"
                                    : "bg-red-500"
                              }`}
                              style={{
                                width: `${results.diabetic_retinopathy.probability * 100}%`,
                              }}
                            />
                          </div>
                          <span className="text-[11px] font-medium text-zinc-600">
                            {(
                              results.diabetic_retinopathy.probability * 100
                            ).toFixed(0)}
                            % Conf.
                          </span>
                        </div>

                        <div className="mt-3 text-[11px] font-medium text-zinc-700 bg-white/60 p-2 rounded-lg border border-black/5">
                          Action: {results.diabetic_retinopathy.clinical_action}
                        </div>
                      </div>
                    )}

                    {/* Risk Assessment Card */}
                    {results.risk_assessment && (
                      <div className="bg-white rounded-xl border border-zinc-200 p-5 shadow-sm flex items-center justify-between gap-4">
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
                                  ? "text-green-600"
                                  : results.risk_assessment.category ===
                                      "moderate"
                                    ? "text-yellow-600"
                                    : "text-red-600"
                              }`}
                            >
                              {results.risk_assessment.overall_score.toFixed(0)}
                            </span>
                            <span className="text-[13px] font-medium text-zinc-600 capitalize">
                              {results.risk_assessment.category} Risk
                            </span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] text-zinc-400 mb-1">
                            Primary Indicator
                          </div>
                          <div className="text-[11px] font-medium text-zinc-700 max-w-[120px] leading-tight">
                            {results.risk_assessment.primary_finding}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Biomarkers Grid */}
                {results.biomarkers && (
                  <div className="bg-white rounded-xl border border-zinc-200 shadow-sm overflow-hidden">
                    <div className="px-5 py-4 border-b border-zinc-100 flex items-center gap-2 bg-zinc-50/50">
                      <Activity className="h-4 w-4 text-cyan-600" />
                      <h3 className="text-[13px] font-semibold text-zinc-800">
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
                    </div>
                  </div>
                )}

                {/* Clinical Findings */}
                {results.findings.length > 0 && (
                  <div className="bg-white rounded-xl border border-zinc-200 shadow-sm p-5">
                    <div className="text-[13px] font-semibold text-zinc-800 mb-4 flex items-center gap-2">
                      <FileText className="h-4 w-4 text-zinc-500" />
                      Clinical Findings
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {results.findings.map((f, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-3 p-3 bg-zinc-50 rounded-lg border border-zinc-100"
                        >
                          {f.severity === "normal" ? (
                            <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0 mt-0.5" />
                          ) : f.severity === "mild" ? (
                            <AlertTriangle className="h-4 w-4 text-yellow-500 flex-shrink-0 mt-0.5" />
                          ) : (
                            <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0 mt-0.5" />
                          )}
                          <div>
                            <div className="text-[12px] font-semibold text-zinc-700">
                              {f.finding_type}
                            </div>
                            <div className="text-[11px] text-zinc-500 mt-0.5">
                              {f.description}
                            </div>
                            {f.icd10_code && (
                              <div className="text-[10px] font-mono text-zinc-400 mt-1">
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
                  <div className="bg-blue-50/50 rounded-xl border border-blue-100 p-5">
                    <div className="text-[12px] font-semibold text-blue-900 mb-3 flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-blue-600" />
                      Recommendations
                    </div>
                    <ul className="space-y-2.5">
                      {results.recommendations.map((rec, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2.5 text-[12px] text-blue-800 leading-relaxed"
                        >
                          <span className="block w-1 h-1 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Differential Diagnoses */}
                {results.differential_diagnoses &&
                  results.differential_diagnoses.length > 0 && (
                    <div className="bg-white rounded-xl border border-zinc-200 p-5 shadow-sm">
                      <div className="text-[12px] font-semibold text-zinc-700 mb-3">
                        Differential Diagnoses
                      </div>
                      <div className="divide-y divide-zinc-100">
                        {results.differential_diagnoses.map((dx, i) => (
                          <div key={i} className="py-2 first:pt-0 last:pb-0">
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-[12px] font-medium text-zinc-800">
                                {dx.diagnosis}
                              </span>
                              <span className="text-[11px] font-bold text-zinc-500 bg-zinc-100 px-1.5 py-0.5 rounded">
                                {(dx.probability * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="text-[10px] font-mono text-zinc-400">
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
                className="flex items-center gap-2 px-4 py-2 bg-zinc-100 text-zinc-700 rounded-lg hover:bg-zinc-200 transition font-medium text-[13px]"
              >
                <RefreshCw className="h-4 w-4" />
                Analyze Another Image
              </button>
            </div>
          </motion.div>
        ) : (
          /* Input State - Matches Speech Analysis Clean Card Style */
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-2xl mx-auto"
          >
            {/* Left: Input Card */}
            <div className="bg-white rounded-xl border border-zinc-200 p-6 flex flex-col h-full">
              <h2 className="text-[14px] font-semibold text-zinc-900 mb-4">
                Upload Fundus Image
              </h2>

              <div className="mb-6 space-y-3">
                <div className="flex items-start gap-3 text-[13px] text-zinc-600">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
                    1
                  </span>
                  <p>Use a fundus camera to capture a clear retinal image.</p>
                </div>
                <div className="flex items-start gap-3 text-[13px] text-zinc-600">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
                    2
                  </span>
                  <p>Ensure macular and optic disc are visible.</p>
                </div>
                <div className="flex items-start gap-3 text-[13px] text-zinc-600">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
                    3
                  </span>
                  <p>Supported formats: JPEG, PNG, TIFF (Max 15MB).</p>
                </div>
              </div>

              {/* Upload Zone */}
              <div
                className={`flex-1 border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all ${
                  isDragging
                    ? "border-cyan-500 bg-cyan-50"
                    : error
                      ? "border-red-300 bg-red-50"
                      : "border-zinc-200 hover:border-cyan-400 hover:bg-zinc-50"
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
                    <Loader2 className="h-10 w-10 text-cyan-600 mx-auto animate-spin mb-3" />
                    <div className="text-[14px] font-medium text-zinc-900">
                      Processing Image...
                    </div>
                    <div className="text-[12px] text-zinc-500 mt-1 max-w-[200px] mx-auto">
                      Running multi-layer neural network analysis
                    </div>
                  </div>
                ) : error ? (
                  <div className="py-4">
                    <XCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
                    <div className="text-[13px] font-medium text-red-700">
                      {error}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReset();
                      }}
                      className="mt-2 text-[11px] font-medium text-red-600 hover:underline"
                    >
                      Try again
                    </button>
                  </div>
                ) : (
                  <div className="py-4">
                    <div className="w-12 h-12 bg-zinc-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Upload className="h-6 w-6 text-zinc-400" />
                    </div>
                    <div className="text-[14px] font-medium text-zinc-700">
                      Click to Browse or Drag File
                    </div>
                    <div className="text-[11px] text-zinc-400 mt-2">
                      Secure HIPAA-Compliant Upload
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
