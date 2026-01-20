"use client";

import { useState } from "react";
import {
  Upload,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Camera,
  Info,
  Activity,
  TrendingUp,
  Shield,
  AlertCircle,
  XCircle,
} from "lucide-react";
import { usePipelineStatus } from "@/components/pipeline";

interface ABCDEDetail {
  score: number;
  is_concerning: boolean;
  classification: string;
  num_colors?: number;
  has_blue_white_veil?: boolean;
  value_mm?: number;
}

interface ABCDEDetails {
  asymmetry: ABCDEDetail;
  border: ABCDEDetail;
  color: ABCDEDetail;
  diameter: ABCDEDetail;
  evolution: ABCDEDetail;
}

interface Escalation {
  rule: string;
  action: string;
  reason: string;
  priority: number;
}

interface Explanation {
  summary: string;
  detailed?: string;
  recommendations: string[];
  disclaimers?: string[];
}

interface DermatologyResult {
  success: boolean;
  request_id: string;
  processing_time_ms: number;

  lesion_detected: boolean;
  lesion_confidence: number;
  geometry?: {
    area_mm2: number;
    diameter_mm: number;
    major_axis_mm: number;
    minor_axis_mm: number;
    circularity: number;
    asymmetry_index: number;
  };

  risk_tier: number;
  risk_tier_name: string;
  risk_score: number;
  urgency: string;
  action: string;
  escalations: Escalation[];

  melanoma_probability: number;
  melanoma_classification: string;
  malignancy_classification: string;
  benign_probability: number;
  malignant_probability: number;
  primary_subtype: string;
  subtype_probability: number;

  abcde_score: number;
  abcde_criteria_met: number;
  abcde_details: ABCDEDetails;

  explanation?: Explanation;
  visualizations?: {
    segmentation_overlay_base64?: string;
    heatmap_base64?: string;
  };

  image_quality: number;
  analysis_confidence: number;
  warnings: string[];

  // Error fields
  error_code?: string;
  error_title?: string;
  error_message?: string;
  error_action?: string;
  tips?: string[];
  recoverable?: boolean;
}

// Component props
interface DermatologyAssessmentProps {
  onProcessingChange?: (isProcessing: boolean) => void;
}

export function DermatologyAssessment({
  onProcessingChange,
}: DermatologyAssessmentProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<DermatologyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { startPipeline, updatePipeline, completePipeline } =
    usePipelineStatus();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);

      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    onProcessingChange?.(true);
    startPipeline("dermatology", [
      "upload",
      "validate",
      "preprocess",
      "segment",
      "analyze",
      "classify",
      "complete",
    ]);
    updatePipeline("dermatology", { currentStage: "Uploading Image..." });

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("generate_explanation", "true");
      formData.append("include_visualizations", "true");

      updatePipeline("dermatology", { currentStage: "Validating Image..." });

      const response = await fetch("/api/dermatology/analyze", {
        method: "POST",
        body: formData,
      });

      const data: DermatologyResult = await response.json();

      if (!data.success) {
        setError(data.error_message || "Analysis failed");
        completePipeline("dermatology", false, data.error_title || "Error");
      } else {
        setResult(data);
        completePipeline("dermatology", true, data.risk_tier_name);
      }
    } catch (err) {
      console.error("Analysis error:", err);
      setError(err instanceof Error ? err.message : "Analysis failed");
      completePipeline("dermatology", false, "Error");
    } finally {
      setIsAnalyzing(false);
      onProcessingChange?.(false);
    }
  };

  const getRiskColor = (tier: number) => {
    switch (tier) {
      case 5:
        return "text-green-600 bg-green-50 border-green-200";
      case 4:
        return "text-blue-600 bg-blue-50 border-blue-200";
      case 3:
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case 2:
        return "text-orange-600 bg-orange-50 border-orange-200";
      case 1:
        return "text-red-600 bg-red-50 border-red-200";
      default:
        return "text-zinc-600 bg-zinc-50 border-zinc-200";
    }
  };

  const getABCDEStatus = (detail: ABCDEDetail | undefined) => {
    if (!detail) {
      return { icon: CheckCircle2, color: "text-zinc-400", label: "Unknown" };
    }
    if (detail.is_concerning) {
      return { icon: XCircle, color: "text-red-500", label: "Concerning" };
    }
    return { icon: CheckCircle2, color: "text-green-500", label: "Normal" };
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="space-y-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-sm">
            <h2 className="text-lg font-medium text-zinc-100 mb-4 flex items-center gap-2">
              <Camera size={20} className="text-fuchsia-400" />
              Capture or Upload Image
            </h2>

            {/* File Upload Area */}
            <div className="relative">
              <input
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleFileSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                id="skin-upload"
              />
              <label
                htmlFor="skin-upload"
                className="flex flex-col items-center justify-center border-2 border-dashed border-zinc-700 rounded-lg p-8 hover:border-fuchsia-500/50 hover:bg-zinc-800/50 transition-all cursor-pointer bg-zinc-800/30"
              >
                <Sparkles size={48} className="text-zinc-500 mb-4" />
                <p className="text-sm font-medium text-zinc-200 mb-1">
                  Take photo or upload image
                </p>
                <p className="text-xs text-zinc-500">
                  Supports: JPEG, PNG, HEIC (Max 50MB)
                </p>
              </label>
            </div>

            {/* Preview */}
            {previewUrl && (
              <div className="mt-4">
                <div className="relative">
                  <img
                    src={previewUrl}
                    alt="Skin lesion preview"
                    className="w-full rounded-lg border border-zinc-700"
                  />
                  {result?.visualizations?.segmentation_overlay_base64 && (
                    <img
                      src={`data:image/png;base64,${result.visualizations.segmentation_overlay_base64}`}
                      alt="Segmentation overlay"
                      className="absolute inset-0 w-full h-full rounded-lg opacity-60"
                    />
                  )}
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <span className="text-xs text-zinc-400">
                    {selectedFile?.name}
                  </span>
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="px-4 py-2 bg-fuchsia-600 text-white rounded-lg text-sm font-medium hover:bg-fuchsia-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 shadow-sm"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 size={16} className="animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles size={16} />
                        Analyze Lesion
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Photography Tips */}
          <div className="bg-fuchsia-500/10 border border-fuchsia-500/30 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <Info
                size={20}
                className="text-fuchsia-400 mt-0.5 flex-shrink-0"
              />
              <div>
                <h3 className="text-sm font-medium text-fuchsia-300 mb-2">
                  Photography Tips
                </h3>
                <ul className="text-xs text-fuchsia-400/80 space-y-1">
                  <li>Use good lighting (natural light preferred)</li>
                  <li>Keep camera 6-12 inches from lesion</li>
                  <li>Center the lesion in the frame</li>
                  <li>Capture entire lesion with surrounding skin</li>
                  <li>Avoid shadows and reflections</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
              <div className="flex items-start gap-3">
                <AlertCircle
                  size={20}
                  className="text-red-400 mt-0.5 flex-shrink-0"
                />
                <div>
                  <h3 className="text-sm font-medium text-red-400 mb-1">
                    Analysis Error
                  </h3>
                  <p className="text-xs text-red-400/80">{error}</p>
                  {result?.tips && result.tips.length > 0 && (
                    <ul className="text-xs text-red-400/80 mt-2 space-y-1">
                      {result.tips.map((tip, i) => (
                        <li key={i}>{tip}</li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-sm">
            <h2 className="text-lg font-medium text-zinc-100 mb-4 flex items-center gap-2">
              <Shield size={20} className="text-fuchsia-400" />
              Analysis Results
            </h2>

            {!result && !isAnalyzing && (
              <div className="text-center py-12">
                <Sparkles size={48} className="text-zinc-600 mx-auto mb-4" />
                <p className="text-sm text-zinc-400">
                  Upload a skin lesion image to begin analysis
                </p>
              </div>
            )}

            {isAnalyzing && (
              <div className="text-center py-12">
                <Loader2
                  size={48}
                  className="text-fuchsia-400 mx-auto mb-4 animate-spin"
                />
                <p className="text-sm text-zinc-200 mb-2">
                  Analyzing skin lesion...
                </p>
                <p className="text-xs text-zinc-400">
                  Running ABCDE criteria and classification
                </p>
              </div>
            )}

            {result && result.success && (
              <div className="space-y-4">
                {/* Risk Tier */}
                <div
                  className={`border rounded-lg p-4 ${getRiskColor(result.risk_tier)}`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-base font-semibold mb-1">
                        Risk Level: {result.risk_tier_name}
                      </h3>
                      <p className="text-xs opacity-80">{result.action}</p>
                    </div>
                    <span className="text-2xl font-bold">
                      {result.risk_score.toFixed(0)}
                    </span>
                  </div>
                  <p className="text-xs font-medium">
                    Urgency: {result.urgency}
                  </p>
                </div>

                {/* Classification */}
                <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-zinc-900 mb-3">
                    Classification Results
                  </h4>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="bg-white rounded-lg p-3 border border-zinc-100">
                      <span className="text-zinc-500 block mb-1">
                        Primary Type
                      </span>
                      <span className="font-medium text-zinc-900 capitalize">
                        {result.primary_subtype.replace(/_/g, " ")}
                      </span>
                      <span className="text-zinc-500 ml-1">
                        ({formatPercentage(result.subtype_probability)})
                      </span>
                    </div>
                    <div className="bg-white rounded-lg p-3 border border-zinc-100">
                      <span className="text-zinc-500 block mb-1">
                        Melanoma Risk
                      </span>
                      <span
                        className={`font-medium capitalize ${
                          result.melanoma_probability > 0.5
                            ? "text-red-600"
                            : result.melanoma_probability > 0.25
                              ? "text-yellow-600"
                              : "text-green-600"
                        }`}
                      >
                        {result.melanoma_classification.replace(/_/g, " ")}
                      </span>
                      <span className="text-zinc-500 ml-1">
                        ({formatPercentage(result.melanoma_probability)})
                      </span>
                    </div>
                    <div className="bg-white rounded-lg p-3 border border-zinc-100">
                      <span className="text-zinc-500 block mb-1">Benign</span>
                      <span className="font-medium text-green-600">
                        {formatPercentage(result.benign_probability)}
                      </span>
                    </div>
                    <div className="bg-white rounded-lg p-3 border border-zinc-100">
                      <span className="text-zinc-500 block mb-1">
                        Malignant
                      </span>
                      <span
                        className={`font-medium ${
                          result.malignant_probability > 0.5
                            ? "text-red-600"
                            : "text-zinc-600"
                        }`}
                      >
                        {formatPercentage(result.malignant_probability)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Geometry */}
                {result.geometry && (
                  <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-zinc-900 mb-3">
                      Lesion Measurements
                    </h4>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="bg-white rounded-lg p-2 border border-zinc-100 text-center">
                        <span className="text-zinc-500 block">Diameter</span>
                        <span className="font-medium text-zinc-900">
                          {result.geometry.diameter_mm.toFixed(1)} mm
                        </span>
                      </div>
                      <div className="bg-white rounded-lg p-2 border border-zinc-100 text-center">
                        <span className="text-zinc-500 block">Area</span>
                        <span className="font-medium text-zinc-900">
                          {result.geometry.area_mm2.toFixed(1)} mm²
                        </span>
                      </div>
                      <div className="bg-white rounded-lg p-2 border border-zinc-100 text-center">
                        <span className="text-zinc-500 block">Circularity</span>
                        <span className="font-medium text-zinc-900">
                          {(result.geometry.circularity * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Recommendation */}
                {result.explanation && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-blue-700 mb-2">
                      Summary
                    </h4>
                    <p className="text-xs text-blue-600 mb-3">
                      {result.explanation.summary}
                    </p>
                    {result.explanation.recommendations && (
                      <ul className="text-xs text-blue-700 space-y-1">
                        {result.explanation.recommendations
                          .slice(0, 3)
                          .map((rec, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <CheckCircle2
                                size={12}
                                className="text-blue-500 mt-0.5"
                              />
                              {rec}
                            </li>
                          ))}
                      </ul>
                    )}
                  </div>
                )}

                {/* Escalations */}
                {result.escalations && result.escalations.length > 0 && (
                  <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-orange-700 mb-2 flex items-center gap-2">
                      <AlertTriangle size={14} />
                      Escalation Alerts
                    </h4>
                    <ul className="text-xs text-orange-700 space-y-1">
                      {result.escalations.map((esc, i) => (
                        <li key={i}>{esc.reason}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Warning */}
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle
                      size={16}
                      className="text-yellow-600 mt-0.5 flex-shrink-0"
                    />
                    <p className="text-xs text-yellow-700">
                      AI analysis is for screening purposes only. Always consult
                      a dermatologist for definitive diagnosis and treatment.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ABCDE Criteria Card */}
          {result && result.success && result.abcde_details && (
            <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-zinc-900">
                  ABCDE Criteria Assessment
                </h3>
                <span className="text-xs font-medium text-purple-600">
                  {result.abcde_criteria_met}/5 Concerning
                </span>
              </div>
              <div className="space-y-2">
                {[
                  {
                    letter: "A",
                    label: "Asymmetry",
                    detail: result.abcde_details?.asymmetry,
                  },
                  {
                    letter: "B",
                    label: "Border",
                    detail: result.abcde_details?.border,
                  },
                  {
                    letter: "C",
                    label: "Color",
                    detail: result.abcde_details?.color,
                  },
                  {
                    letter: "D",
                    label: "Diameter",
                    detail: result.abcde_details?.diameter,
                  },
                  {
                    letter: "E",
                    label: "Evolution",
                    detail: result.abcde_details?.evolution,
                  },
                ].map((criteria) => {
                  const status = getABCDEStatus(criteria.detail);
                  const StatusIcon = status.icon;
                  const classification =
                    criteria.detail?.classification ||
                    (criteria.letter === "C"
                      ? `${criteria.detail?.num_colors || 0} colors`
                      : criteria.letter === "D"
                        ? `${criteria.detail?.value_mm?.toFixed(1) || "?"} mm`
                        : "unknown");
                  return (
                    <div
                      key={criteria.letter}
                      className="flex items-center justify-between bg-zinc-50 rounded-lg p-3 border border-zinc-100"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-bold text-purple-600 w-6">
                          {criteria.letter}
                        </span>
                        <div>
                          <span className="text-xs font-medium text-zinc-900">
                            {criteria.label}
                          </span>
                          <span className="text-xs text-zinc-500 ml-2 capitalize">
                            (
                            {typeof classification === "string"
                              ? classification.replace(/_/g, " ")
                              : classification}
                            )
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-zinc-500">
                          {((criteria.detail?.score || 0) * 100).toFixed(0)}%
                        </span>
                        <StatusIcon size={16} className={status.color} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* ABCDE Score Bar */}
              <div className="mt-4">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-zinc-500">
                    Overall ABCDE Score
                  </span>
                  <span className="text-xs font-medium text-zinc-900">
                    {(result.abcde_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-zinc-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      result.abcde_score > 0.6
                        ? "bg-red-500"
                        : result.abcde_score > 0.4
                          ? "bg-yellow-500"
                          : "bg-green-500"
                    }`}
                    style={{ width: `${result.abcde_score * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
