"use client";

/**
 * Radiology Assessment Component
 *
 * Complete chest X-ray analysis interface with:
 * - Image upload with drag-and-drop
 * - Real API integration with backend
 * - 18 pathology predictions display
 * - Interactive heatmap visualization
 * - Risk score gauge
 * - Quality metrics
 * - AI explanation panel
 * - Clinical recommendations
 */

import React, { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  Scan,
  AlertCircle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Activity,
  Heart,
  Stethoscope,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ImageIcon,
  Zap,
  Clock,
  FileCheck,
} from "lucide-react";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";
import { usePipelineStatus } from "@/components/pipeline";
import type {
  RadiologyAnalysisResponse,
  Finding,
  getSeverityColor,
  getRiskColor,
} from "@/types/radiology";

// Analysis states
type AnalysisState = "idle" | "uploading" | "processing" | "complete" | "error";

// Component props
interface RadiologyAssessmentProps {
  onProcessingChange?: (isProcessing: boolean) => void;
}

export function RadiologyAssessment({
  onProcessingChange,
}: RadiologyAssessmentProps) {
  // State management
  const [state, setState] = useState<AnalysisState>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<RadiologyAnalysisResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [showAllPredictions, setShowAllPredictions] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Pipeline status integration
  const { startPipeline, updatePipeline, completePipeline } =
    usePipelineStatus();

  // Handle file selection
  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG)");
      return;
    }

    setSelectedFile(file);
    setResults(null);
    setError(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

  // Handle input change
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Handle drag and drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Analyze image
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    setState("processing");
    setError(null);
    onProcessingChange?.(true);

    // Start pipeline tracking
    startPipeline("radiology", [
      "upload",
      "validate",
      "preprocess",
      "analyze",
      "score",
      "format",
    ]);
    updatePipeline("radiology", { currentStage: "Uploading X-Ray..." });

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      updatePipeline("radiology", { currentStage: "Analyzing..." });

      // Call API route
      const response = await fetch("/api/radiology/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error || `Analysis failed: ${response.status}`,
        );
      }

      const data: RadiologyAnalysisResponse = await response.json();

      if (!data.success) {
        throw new Error(data.error?.message || "Analysis failed");
      }

      setResults(data);
      setState("complete");

      // Update pipeline status
      const riskLevel = data.risk_level || "low";
      completePipeline("radiology", true, `${riskLevel.toUpperCase()} Risk`);
    } catch (err) {
      console.error("Radiology analysis error:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Analysis failed. Please try again.",
      );
      setState("error");
      completePipeline("radiology", false, "Error");
    } finally {
      onProcessingChange?.(false);
    }
  }, [
    selectedFile,
    startPipeline,
    updatePipeline,
    completePipeline,
    onProcessingChange,
  ]);

  // Reset state
  const handleReset = useCallback(() => {
    setState("idle");
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    setShowAllPredictions(false);
    onProcessingChange?.(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [onProcessingChange]);

  const isProcessing = state === "processing" || state === "uploading";

  // Render results view
  if (state === "complete" && results) {
    return (
      <div className="space-y-6">
        {/* Results Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Results Panel (Left 2/3) */}
          <div className="lg:col-span-2 space-y-6">
            {/* Image + Primary Finding */}
            <div className="bg-white rounded-xl border border-zinc-200 p-5 shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-[14px] font-semibold text-zinc-800 flex items-center gap-2">
                  <Scan className="h-4 w-4 text-blue-600" />
                  Radiological Examination
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-medium text-zinc-400">
                    {results.processing_time_ms}ms
                  </span>
                  <span className="text-[11px] font-medium text-zinc-500">
                    {selectedFile?.name}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* X-Ray Image with Heatmap */}
                <div className="relative bg-zinc-950 rounded-lg overflow-hidden min-h-[280px]">
                  {results.heatmap_base64 ? (
                    <img
                      src={`data:image/png;base64,${results.heatmap_base64}`}
                      alt="X-ray with heatmap"
                      className="absolute inset-0 w-full h-full object-contain"
                    />
                  ) : previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="X-ray"
                      className="absolute inset-0 w-full h-full object-contain"
                    />
                  ) : null}

                  {/* Heatmap legend */}
                  {results.heatmap_base64 && (
                    <div className="absolute bottom-2 right-2 bg-black/70 rounded-lg px-2 py-1 text-[10px] text-white">
                      Attention Heatmap
                    </div>
                  )}
                </div>

                {/* Primary Finding + Risk Score */}
                <div className="space-y-4">
                  {/* Risk Score Gauge */}
                  <RiskScoreGauge
                    score={results.risk_score || 0}
                    level={results.risk_level || "low"}
                  />

                  {/* Primary Finding */}
                  {results.primary_finding && (
                    <div
                      className={`p-4 rounded-lg border ${getSeverityColors(results.primary_finding.severity)}`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="font-semibold text-[14px] text-zinc-900">
                          {results.primary_finding.condition}
                        </div>
                        <span
                          className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${getSeverityBadge(results.primary_finding.severity)}`}
                        >
                          {results.primary_finding.severity}
                        </span>
                      </div>
                      <div className="text-[12px] text-zinc-600 leading-relaxed mb-3">
                        {results.primary_finding.description}
                      </div>
                      <ConfidenceBar
                        value={results.primary_finding.probability}
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* All Predictions Grid */}
            <div className="bg-white rounded-xl border border-zinc-200 p-5 shadow-sm">
              <button
                onClick={() => setShowAllPredictions(!showAllPredictions)}
                className="w-full flex justify-between items-center mb-4"
              >
                <h3 className="text-[13px] font-semibold text-zinc-800 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-zinc-500" />
                  All Pathology Predictions (18)
                </h3>
                {showAllPredictions ? (
                  <ChevronUp className="h-4 w-4 text-zinc-400" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-zinc-400" />
                )}
              </button>

              <AnimatePresence>
                {showAllPredictions && results.all_predictions && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(results.all_predictions)
                        .sort(([, a], [, b]) => b - a)
                        .map(([condition, probability]) => (
                          <PredictionCard
                            key={condition}
                            condition={condition}
                            probability={probability}
                          />
                        ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {!showAllPredictions && results.all_predictions && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.entries(results.all_predictions)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 4)
                    .map(([condition, probability]) => (
                      <PredictionCard
                        key={condition}
                        condition={condition}
                        probability={probability}
                      />
                    ))}
                </div>
              )}
            </div>

            {/* Quality Metrics */}
            {results.quality && (
              <div className="bg-white rounded-xl border border-zinc-200 p-5 shadow-sm">
                <h3 className="text-[13px] font-semibold text-zinc-800 mb-4 flex items-center gap-2">
                  <FileCheck className="h-4 w-4 text-zinc-500" />
                  Image Quality Assessment
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <QualityMetric
                    label="Quality"
                    value={results.quality.overall_quality}
                    status={
                      results.quality.overall_quality === "good"
                        ? "good"
                        : "warning"
                    }
                  />
                  <QualityMetric
                    label="Resolution"
                    value={results.quality.resolution || "N/A"}
                    status={
                      results.quality.resolution_adequate ? "good" : "warning"
                    }
                  />
                  <QualityMetric
                    label="Contrast"
                    value={
                      results.quality.contrast
                        ? `${(results.quality.contrast * 100).toFixed(0)}%`
                        : "N/A"
                    }
                    status={
                      results.quality.contrast && results.quality.contrast > 0.5
                        ? "good"
                        : "warning"
                    }
                  />
                  <QualityMetric
                    label="Usable"
                    value={results.quality.usable ? "Yes" : "No"}
                    status={results.quality.usable ? "good" : "error"}
                  />
                </div>
                {results.quality.issues.length > 0 && (
                  <div className="mt-4 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                    <div className="text-[11px] font-medium text-yellow-800 mb-1">
                      Quality Issues:
                    </div>
                    <ul className="text-[11px] text-yellow-700 space-y-1">
                      {results.quality.issues.map((issue, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <AlertTriangle className="h-3 w-3 mt-0.5 flex-shrink-0" />
                          {issue}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Pipeline Stages */}
            {results.stages_completed &&
              results.stages_completed.length > 0 && (
                <div className="bg-zinc-50 rounded-xl border border-zinc-200 p-4">
                  <div className="text-[11px] font-medium text-zinc-600 mb-3">
                    Pipeline Stages
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {results.stages_completed.map((stage, i) => (
                      <div
                        key={i}
                        className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-medium ${
                          stage.status === "success"
                            ? "bg-green-100 text-green-700"
                            : "bg-red-100 text-red-700"
                        }`}
                      >
                        {stage.status === "success" ? (
                          <CheckCircle2 className="h-3 w-3" />
                        ) : (
                          <AlertCircle className="h-3 w-3" />
                        )}
                        {stage.stage}
                        <span className="text-opacity-70 ml-1">
                          {stage.time_ms.toFixed(0)}ms
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
          </div>

          {/* Sidebar (Right 1/3) */}
          <div className="lg:col-span-1 space-y-6">
            {/* AI Explanation */}
            <ExplanationPanel
              pipeline="radiology"
              results={{
                diagnosis: results.findings,
                primary_finding: results.primary_finding,
                risk_level: results.risk_level,
                risk_score: results.risk_score,
              }}
              patientContext={undefined}
            />

            {/* Recommendations */}
            {results.recommendations && results.recommendations.length > 0 && (
              <div className="bg-blue-50/50 rounded-xl border border-blue-100 p-5">
                <div className="text-[12px] font-semibold text-blue-900 mb-3 flex items-center gap-2">
                  <Stethoscope className="h-4 w-4 text-blue-600" />
                  Clinical Recommendations
                </div>
                <ul className="space-y-2.5">
                  {results.recommendations.map((rec, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2.5 text-[12px] text-blue-800 leading-relaxed"
                    >
                      <span className="block w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 flex-shrink-0" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Findings Detail */}
            {results.findings && results.findings.length > 0 && (
              <div className="bg-white rounded-xl border border-zinc-200 p-5">
                <h3 className="text-[12px] font-semibold text-zinc-800 mb-3 flex items-center gap-2">
                  <Heart className="h-4 w-4 text-rose-500" />
                  Detailed Findings
                </h3>
                <div className="space-y-3">
                  {results.findings.map((finding, i) => (
                    <FindingCard key={i} finding={finding} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Analyze Another */}
        <div className="flex justify-center">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-5 py-2.5 bg-zinc-100 text-zinc-700 rounded-lg hover:bg-zinc-200 transition font-medium text-[13px]"
          >
            <RefreshCw className="h-4 w-4" />
            Analyze Another X-Ray
          </button>
        </div>
      </div>
    );
  }

  // Render upload/input view
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Upload Card */}
      <div className="bg-white rounded-xl border border-zinc-200 p-6">
        <h2 className="text-[14px] font-semibold text-zinc-900 mb-4">
          Upload Chest X-Ray
        </h2>

        {/* Instructions */}
        <div className="mb-6 space-y-3">
          <div className="flex items-start gap-3 text-[13px] text-zinc-600">
            <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
              1
            </span>
            <p>Ensure patient ID is redacted (HIPAA compliant)</p>
          </div>
          <div className="flex items-start gap-3 text-[13px] text-zinc-600">
            <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
              2
            </span>
            <p>Standard PA or AP view preferred</p>
          </div>
          <div className="flex items-start gap-3 text-[13px] text-zinc-600">
            <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
              3
            </span>
            <p>Supported: JPEG, PNG (Max 10MB)</p>
          </div>
        </div>

        {/* Upload Area */}
        <div className="relative min-h-[280px]">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            onChange={handleInputChange}
            className="hidden"
            id="xray-upload"
          />

          {isProcessing ? (
            <div className="h-full border-2 border-dashed border-blue-200 bg-blue-50 rounded-xl flex flex-col items-center justify-center p-6 min-h-[280px]">
              <Loader2 className="h-12 w-12 text-blue-600 animate-spin mb-4" />
              <div className="text-[14px] font-medium text-blue-900">
                Analyzing X-Ray...
              </div>
              <div className="text-[12px] text-blue-600 mt-1">
                Detecting abnormalities using AI
              </div>
              <div className="mt-4 flex gap-2">
                {["Preprocessing", "Detecting", "Scoring"].map((step, i) => (
                  <span
                    key={step}
                    className="px-2 py-1 bg-blue-100 text-blue-700 text-[10px] font-medium rounded-full animate-pulse"
                    style={{ animationDelay: `${i * 200}ms` }}
                  >
                    {step}
                  </span>
                ))}
              </div>
            </div>
          ) : previewUrl ? (
            <div className="h-full border border-zinc-200 rounded-xl overflow-hidden relative group min-h-[280px]">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-full object-contain bg-zinc-950"
              />
              <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <p className="text-white text-sm font-medium mb-4">
                  {selectedFile?.name}
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={handleAnalyze}
                    className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 shadow-lg flex items-center gap-2"
                  >
                    <Zap className="h-4 w-4" />
                    Run Analysis
                  </button>
                  <button
                    onClick={handleReset}
                    className="px-4 py-2.5 bg-white/10 text-white rounded-lg text-sm font-medium hover:bg-white/20 backdrop-blur-sm"
                  >
                    Clear
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <label
              htmlFor="xray-upload"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`h-full border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 cursor-pointer transition-all text-center min-h-[280px] ${
                isDragging
                  ? "border-blue-400 bg-blue-50"
                  : "border-zinc-200 hover:border-blue-400 hover:bg-zinc-50"
              }`}
            >
              <div
                className={`w-14 h-14 rounded-full flex items-center justify-center mb-4 ${
                  isDragging ? "bg-blue-100" : "bg-zinc-100"
                }`}
              >
                <Upload
                  className={`h-7 w-7 ${isDragging ? "text-blue-500" : "text-zinc-400"}`}
                />
              </div>
              <div className="text-[14px] font-medium text-zinc-700">
                {isDragging
                  ? "Drop your X-Ray here"
                  : "Click to Browse or Drag File"}
              </div>
              <div className="text-[11px] text-zinc-400 mt-2">
                High-resolution JPEG/PNG supported
              </div>
            </label>
          )}
        </div>

        {/* Error message */}
        {state === "error" && error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg"
          >
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-[13px] font-medium text-red-800">
                  Analysis Failed
                </div>
                <div className="text-[12px] text-red-700 mt-1">{error}</div>
                <button
                  onClick={handleReset}
                  className="mt-3 text-[12px] font-medium text-red-500 hover:text-red-700"
                >
                  Try Again
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Info Card */}
      <div className="bg-zinc-50 rounded-xl border border-zinc-200 p-6">
        <h3 className="text-[13px] font-semibold text-zinc-800 mb-4 flex items-center gap-2">
          <Scan className="h-4 w-4 text-blue-600" />
          Model Capabilities
        </h3>

        <div className="space-y-4 mb-6">
          <CapabilityItem
            icon={Activity}
            title="18 Pathology Detection"
            description="Pneumonia, Cardiomegaly, Effusion, and more"
          />
          <CapabilityItem
            icon={Heart}
            title="Cardiac Assessment"
            description="Heart size and mediastinum evaluation"
          />
          <CapabilityItem
            icon={AlertTriangle}
            title="Emergency Triage"
            description="Pneumothorax & Mass flagging"
          />
          <CapabilityItem
            icon={ImageIcon}
            title="Explainability"
            description="Grad-CAM attention heatmaps"
          />
        </div>

        {/* Model Info */}
        <div className="p-4 bg-white rounded-lg border border-zinc-200">
          <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-2">
            AI Model
          </div>
          <div className="text-[13px] font-semibold text-zinc-800">
            TorchXRayVision DenseNet121
          </div>
          <div className="text-[11px] text-zinc-500 mt-1">
            Trained on 800,000+ chest X-rays from 8 medical datasets
          </div>
          <div className="flex items-center gap-4 mt-3">
            <div className="text-[10px]">
              <span className="text-zinc-400">Accuracy: </span>
              <span className="font-medium text-green-600">92%</span>
            </div>
            <div className="text-[10px]">
              <span className="text-zinc-400">Processing: </span>
              <span className="font-medium text-blue-600">&lt;3s</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Helper Components ---

// Risk Score Gauge
function RiskScoreGauge({ score, level }: { score: number; level: string }) {
  const getGradient = () => {
    switch (level) {
      case "critical":
        return "from-red-500 to-red-600";
      case "high":
        return "from-orange-500 to-orange-600";
      case "moderate":
        return "from-yellow-500 to-yellow-600";
      default:
        return "from-green-500 to-green-600";
    }
  };

  return (
    <div className="p-4 bg-zinc-50 rounded-lg border border-zinc-100">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">
        Risk Score
      </div>
      <div className="flex items-end gap-3">
        <div
          className={`text-[32px] font-bold bg-gradient-to-r ${getGradient()} bg-clip-text text-transparent`}
        >
          {score.toFixed(1)}
        </div>
        <div
          className={`text-[12px] font-semibold uppercase px-2 py-1 rounded mb-1 ${
            level === "critical"
              ? "bg-red-100 text-red-700"
              : level === "high"
                ? "bg-orange-100 text-orange-700"
                : level === "moderate"
                  ? "bg-yellow-100 text-yellow-700"
                  : "bg-green-100 text-green-700"
          }`}
        >
          {level}
        </div>
      </div>
      <div className="mt-2 h-2 bg-zinc-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${getGradient()}`}
          style={{ width: `${Math.min(100, score)}%` }}
        />
      </div>
    </div>
  );
}

// Prediction Card
function PredictionCard({
  condition,
  probability,
}: {
  condition: string;
  probability: number;
}) {
  return (
    <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100">
      <div className="text-[11px] font-medium text-zinc-800 truncate mb-1">
        {condition.replace(/_/g, " ")}
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1.5 bg-zinc-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${
              probability > 50
                ? "bg-red-500"
                : probability > 25
                  ? "bg-yellow-500"
                  : "bg-green-500"
            }`}
            style={{ width: `${Math.min(100, probability)}%` }}
          />
        </div>
        <span className="text-[10px] font-medium text-zinc-500 w-10 text-right">
          {probability.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

// Confidence Bar
function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${
            value > 80
              ? "bg-green-500"
              : value > 50
                ? "bg-yellow-500"
                : "bg-red-500"
          }`}
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="text-[10px] font-medium text-zinc-500">
        {value.toFixed(0)}% confidence
      </span>
    </div>
  );
}

// Quality Metric
function QualityMetric({
  label,
  value,
  status,
}: {
  label: string;
  value: string;
  status: "good" | "warning" | "error";
}) {
  return (
    <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div
        className={`text-[14px] font-semibold ${
          status === "good"
            ? "text-green-600"
            : status === "warning"
              ? "text-yellow-600"
              : "text-red-600"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

// Finding Card
function FindingCard({ finding }: { finding: Finding }) {
  return (
    <div
      className={`p-3 rounded-lg border ${getSeverityColors(finding.severity)}`}
    >
      <div className="flex justify-between items-start mb-1">
        <div className="text-[12px] font-semibold text-zinc-800">
          {finding.condition}
        </div>
        {finding.is_critical && (
          <AlertTriangle className="h-3.5 w-3.5 text-red-500" />
        )}
      </div>
      <div className="text-[11px] text-zinc-600 leading-relaxed">
        {finding.description}
      </div>
      {finding.urgency && (
        <div className="mt-2 text-[10px] text-zinc-500">
          Urgency: <span className="font-medium">{finding.urgency}</span>
        </div>
      )}
    </div>
  );
}

// Capability Item
function CapabilityItem({
  icon: Icon,
  title,
  description,
}: {
  icon: any;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-8 h-8 rounded-lg bg-white border border-zinc-200 flex items-center justify-center text-zinc-500">
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <div className="text-[12px] font-medium text-zinc-900">{title}</div>
        <div className="text-[10px] text-zinc-500">{description}</div>
      </div>
    </div>
  );
}

// Helper functions
function getSeverityColors(severity: string): string {
  switch (severity) {
    case "normal":
      return "bg-green-50 border-green-200";
    case "minimal":
    case "low":
      return "bg-blue-50 border-blue-200";
    case "moderate":
    case "possible":
      return "bg-yellow-50 border-yellow-200";
    case "high":
    case "likely":
      return "bg-orange-50 border-orange-200";
    case "critical":
      return "bg-red-100 border-red-300";
    default:
      return "bg-zinc-50 border-zinc-200";
  }
}

function getSeverityBadge(severity: string): string {
  switch (severity) {
    case "normal":
      return "bg-green-100 text-green-700";
    case "minimal":
    case "low":
      return "bg-blue-100 text-blue-700";
    case "moderate":
    case "possible":
      return "bg-yellow-100 text-yellow-700";
    case "high":
    case "likely":
      return "bg-orange-100 text-orange-700";
    case "critical":
      return "bg-red-200 text-red-800";
    default:
      return "bg-zinc-100 text-zinc-700";
  }
}
