"use client";

import { useState } from "react";
import {
  Upload,
  Heart,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Activity,
  Zap,
  TrendingUp,
  PlayCircle,
} from "lucide-react";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";
import { usePipelineStatus } from "@/components/pipeline";

// API Response interface matching backend schemas
interface CardiologyAnalysisResult {
  success: boolean;
  request_id: string;
  processing_time_ms: number;
  ecg_analysis: {
    rhythm_analysis: {
      classification: string;
      heart_rate_bpm: number;
      confidence: number;
      regularity: string;
      r_peaks_detected: number;
      rr_variability_cv?: number;
    };
    hrv_metrics: {
      time_domain: {
        rmssd_ms: number | null;
        sdnn_ms: number | null;
        pnn50_percent: number | null;
        mean_rr_ms: number | null;
        sdsd_ms: number | null;
        cv_rr_percent: number | null;
      };
      interpretation: {
        autonomic_balance: string;
        parasympathetic: string;
        sympathetic: string;
      };
    };
    intervals: {
      pr_interval_ms: number | null;
      qrs_duration_ms: number | null;
      qt_interval_ms: number | null;
      qtc_ms: number | null;
      all_normal: boolean;
    };
    arrhythmias_detected: Array<{
      type: string;
      confidence: number;
      urgency: string;
      count?: number;
      description: string;
    }>;
    signal_quality_score: number;
  };
  findings: Array<{
    id: string;
    type: string;
    title: string;
    severity: string;
    description: string;
    source: string;
    confidence?: number;
  }>;
  risk_assessment: {
    risk_score: number;
    risk_category: string;
    risk_factors: Array<{ factor: string; severity: string }>;
    confidence: number;
  };
  recommendations: string[];
  quality_assessment: {
    overall_quality: string;
    ecg_quality?: {
      signal_quality_score: number;
      snr_db?: number;
      usable_segments_percent: number;
      artifacts_detected: number;
    };
  };
  visualizations?: {
    ecg?: {
      waveform_data: number[];
      sample_rate: number;
      annotations: Array<{
        type: string;
        sample_index: number;
        time_sec?: number;
        label?: string;
      }>;
    };
  };
}

// Component props
interface CardiologyAssessmentProps {
  onProcessingChange?: (isProcessing: boolean) => void;
}

export function CardiologyAssessment({
  onProcessingChange,
}: CardiologyAssessmentProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<CardiologyAnalysisResult | null>(null);
  const [recordingMode, setRecordingMode] = useState<"upload" | "demo">(
    "upload",
  );
  const [error, setError] = useState<string | null>(null);

  const { startPipeline, updatePipeline, completePipeline } =
    usePipelineStatus();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    onProcessingChange?.(true);

    startPipeline("cardiology", [
      "upload",
      "process",
      "ecg",
      "analyze",
      "risk",
      "complete",
    ]);
    updatePipeline("cardiology", { currentStage: "Uploading ECG..." });

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      updatePipeline("cardiology", { currentStage: "Processing Signal..." });

      // Use frontend API proxy instead of direct backend call
      const response = await fetch(
        `/api/cardiology/analyze?include_waveform=true`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || errorData.error || "Analysis failed",
        );
      }

      updatePipeline("cardiology", { currentStage: "Analyzing Rhythm..." });

      const data: CardiologyAnalysisResult = await response.json();
      setResult(data);
      completePipeline(
        "cardiology",
        true,
        `Risk: ${data.risk_assessment.risk_score.toFixed(0)}`,
      );
    } catch (err) {
      console.error("ECG analysis failed:", err);
      const errMsg = err instanceof Error ? err.message : "Analysis failed";
      setError(errMsg);
      completePipeline("cardiology", false, "Failed");
    } finally {
      setIsAnalyzing(false);
      onProcessingChange?.(false);
    }
  };

  const handleDemo = async () => {
    setIsAnalyzing(true);
    setError(null);
    setSelectedFile(null);
    onProcessingChange?.(true);

    startPipeline("cardiology", [
      "generate",
      "process",
      "analyze",
      "risk",
      "complete",
    ]);
    updatePipeline("cardiology", {
      currentStage: "Generating Synthetic ECG...",
    });

    try {
      // Use frontend API proxy instead of direct backend call
      const response = await fetch(
        `/api/cardiology/demo?heart_rate=72&duration=10`,
        {
          method: "POST",
        },
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.error || "Demo failed");
      }

      updatePipeline("cardiology", { currentStage: "Analyzing Demo Data..." });

      const data: CardiologyAnalysisResult = await response.json();
      setResult(data);
      completePipeline("cardiology", true, "Demo Complete");
    } catch (err) {
      console.error("Demo analysis failed:", err);
      setError(err instanceof Error ? err.message : "Demo failed");
      completePipeline("cardiology", false, "Demo Failed");
    } finally {
      setIsAnalyzing(false);
      onProcessingChange?.(false);
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low":
        return "text-emerald-400 bg-emerald-500/15 border-emerald-500/30";
      case "moderate":
        return "text-amber-400 bg-amber-500/15 border-amber-500/30";
      case "high":
        return "text-orange-400 bg-orange-500/15 border-orange-500/30";
      case "critical":
        return "text-red-400 bg-red-500/15 border-red-500/30";
      default:
        return "text-emerald-400 bg-emerald-500/15 border-emerald-500/30";
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "normal":
        return "text-emerald-400";
      case "mild":
        return "text-amber-400";
      case "moderate":
        return "text-orange-400";
      case "severe":
        return "text-red-400";
      default:
        return "text-zinc-400";
    }
  };

  const getQualityColor = (quality: string) => {
    switch (quality) {
      case "excellent":
        return "text-emerald-400";
      case "good":
        return "text-cyan-400";
      case "fair":
        return "text-amber-400";
      case "poor":
        return "text-red-400";
      default:
        return "text-zinc-400";
    }
  };

  // Helper to draw waveform from visualization data
  const renderWaveformSVG = () => {
    if (
      !result?.visualizations?.ecg?.waveform_data ||
      result.visualizations.ecg.waveform_data.length === 0
    ) {
      return null;
    }

    const data = result.visualizations.ecg.waveform_data;
    const width = 400;
    const height = 100;
    const step = Math.max(1, Math.floor(data.length / width));

    // Normalize data
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const points = [];
    for (let i = 0; i < Math.min(data.length, width); i++) {
      const rawValue = data[i * step];
      // Use fallback if undefined
      const value =
        rawValue !== undefined ? rawValue : (data[data.length - 1] ?? 0);
      const y = ((value - min) / range) * (height - 10) + 5;
      points.push(`${i},${height - y}`);
    }

    return (
      <svg
        className="w-full h-full"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
      >
        <polyline
          points={points.join(" ")}
          stroke="#ef4444"
          strokeWidth="1.5"
          fill="none"
        />
        <line
          x1="0"
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="#e4e4e7"
          strokeWidth="0.5"
        />
      </svg>
    );
  };

  return (
    <div className="space-y-6">
      {/* Main Content */}
      {result && result.success ? (
        <div className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Results Panel (Left 2/3) */}
            <div className="lg:col-span-2 space-y-6">
              {/* Top Section: Waveform & Diagnosis */}
              <div className="space-y-6">
                {/* ECG Waveform Card */}
                {result.visualizations?.ecg?.waveform_data && (
                  <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4 ">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className="text-[13px] font-semibold text-zinc-200 flex items-center gap-2">
                        <Activity className="h-4 w-4 text-red-400" />
                        ECG Lead II Rhythm Strip
                      </h3>
                      <div className="text-[11px] font-medium text-zinc-500">
                        {result.visualizations.ecg.sample_rate} Hz | 10s Window
                      </div>
                    </div>
                    <div className="bg-zinc-50 rounded-lg p-4 h-48 flex items-center justify-center border border-zinc-700/50 relative overflow-hidden">
                      {renderWaveformSVG()}
                      {/* Grid overlay could go here */}
                    </div>
                  </div>
                )}

                {/* Diagnosis & Vitals Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Primary Diagnosis */}
                  <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5  flex flex-col justify-between">
                    <div>
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                          Primary Rhythm
                        </span>
                        <span
                          className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-full border ${getRiskColor(result.risk_assessment.risk_category)} bg-opacity-20`}
                        >
                          {result.risk_assessment.risk_category} Risk
                        </span>
                      </div>
                      <div className="text-[18px] font-bold text-zinc-100 leading-tight mb-1">
                        {result.ecg_analysis.rhythm_analysis.classification}
                      </div>
                    </div>

                    <div className="mt-4">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-red-500 to-pink-500"
                            style={{
                              width: `${result.ecg_analysis.rhythm_analysis.confidence * 100}%`,
                            }}
                          />
                        </div>
                        <span className="text-[11px] font-medium text-zinc-400">
                          {(
                            result.ecg_analysis.rhythm_analysis.confidence * 100
                          ).toFixed(0)}
                          % Conf.
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Vitals & Metrics */}
                  <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5 ">
                    <div className="grid grid-cols-2 gap-4 h-full">
                      <div className="flex flex-col justify-center">
                        <div className="flex items-center gap-2 text-[11px] font-medium text-zinc-500 mb-1">
                          <Heart className="h-3.5 w-3.5 text-red-500" /> Heart
                          Rate
                        </div>
                        <div className="text-[24px] font-bold text-zinc-100">
                          {result.ecg_analysis.rhythm_analysis.heart_rate_bpm}
                          <span className="text-[12px] font-medium text-zinc-400 ml-1">
                            bpm
                          </span>
                        </div>
                      </div>
                      <div className="flex flex-col justify-center pl-4 border-l border-zinc-700/50">
                        <div className="flex items-center gap-2 text-[11px] font-medium text-zinc-500 mb-1">
                          <TrendingUp className="h-3.5 w-3.5 text-blue-500" />{" "}
                          Regularity
                        </div>
                        <div className="text-[16px] font-semibold text-zinc-100 capitalize">
                          {result.ecg_analysis.rhythm_analysis.regularity.replace(
                            "_",
                            " ",
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* HRV & Intervals */}
              <div className="bg-zinc-900 rounded-lg border border-zinc-800  overflow-hidden">
                <div className="px-5 py-3 border-b border-zinc-700/50 bg-zinc-800/50 flex items-center justify-between">
                  <h3 className="text-[13px] font-semibold text-zinc-200 flex items-center gap-2">
                    <Zap className="h-4 w-4 text-purple-600" />
                    Advanced Metrics
                  </h3>
                  <span className="text-[10px] bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded text-zinc-500 ">
                    ANS:{" "}
                    {result.ecg_analysis.hrv_metrics.interpretation.autonomic_balance.replace(
                      "_",
                      " ",
                    )}
                  </span>
                </div>
                <div className="p-5">
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <BiomarkerCard
                      label="QRS Duration"
                      value={result.ecg_analysis.intervals.qrs_duration_ms}
                      unit="ms"
                      normalRange={[80, 120]}
                    />
                    <BiomarkerCard
                      label="QTc Interval"
                      value={result.ecg_analysis.intervals.qtc_ms}
                      unit="ms"
                      normalRange={[350, 450]}
                    />
                    <BiomarkerCard
                      label="HRV (RMSSD)"
                      value={
                        result.ecg_analysis.hrv_metrics.time_domain.rmssd_ms
                      }
                      unit="ms"
                      normalRange={[25, 60]}
                    />
                    <BiomarkerCard
                      label="HRV (SDNN)"
                      value={
                        result.ecg_analysis.hrv_metrics.time_domain.sdnn_ms
                      }
                      unit="ms"
                      normalRange={[50, 120]}
                    />
                  </div>
                </div>
              </div>

              {/* Detailed Clinical Findings */}
              {result.findings.length > 0 && (
                <div className="bg-zinc-900 rounded-lg border border-zinc-800  p-5">
                  <h3 className="text-[13px] font-semibold text-zinc-200 mb-4 flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-zinc-500" />
                    Clinical Assessment Findings
                  </h3>
                  <div className="space-y-3">
                    {result.findings.map((finding) => (
                      <div
                        key={finding.id}
                        className="flex items-start gap-3 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50"
                      >
                        <CheckCircle2
                          className={`${getSeverityColor(finding.severity)} h-4 w-4 flex-shrink-0 mt-0.5`}
                        />
                        <div>
                          <div className="text-[12px] font-semibold text-zinc-300">
                            {finding.title}
                          </div>
                          <div className="text-[11px] text-zinc-500 mt-0.5 leading-relaxed">
                            {finding.description}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Sidebar (Right 1/3) */}
            <div className="lg:col-span-1 space-y-6">
              {/* Explanation Panel */}
              <ExplanationPanel
                pipeline="cardiology"
                results={result}
                patientContext={{ age: 65, sex: "male" }}
              />

              {/* Recommendations */}
              {result.recommendations.length > 0 && (
                <div className="bg-zinc-900 rounded-xl border border-cyan-500/30 p-5">
                  <div className="text-[12px] font-semibold text-zinc-100 mb-3 flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-cyan-400" />
                    Recommendations
                  </div>
                  <ul className="space-y-2.5">
                    {result.recommendations.map((rec, i) => (
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

              {/* Signal Quality */}
              <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-5 ">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-[12px] font-semibold text-zinc-300">
                    Signal Quality
                  </span>
                  <span
                    className={`text-[11px] font-bold capitalize ${getQualityColor(result.quality_assessment.overall_quality)}`}
                  >
                    {result.quality_assessment.overall_quality}
                  </span>
                </div>
                {result.quality_assessment.ecg_quality && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-[11px] text-zinc-500">
                      <span>Score</span>
                      <span>
                        {(
                          result.quality_assessment.ecg_quality
                            .signal_quality_score * 100
                        ).toFixed(0)}
                        %
                      </span>
                    </div>
                    <div className="w-full bg-zinc-100 rounded-full h-1.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          result.quality_assessment.overall_quality ===
                          "excellent"
                            ? "bg-emerald-500/150"
                            : result.quality_assessment.overall_quality ===
                                "good"
                              ? "bg-blue-500"
                              : result.quality_assessment.overall_quality ===
                                  "poor"
                                ? "bg-red-500/150"
                                : "bg-amber-500/150"
                        }`}
                        style={{
                          width: `${result.quality_assessment.ecg_quality.signal_quality_score * 100}%`,
                        }}
                      />
                    </div>
                    {result.quality_assessment.ecg_quality.snr_db && (
                      <div className="text-[10px] text-zinc-400 text-right mt-1">
                        SNR:{" "}
                        {result.quality_assessment.ecg_quality.snr_db.toFixed(
                          1,
                        )}{" "}
                        dB
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="flex justify-center mt-6">
            <button
              onClick={() => {
                setResult(null);
                setSelectedFile(null);
                setRecordingMode("upload");
              }}
              className="flex items-center gap-2 px-4 py-2 bg-zinc-100 text-zinc-300 rounded-lg hover:bg-zinc-200 transition font-medium text-[13px]"
            >
              <Activity className="h-4 w-4" />
              Analyze Another ECG
            </button>
          </div>
        </div>
      ) : (
        /* Input State - Clean 2-Column Card Style */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Card */}
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6 flex flex-col h-full">
            <h2 className="text-[14px] font-semibold text-zinc-100 mb-4 flex justify-between items-center">
              <span>Cardiology Assessment</span>
              {/* Mode Toggle inside header */}
              <div className="flex bg-zinc-800 rounded-lg p-0.5">
                <button
                  onClick={() => setRecordingMode("upload")}
                  className={`px-3 py-1 text-[11px] font-medium rounded-md transition-all ${recordingMode === "upload" ? "bg-zinc-700  text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`}
                >
                  Upload
                </button>
                <button
                  onClick={() => setRecordingMode("demo")}
                  className={`px-3 py-1 text-[11px] font-medium rounded-md transition-all ${recordingMode === "demo" ? "bg-zinc-700  text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`}
                >
                  Demo
                </button>
              </div>
            </h2>

            <div className="mb-6 space-y-3">
              <div className="flex items-start gap-3 text-[13px] text-zinc-400">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-rose-500/15 text-rose-400 flex items-center justify-center text-[10px] font-bold">
                  1
                </span>
                <p>Minimum 10s lead-II recording required.</p>
              </div>
              <div className="flex items-start gap-3 text-[13px] text-zinc-400">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-rose-500/15 text-rose-400 flex items-center justify-center text-[10px] font-bold">
                  2
                </span>
                <p>Supported formats: CSV, JSON, TXT (MIT-BIH).</p>
              </div>
              <div className="flex items-start gap-3 text-[13px] text-zinc-400">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-rose-500/15 text-rose-400 flex items-center justify-center text-[10px] font-bold">
                  3
                </span>
                <p>Ensure signal is free from major motion artifacts.</p>
              </div>
            </div>

            {/* Upload / Demo Area */}
            <div className="relative flex-1">
              {recordingMode === "upload" ? (
                <>
                  <input
                    type="file"
                    accept=".csv,.txt,.json"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="ecg-upload"
                  />
                  {isAnalyzing ? (
                    <div className="h-full border-2 border-dashed border-rose-500/30 bg-rose-500/10 rounded-xl flex flex-col items-center justify-center p-6 transition-all">
                      <Loader2 className="h-10 w-10 text-rose-400 animate-spin mb-3" />
                      <div className="text-[14px] font-medium text-zinc-100">
                        Analyzing ECG Signal...
                      </div>
                      <div className="text-[12px] text-rose-400 mt-1">
                        Detecting arrhythmias
                      </div>
                    </div>
                  ) : selectedFile ? (
                    <div className="h-full border border-zinc-700 rounded-xl bg-zinc-800/50 flex flex-col items-center justify-center p-6">
                      <Activity className="h-10 w-10 text-rose-400 mb-3" />
                      <p className="text-sm font-medium text-zinc-100 mb-1">
                        {selectedFile.name}
                      </p>
                      <p className="text-xs text-zinc-400 mb-4">
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={handleAnalyze}
                          className="px-4 py-2 bg-rose-600 text-white rounded-lg text-sm font-medium hover:bg-rose-700 shadow-lg"
                        >
                          Start Analysis
                        </button>
                        <button
                          onClick={() => setSelectedFile(null)}
                          className="px-4 py-2 bg-zinc-700 border border-zinc-600 text-zinc-200 rounded-lg text-sm font-medium hover:bg-zinc-600"
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                  ) : (
                    <label
                      htmlFor="ecg-upload"
                      className="h-full border-2 border-dashed border-zinc-700 rounded-xl flex flex-col items-center justify-center p-6 cursor-pointer hover:border-rose-500/50 hover:bg-zinc-800/50 transition-all text-center"
                    >
                      <div className="w-12 h-12 bg-zinc-800 rounded-full flex items-center justify-center mb-4">
                        <Upload className="h-6 w-6 text-zinc-400" />
                      </div>
                      <div className="text-[14px] font-medium text-zinc-200">
                        Upload ECG File
                      </div>
                      <div className="text-[11px] text-zinc-500 mt-2">
                        Drag & Drop or Click
                      </div>
                    </label>
                  )}
                </>
              ) : (
                <div className="h-full border-2 border-dashed border-zinc-700 rounded-xl flex flex-col items-center justify-center p-6 bg-zinc-800/30">
                  <PlayCircle className="h-12 w-12 text-rose-400 mb-4" />
                  <div className="text-[14px] font-medium text-zinc-200 mb-2">
                    Run Demo Analysis
                  </div>
                  <p className="text-[11px] text-zinc-400 text-center max-w-[200px] mb-4">
                    Simulate a patient with Atrial Fibrillation or Arrhythmia
                    using synthetic data.
                  </p>
                  <button
                    onClick={handleDemo}
                    disabled={isAnalyzing}
                    className="px-6 py-2 bg-rose-600 text-white rounded-lg text-sm font-medium hover:bg-rose-700 shadow-md disabled:opacity-50 flex items-center gap-2"
                  >
                    {isAnalyzing ? (
                      <Loader2 className="animate-spin h-4 w-4" />
                    ) : (
                      <Zap className="h-4 w-4" />
                    )}
                    Generate & Analyze
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Info / Capabilities */}
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6 h-full">
            <h3 className="text-[13px] font-semibold text-zinc-200 mb-4 flex items-center gap-2">
              <Activity className="h-4 w-4 text-rose-400" />
              Pipeline Capabilities
            </h3>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-400">
                  <Heart className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-[12px] font-medium text-zinc-200">
                    Arrhythmia Detection
                  </div>
                  <div className="text-[10px] text-zinc-500">
                    AFib, PVC, Bradycardia, Tachycardia
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-400">
                  <Activity className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-[12px] font-medium text-zinc-200">
                    HRV Analysis
                  </div>
                  <div className="text-[10px] text-zinc-500">
                    Time-domain metrics (RMSSD, SDNN)
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-zinc-400">
                  <Zap className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-[12px] font-medium text-zinc-200">
                    Signal Quality
                  </div>
                  <div className="text-[10px] text-zinc-500">
                    Noise detection & artifact removal
                  </div>
                </div>
              </div>
            </div>

            {error && (
              <div className="mt-6 bg-red-500/150/10 border border-red-500/30 rounded-lg p-3">
                <div className="flex items-center gap-2 text-red-400 font-medium text-[12px] mb-1">
                  <AlertTriangle className="h-3 w-3" />
                  Analysis Error
                </div>
                <div className="text-[11px] text-red-400/80 leading-snug">
                  {error}
                </div>
                <button
                  onClick={() => setError(null)}
                  className="mt-2 text-[10px] font-medium text-red-400 hover:text-red-300"
                >
                  Dismiss
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Biomarker Card Component
interface BiomarkerCardProps {
  label: string;
  value: number | null;
  unit: string;
  normalRange: [number, number];
}

function BiomarkerCard({
  label,
  value,
  unit,
  normalRange,
}: BiomarkerCardProps) {
  const isNormal =
    value !== null && value >= normalRange[0] && value <= normalRange[1];
  const isLow = value !== null && value < normalRange[0];
  const isHigh = value !== null && value > normalRange[1];

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-2">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span
          className={`text-sm font-semibold ${
            value === null
              ? "text-zinc-400"
              : isNormal
                ? "text-emerald-400"
                : isLow
                  ? "text-cyan-400"
                  : "text-orange-400"
          }`}
        >
          {value !== null ? value.toFixed(1) : "N/A"}
        </span>
        <span className="text-xs text-zinc-400">{unit}</span>
      </div>
      <p className="text-[10px] text-zinc-400 mt-0.5">
        {isNormal ? "✓ Normal" : isLow ? "↓ Low" : isHigh ? "↑ High" : ""}
      </p>
    </div>
  );
}
