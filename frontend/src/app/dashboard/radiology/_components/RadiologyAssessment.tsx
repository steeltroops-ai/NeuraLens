"use client";

import { useState } from "react";
import {
  Upload,
  Scan,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Download,
  Share2,
  Activity,
  Zap,
  TrendingUp,
} from "lucide-react";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";
import { usePipelineStatus } from "@/components/pipeline";

interface DiagnosisResult {
  condition: string;
  confidence: number;
  severity: "normal" | "mild" | "moderate" | "severe" | "critical";
  description: string;
}

export function RadiologyAssessment() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<DiagnosisResult[] | null>(null);
  const { startPipeline, updatePipeline, completePipeline } =
    usePipelineStatus();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResults(null);

      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    startPipeline("radiology", [
      "upload",
      "process",
      "denoise",
      "segment",
      "classify",
      "localize",
      "complete",
    ]);
    updatePipeline("radiology", { currentStage: "Uploading X-Ray..." });

    // Simulate AI analysis (replace with actual API call)
    await new Promise((resolve) => setTimeout(resolve, 1000));
    updatePipeline("radiology", { currentStage: "Denoising & Enhancing..." });

    await new Promise((resolve) => setTimeout(resolve, 1000));
    updatePipeline("radiology", { currentStage: "Segmenting Lungs..." });

    await new Promise((resolve) => setTimeout(resolve, 1000));
    updatePipeline("radiology", {
      currentStage: "Classifying Abnormalities...",
    });

    // Mock results
    setResults([
      {
        condition: "Pneumonia",
        confidence: 87.5,
        severity: "moderate",
        description:
          "Bilateral infiltrates detected in lower lobes. Bacterial pneumonia suspected.",
      },
      {
        condition: "Pleural Effusion",
        confidence: 72.3,
        severity: "mild",
        description: "Small amount of fluid detected in right pleural space.",
      },
      {
        condition: "Normal Heart Size",
        confidence: 94.1,
        severity: "normal",
        description: "Cardiothoracic ratio within normal limits.",
      },
    ]);

    setIsAnalyzing(false);
    completePipeline("radiology", true, "Pneumonia");
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "normal":
        return "text-green-600 bg-green-50 border-green-200";
      case "mild":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "moderate":
        return "text-orange-600 bg-orange-50 border-orange-200";
      case "severe":
        return "text-red-500 bg-red-50 border-red-200";
      case "critical":
        return "text-red-700 bg-red-100 border-red-300";
      default:
        return "text-zinc-500 bg-zinc-50 border-zinc-200";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header removed and lifted to page.tsx */}

      {/* Main Content Grid */}

      <div className="space-y-6">
        {/* Main Content */}
        {results ? (
          <div className="space-y-4">
            {/* Results Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Results Panel (Left 2/3) */}
              <div className="lg:col-span-2 space-y-6">
                {/* Image & Primary Findings */}
                <div className="bg-white rounded-xl border border-zinc-200 p-4 shadow-sm">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-[13px] font-semibold text-zinc-800 flex items-center gap-2">
                      <Scan className="h-4 w-4 text-blue-600" />
                      Radiological Examination
                    </h3>
                    <div className="text-[11px] font-medium text-zinc-500">
                      {selectedFile?.name}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* X-Ray Image */}
                    <div className="relative bg-zinc-950 rounded-lg overflow-hidden min-h-[240px]">
                      {previewUrl && (
                        <img
                          src={previewUrl}
                          alt="X-ray"
                          className="absolute inset-0 w-full h-full object-contain"
                        />
                      )}
                    </div>

                    {/* Primary Diagnosis List */}
                    <div className="space-y-3">
                      {results.map((result, index) => (
                        <div
                          key={index}
                          className={`p-3 rounded-lg border ${getSeverityColor(
                            result.severity,
                          )
                            .replace("text-", "border-")
                            .replace(" bg-", " bg-opacity-20 bg-")} bg-white`}
                        >
                          <div className="flex justify-between items-start mb-1">
                            <div className="font-semibold text-[13px] text-zinc-900">
                              {result.condition}
                            </div>
                            <span
                              className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded border ${getSeverityColor(result.severity)}`}
                            >
                              {result.severity}
                            </span>
                          </div>
                          <div className="text-[11px] text-zinc-600 leading-snug mb-2">
                            {result.description}
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  result.confidence > 80
                                    ? "bg-green-500"
                                    : result.confidence > 50
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                }`}
                                style={{ width: `${result.confidence}%` }}
                              />
                            </div>
                            <span className="text-[10px] font-medium text-zinc-400">
                              {result.confidence.toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Detailed Metrics / Clinical Findings (Mocked for now as they aren't in the state) */}
                <div className="bg-white rounded-xl border border-zinc-200 shadow-sm p-5">
                  <div className="text-[13px] font-semibold text-zinc-800 mb-4 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-zinc-500" />
                    Quantitative Analysis
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100">
                      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">
                        CTR
                      </div>
                      <div className="text-[16px] font-bold text-zinc-900">
                        0.45
                      </div>
                      <div className="text-[10px] text-green-600 font-medium">
                        Normal
                      </div>
                    </div>
                    <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100">
                      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">
                        Lung Opacity
                      </div>
                      <div className="text-[16px] font-bold text-zinc-900">
                        12%
                      </div>
                      <div className="text-[10px] text-yellow-600 font-medium">
                        Mild Elev.
                      </div>
                    </div>
                    <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100">
                      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">
                        Diaphragm
                      </div>
                      <div className="text-[16px] font-bold text-zinc-900">
                        Clear
                      </div>
                      <div className="text-[10px] text-green-600 font-medium">
                        Normal
                      </div>
                    </div>
                    <div className="p-3 bg-zinc-50 rounded-lg border border-zinc-100">
                      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">
                        Hardware
                      </div>
                      <div className="text-[16px] font-bold text-zinc-900">
                        None
                      </div>
                      <div className="text-[10px] text-zinc-400 font-medium">
                        -
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sidebar (Right 1/3) */}
              <div className="lg:col-span-1 space-y-6">
                {/* Explanation Panel */}
                <ExplanationPanel
                  pipeline="radiology"
                  results={{ diagnosis: results }}
                  patientContext={{
                    age: 45,
                    sex: "male",
                    symptoms: ["Cough", "Fever"],
                  }}
                />

                {/* Recommendations */}
                <div className="bg-blue-50/50 rounded-xl border border-blue-100 p-5">
                  <div className="text-[12px] font-semibold text-blue-900 mb-3 flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-600" />
                    Recommendations
                  </div>
                  <ul className="space-y-2.5">
                    <li className="flex items-start gap-2.5 text-[12px] text-blue-800 leading-relaxed">
                      <span className="block w-1 h-1 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
                      Clinical correlation with physical exam recommended.
                    </li>
                    <li className="flex items-start gap-2.5 text-[12px] text-blue-800 leading-relaxed">
                      <span className="block w-1 h-1 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
                      Consider follow-up X-ray in 2-4 weeks to monitor
                      infiltrates.
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="flex justify-center mt-6">
              <button
                onClick={() => {
                  setResults(null);
                  setSelectedFile(null);
                  setPreviewUrl(null);
                }}
                className="flex items-center gap-2 px-4 py-2 bg-zinc-100 text-zinc-700 rounded-lg hover:bg-zinc-200 transition font-medium text-[13px]"
              >
                <Scan className="h-4 w-4" />
                Analyze Another scan
              </button>
            </div>
          </div>
        ) : (
          /* Input State - Clean Card Style */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Upload Card */}
            <div className="bg-white rounded-xl border border-zinc-200 p-6 flex flex-col h-full">
              <h2 className="text-[14px] font-semibold text-zinc-900 mb-4">
                Upload Chest X-Ray
              </h2>

              <div className="mb-6 space-y-3">
                <div className="flex items-start gap-3 text-[13px] text-zinc-600">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
                    1
                  </span>
                  <p>Ensure patient ID is redacted (HIPAA compliant).</p>
                </div>
                <div className="flex items-start gap-3 text-[13px] text-zinc-600">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
                    2
                  </span>
                  <p>Standard PA or AP view preferred.</p>
                </div>
                <div className="flex items-start gap-3 text-[13px] text-zinc-600">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center text-[10px] font-bold">
                    3
                  </span>
                  <p>Supported: DICOM, JPEG, PNG (Max 50MB).</p>
                </div>
              </div>

              <div className="relative flex-1">
                <input
                  type="file"
                  accept="image/*,.dcm"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="xray-upload"
                />

                {isAnalyzing ? (
                  <div className="h-full border-2 border-dashed border-blue-200 bg-blue-50 rounded-xl flex flex-col items-center justify-center p-6 transition-all">
                    <Loader2 className="h-10 w-10 text-blue-600 animate-spin mb-3" />
                    <div className="text-[14px] font-medium text-blue-900">
                      Analyzing X-Ray...
                    </div>
                    <div className="text-[12px] text-blue-600 mt-1">
                      Detecting abnormalities
                    </div>
                  </div>
                ) : previewUrl ? (
                  <div className="h-full border  border-zinc-200 rounded-xl overflow-hidden relative group">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="w-full h-full object-contain bg-zinc-950"
                    />
                    <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <p className="text-white text-sm font-medium mb-3">
                        {selectedFile?.name}
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={handleAnalyze}
                          className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 shadow-lg"
                        >
                          Run Analysis
                        </button>
                        <button
                          onClick={() => {
                            setSelectedFile(null);
                            setPreviewUrl(null);
                          }}
                          className="px-4 py-2 bg-white/10 text-white rounded-lg text-sm font-medium hover:bg-white/20 backdrop-blur-sm"
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <label
                    htmlFor="xray-upload"
                    className="h-full border-2 border-dashed border-zinc-200 rounded-xl flex flex-col items-center justify-center p-6 cursor-pointer hover:border-blue-400 hover:bg-zinc-50 transition-all text-center"
                  >
                    <div className="w-12 h-12 bg-zinc-100 rounded-full flex items-center justify-center mb-4">
                      <Upload className="h-6 w-6 text-zinc-400" />
                    </div>
                    <div className="text-[14px] font-medium text-zinc-700">
                      Click to Browse or Drag File
                    </div>
                    <div className="text-[11px] text-zinc-400 mt-2">
                      High-resolution DICOM supported
                    </div>
                  </label>
                )}
              </div>
            </div>

            {/* Info Card */}
            <div className="bg-zinc-50 rounded-xl border border-zinc-200 p-6 h-full">
              <h3 className="text-[13px] font-semibold text-zinc-800 mb-4 flex items-center gap-2">
                <Scan className="h-4 w-4 text-blue-600" />
                Model Capabilities
              </h3>
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-white border border-zinc-200 flex items-center justify-center text-zinc-500">
                    <Activity className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-[12px] font-medium text-zinc-900">
                      Pneumonia Detection
                    </div>
                    <div className="text-[10px] text-zinc-500">
                      Bacterial & Viral classification
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-white border border-zinc-200 flex items-center justify-center text-zinc-500">
                    <TrendingUp className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-[12px] font-medium text-zinc-900">
                      Nodule Identification
                    </div>
                    <div className="text-[10px] text-zinc-500">
                      Early stage detection &gt;2mm
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-white border border-zinc-200 flex items-center justify-center text-zinc-500">
                    <Zap className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-[12px] font-medium text-zinc-900">
                      Emergency Triage
                    </div>
                    <div className="text-[10px] text-zinc-500">
                      Pneumothorax & Effusion flagging
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
