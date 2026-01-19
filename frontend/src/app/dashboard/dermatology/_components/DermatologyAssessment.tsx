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
} from "lucide-react";
import { usePipelineStatus } from "@/components/pipeline";

interface SkinAnalysisResult {
  diagnosis: string;
  confidence: number;
  riskLevel: "benign" | "low" | "moderate" | "high" | "critical";
  recommendation: string;
  features: string[];
}

export function DermatologyAssessment() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SkinAnalysisResult | null>(null);
  const { startPipeline, updatePipeline, completePipeline } =
    usePipelineStatus();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);

      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    startPipeline("dermatology", [
      "upload",
      "process",
      "segment",
      "feature",
      "classify",
      "complete",
    ]);
    updatePipeline("dermatology", { currentStage: "Uploading Image..." });

    // Simulate AI analysis
    await new Promise((resolve) => setTimeout(resolve, 1000));
    updatePipeline("dermatology", { currentStage: "Segmenting Lesion..." });

    await new Promise((resolve) => setTimeout(resolve, 1000));
    updatePipeline("dermatology", { currentStage: "Analyzing ABCDE..." });

    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Mock result
    const mockResult: SkinAnalysisResult = {
      diagnosis: "Benign Nevus (Mole)",
      confidence: 92.3,
      riskLevel: "benign",
      recommendation:
        "No immediate action required. Continue routine monitoring. Schedule follow-up in 6 months.",
      features: [
        "Symmetrical borders",
        "Uniform color distribution",
        "Regular shape",
        "Size within normal range (<6mm)",
        "No irregular pigmentation",
      ],
    };

    setResult(mockResult);
    setIsAnalyzing(false);
    completePipeline("dermatology", true, "Benign");
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "benign":
        return "text-green-600 bg-green-50 border-green-200";
      case "low":
        return "text-blue-600 bg-blue-50 border-blue-200";
      case "moderate":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "high":
        return "text-orange-600 bg-orange-50 border-orange-200";
      case "critical":
        return "text-red-600 bg-red-50 border-red-200";
      default:
        return "text-zinc-600 bg-zinc-50 border-zinc-200";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header removed and lifted to page.tsx */}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="space-y-4">
          <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
            <h2 className="text-lg font-medium text-zinc-900 mb-4 flex items-center gap-2">
              <Camera size={20} className="text-purple-600" />
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
                className="flex flex-col items-center justify-center border-2 border-dashed border-zinc-300 rounded-lg p-8 hover:border-purple-500 hover:bg-purple-50 transition-all cursor-pointer bg-zinc-50"
              >
                <Sparkles size={48} className="text-zinc-400 mb-4" />
                <p className="text-sm font-medium text-zinc-900 mb-1">
                  Take photo or upload image
                </p>
                <p className="text-xs text-zinc-500">
                  Supports: JPEG, PNG (Max 10MB)
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
                    className="w-full rounded-lg border border-zinc-200"
                  />
                  {/* Overlay grid for analysis visualization */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-lg" />
                </div>
                <div className="mt-3 flex items-center justify-between">
                  <span className="text-xs text-zinc-500">
                    {selectedFile?.name}
                  </span>
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 shadow-sm"
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
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <Info size={20} className="text-blue-600 mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="text-sm font-medium text-blue-900 mb-2">
                  Photography Tips
                </h3>
                <ul className="text-xs text-blue-700 space-y-1">
                  <li>• Use good lighting (natural light preferred)</li>
                  <li>• Keep camera 6-12 inches from lesion</li>
                  <li>• Include ruler or coin for size reference</li>
                  <li>• Capture entire lesion with surrounding skin</li>
                  <li>• Avoid shadows and reflections</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
            <h2 className="text-lg font-medium text-zinc-900 mb-4 flex items-center gap-2">
              <CheckCircle2 size={20} className="text-green-500" />
              Analysis Results
            </h2>

            {!result && !isAnalyzing && (
              <div className="text-center py-12">
                <Sparkles size={48} className="text-zinc-300 mx-auto mb-4" />
                <p className="text-sm text-zinc-500">
                  Upload a skin lesion image to begin analysis
                </p>
              </div>
            )}

            {isAnalyzing && (
              <div className="text-center py-12">
                <Loader2
                  size={48}
                  className="text-purple-500 mx-auto mb-4 animate-spin"
                />
                <p className="text-sm text-zinc-900 mb-2">
                  Analyzing skin lesion...
                </p>
                <p className="text-xs text-zinc-500">
                  Running ABCDE criteria and pattern recognition
                </p>
              </div>
            )}

            {result && (
              <div className="space-y-4">
                {/* Primary Diagnosis */}
                <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-base font-semibold text-zinc-900 mb-1">
                        {result.diagnosis}
                      </h3>
                      <p className="text-xs text-zinc-500">
                        Primary Classification
                      </p>
                    </div>
                    <span
                      className={`text-xs font-semibold uppercase px-2.5 py-1 rounded border ${getRiskColor(result.riskLevel)}`}
                    >
                      {result.riskLevel}
                    </span>
                  </div>

                  {/* Confidence Bar */}
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-zinc-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-500"
                        style={{ width: `${result.confidence}%` }}
                      />
                    </div>
                    <span className="text-xs font-medium text-zinc-500">
                      {result.confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Detected Features */}
                <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-zinc-900 mb-3">
                    Detected Features
                  </h4>
                  <div className="space-y-2">
                    {result.features.map((feature, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <CheckCircle2
                          size={14}
                          className="text-green-500 flex-shrink-0"
                        />
                        <span className="text-xs text-zinc-600">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommendation */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-blue-700 mb-2">
                    Clinical Recommendation
                  </h4>
                  <p className="text-xs text-blue-600">
                    {result.recommendation}
                  </p>
                </div>

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
          {result && (
            <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-sm">
              <h3 className="text-sm font-medium text-zinc-900 mb-3">
                ABCDE Criteria Assessment
              </h3>
              <div className="space-y-2">
                {[
                  { letter: "A", label: "Asymmetry", status: "pass" },
                  { letter: "B", label: "Border", status: "pass" },
                  { letter: "C", label: "Color", status: "pass" },
                  { letter: "D", label: "Diameter", status: "pass" },
                  { letter: "E", label: "Evolution", status: "unknown" },
                ].map((criteria) => (
                  <div
                    key={criteria.letter}
                    className="flex items-center justify-between bg-zinc-50 rounded-lg p-2 border border-zinc-100"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-purple-600 w-6">
                        {criteria.letter}
                      </span>
                      <span className="text-xs text-zinc-600">
                        {criteria.label}
                      </span>
                    </div>
                    <span
                      className={`text-xs font-medium ${
                        criteria.status === "pass"
                          ? "text-green-600"
                          : "text-zinc-500"
                      }`}
                    >
                      {criteria.status === "pass" ? "✓ Normal" : "? Unknown"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
