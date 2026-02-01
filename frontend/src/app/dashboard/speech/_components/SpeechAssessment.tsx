"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  Loader2,
  Mic,
  FileAudio,
  Activity,
  Brain,
} from "lucide-react";
import { SpeechRecorder } from "./SpeechRecorder";
import { SpeechResultsPanel } from "./SpeechResultsPanel";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";
import { usePipelineStatus } from "@/components/pipeline";
import type { EnhancedSpeechAnalysisResponse } from "@/types/speech-enhanced";

type AnalysisState =
  | "idle"
  | "recording"
  | "uploading"
  | "processing"
  | "complete"
  | "error";

interface SpeechAssessmentProps {
  onProcessingChange?: (isProcessing: boolean) => void;
}

export default function SpeechAssessment({
  onProcessingChange,
}: SpeechAssessmentProps) {
  const [state, setState] = useState<AnalysisState>("idle");
  const [results, setResults] = useState<EnhancedSpeechAnalysisResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [uploadProgress, setUploadProgress] = useState(0);

  // Pipeline status bar integration
  const { startPipeline, updatePipeline, completePipeline } =
    usePipelineStatus();

  const analyzeAudio = useCallback(
    async (audioData: Blob | File) => {
      setState("processing");
      setError(null);
      setUploadProgress(0);
      onProcessingChange?.(true);

      // Start pipeline in status bar
      startPipeline("speech", [
        "upload",
        "extract",
        "analyze",
        "score",
        "output",
      ]);
      updatePipeline("speech", { currentStage: "Uploading..." });

      try {
        const formData = new FormData();
        if (audioData instanceof File) {
          formData.append("audio", audioData);
        } else {
          formData.append("audio", audioData, "recording.wav");
        }
        formData.append("session_id", `speech_${Date.now()}`);

        updatePipeline("speech", { currentStage: "Analyzing..." });

        // Use the Next.js API route as a proxy
        const response = await fetch("/api/speech/analyze", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.detail || `Analysis failed: ${response.status}`,
          );
        }

        const data: EnhancedSpeechAnalysisResponse = await response.json();

        if (data.status === "error") {
          throw new Error(data.error_message || "Analysis failed");
        }

        setResults(data);
        setState("complete");

        // Update status bar with completion
        const riskScore = data.risk_score?.toFixed(0) || "N/A";
        completePipeline("speech", true, `Risk: ${riskScore}`);
      } catch (err) {
        console.error("Speech analysis error:", err);
        setError(
          err instanceof Error
            ? err.message
            : "Analysis failed. Please try again.",
        );
        setState("error");
        completePipeline("speech", false, "Error");
      } finally {
        onProcessingChange?.(false);
      }
    },
    [startPipeline, updatePipeline, completePipeline, onProcessingChange],
  );

  const handleRecordingComplete = useCallback(
    (audioBlob: Blob) => {
      analyzeAudio(audioBlob);
    },
    [analyzeAudio],
  );

  const handleFileUpload = useCallback(
    (file: File) => {
      setState("uploading");
      analyzeAudio(file);
    },
    [analyzeAudio],
  );

  const handleReset = useCallback(() => {
    setState("idle");
    setResults(null);
    setError(null);
    setUploadProgress(0);
    onProcessingChange?.(false);
  }, [onProcessingChange]);

  const isProcessing = state === "processing" || state === "uploading";

  return (
    <AnimatePresence mode="wait">
      {state === "complete" && results ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <SpeechResultsPanel
              key="results"
              results={results}
              onReset={handleReset}
            />
          </div>
          <div className="lg:col-span-1">
            <ExplanationPanel
              pipeline="speech"
              results={results}
              patientContext={undefined}
            />
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Recorder Card */}
          <motion.div
            key="recorder"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="bg-zinc-900 rounded-xl border border-zinc-800 p-6"
          >
            {/* Instructions */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <div className="p-1.5 rounded bg-violet-500/15">
                  <Mic size={14} className="text-violet-400" />
                </div>
                <h2 className="text-[14px] font-semibold text-zinc-100">
                  Recording Instructions
                </h2>
              </div>
              <ul className="space-y-2 text-[13px] text-zinc-400">
                <li className="flex items-start gap-2">
                  <span className="text-violet-400 font-medium">1.</span>
                  Find a quiet environment with minimal background noise
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-violet-400 font-medium">2.</span>
                  Speak naturally for 10-30 seconds (reading or conversation)
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-violet-400 font-medium">3.</span>
                  Maintain consistent distance from microphone
                </li>
              </ul>
            </div>

            {/* Recorder */}
            <SpeechRecorder
              onRecordingComplete={handleRecordingComplete}
              onFileUpload={handleFileUpload}
              isProcessing={isProcessing}
              maxDuration={30}
            />

            {/* Processing State */}
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                  <div>
                    <div className="text-[13px] font-medium text-blue-300">
                      {state === "uploading"
                        ? "Uploading audio..."
                        : "Analyzing speech patterns..."}
                    </div>
                    <div className="text-[12px] text-blue-400/70">
                      Extracting biomarkers and calculating risk score
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Error State */}
            {state === "error" && error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg"
              >
                <div className="flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <div className="text-[13px] font-medium text-red-300">
                      Analysis Failed
                    </div>
                    <div className="text-[12px] text-red-400/80 mt-1">
                      {error}
                    </div>
                    <button
                      onClick={handleReset}
                      className="mt-3 text-[12px] font-medium text-red-400 hover:text-red-300 transition-colors"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>

          {/* Right: Capabilities & Info Card */}
          <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
            <h3 className="text-[13px] font-semibold text-zinc-100 mb-4 flex items-center gap-2">
              <Mic className="h-4 w-4 text-violet-400" />
              Analysis Capabilities
            </h3>

            <div className="space-y-4 mb-6">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/15">
                  <Activity className="h-4 w-4 text-emerald-400" />
                </div>
                <div>
                  <div className="text-[12px] font-medium text-zinc-200">
                    9 Voice Biomarkers
                  </div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    Jitter, shimmer, HNR, MFCCs & prosodic features
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-violet-500/15">
                  <Brain className="h-4 w-4 text-violet-400" />
                </div>
                <div>
                  <div className="text-[12px] font-medium text-zinc-200">
                    Neurological Screening
                  </div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    Parkinson's, dementia & speech disorder detection
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-cyan-500/15">
                  <FileAudio className="h-4 w-4 text-cyan-400" />
                </div>
                <div>
                  <div className="text-[12px] font-medium text-zinc-200">
                    Flexible Input
                  </div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    Record live or upload WAV, MP3, M4A, WebM, OGG
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-amber-500/15">
                  <AlertCircle className="h-4 w-4 text-amber-400" />
                </div>
                <div>
                  <div className="text-[12px] font-medium text-zinc-200">
                    Risk Scoring
                  </div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    Confidence intervals & clinical recommendations
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
                SpeechMD v2.0 Pipeline
              </div>
              <div className="text-[11px] text-zinc-500 mt-1">
                Deep learning models trained on clinical speech datasets for
                neurological assessment
              </div>
              <div className="flex items-center gap-4 mt-3">
                <div className="text-[10px]">
                  <span className="text-zinc-500">Accuracy: </span>
                  <span className="font-medium text-emerald-400">95.2%</span>
                </div>
                <div className="text-[10px]">
                  <span className="text-zinc-500">Processing: </span>
                  <span className="font-medium text-violet-400">&lt;3s</span>
                </div>
              </div>
            </div>

            {/* Clinical Note */}
            <div className="mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
              <div className="flex items-start gap-2">
                <AlertCircle className="h-3.5 w-3.5 text-blue-400 flex-shrink-0 mt-0.5" />
                <div className="text-[10px] text-blue-400 leading-relaxed">
                  <span className="font-medium">Clinical Note:</span> Voice
                  biomarkers are research-based indicators. Results should be
                  reviewed by a healthcare professional and not used as
                  standalone diagnosis.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </AnimatePresence>
  );
}
