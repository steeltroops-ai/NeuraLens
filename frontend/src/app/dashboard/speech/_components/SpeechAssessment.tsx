"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertCircle, Loader2 } from "lucide-react";
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
              patientContext={undefined} // You might want to get this from a store or context
            />
          </div>
        </div>
      ) : (
        <motion.div
          key="recorder"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="bg-white rounded-xl border border-zinc-200 p-6"
        >
          {/* Instructions */}
          <div className="mb-6">
            <h2 className="text-[14px] font-semibold text-zinc-900 mb-2">
              Recording Instructions
            </h2>
            <ul className="space-y-1.5 text-[13px] text-zinc-500">
              <li className="flex items-start gap-2">
                <span className="text-blue-600">1.</span>
                Find a quiet environment with minimal background noise
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600">2.</span>
                Speak naturally for 10-30 seconds (reading or conversation)
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600">3.</span>
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
              className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
                <div>
                  <div className="text-[13px] font-medium text-blue-800">
                    {state === "uploading"
                      ? "Uploading audio..."
                      : "Analyzing speech patterns..."}
                  </div>
                  <div className="text-[12px] text-blue-600">
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
              className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg"
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
        </motion.div>
      )}
    </AnimatePresence>
  );
}
