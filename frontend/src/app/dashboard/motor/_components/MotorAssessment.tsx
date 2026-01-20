"use client";

import {
  Hand,
  Smartphone,
  Activity,
  Clock,
  TrendingUp,
  Play,
  Pause,
  Square,
  CheckCircle,
  AlertCircle,
  Loader2,
  RotateCcw,
} from "lucide-react";
import React, { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";

interface MotorAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

interface MotorAnalysisResult {
  sessionId: string;
  processingTime: number;
  confidence: number;
  riskScore: number;
  biomarkers: {
    tapFrequency: number;
    coordination: number;
    tremor: number;
    rhythmConsistency: number;
  };
  recommendations: string[];
  timestamp: Date;
}

interface TapTestState {
  isActive: boolean;
  countdown: number;
  tapCount: number;
  tapTimes: number[];
  startTime: number;
  testDuration: number;
  isComplete: boolean;
}

export default function MotorAssessment({
  onProcessingChange,
}: MotorAssessmentProps) {
  const [tapTestState, setTapTestState] = useState<TapTestState>({
    isActive: false,
    countdown: 0,
    tapCount: 0,
    tapTimes: [],
    startTime: 0,
    testDuration: 15,
    isComplete: false,
  });

  const [analysisResult, setAnalysisResult] =
    useState<MotorAnalysisResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Stop tap test
  const handleStopTapTest = useCallback(() => {
    setTapTestState((prev) => ({
      ...prev,
      isActive: false,
      countdown: 0,
      isComplete: true,
    }));
  }, []);

  // Tap test countdown effect
  useEffect(() => {
    if (tapTestState.isActive && tapTestState.countdown > 0) {
      const timer = setTimeout(() => {
        setTapTestState((prev) => ({ ...prev, countdown: prev.countdown - 1 }));
      }, 1000);
      return () => clearTimeout(timer);
    } else if (tapTestState.isActive && tapTestState.countdown === 0) {
      handleStopTapTest();
    }
    return undefined;
  }, [tapTestState.isActive, tapTestState.countdown, handleStopTapTest]);

  // Start tap test
  const startTapTest = useCallback(() => {
    setTapTestState({
      isActive: true,
      countdown: 15,
      tapCount: 0,
      tapTimes: [],
      startTime: Date.now(),
      testDuration: 15,
      isComplete: false,
    });
    setError(null);
  }, []);

  // Handle tap
  const handleTap = useCallback(() => {
    if (!tapTestState.isActive || tapTestState.countdown <= 0) return;

    const currentTime = Date.now();
    setTapTestState((prev) => ({
      ...prev,
      tapCount: prev.tapCount + 1,
      tapTimes: [...prev.tapTimes, currentTime - prev.startTime],
    }));
  }, [tapTestState.isActive, tapTestState.countdown]);

  // Process motor analysis
  const processMotorAnalysis = useCallback(async () => {
    if (tapTestState.tapTimes.length === 0) {
      setError(
        "No tap data available. Please complete the finger tapping test first.",
      );
      return;
    }

    setIsProcessing(true);
    setError(null);
    onProcessingChange(true);

    try {
      // Calculate tap metrics
      const avgInterval =
        tapTestState.tapTimes.length > 1
          ? tapTestState.tapTimes.reduce((acc, time, index) => {
              if (index === 0) return acc;
              const prevTime = tapTestState.tapTimes[index - 1];
              return acc + (time - (prevTime || 0));
            }, 0) /
            (tapTestState.tapTimes.length - 1)
          : 0;

      const tapFrequency = tapTestState.tapCount / tapTestState.testDuration;

      // Calculate rhythm consistency (coefficient of variation)
      const intervals = tapTestState.tapTimes.slice(1).map((time, index) => {
        const prevTime = tapTestState.tapTimes[index];
        return time - (prevTime || 0);
      });
      const intervalMean =
        intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const intervalStd = Math.sqrt(
        intervals.reduce(
          (acc, interval) => acc + Math.pow(interval - intervalMean, 2),
          0,
        ) / intervals.length,
      );
      const rhythmConsistency = 1 - intervalStd / intervalMean;

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Create analysis result
      const mockResult: MotorAnalysisResult = {
        sessionId: `motor_${Date.now()}`,
        processingTime: 134,
        confidence: 0.92,
        riskScore: Math.max(
          0,
          Math.min(
            1,
            (1 - rhythmConsistency) * 0.5 + (tapFrequency < 2 ? 0.3 : 0),
          ),
        ),
        biomarkers: {
          tapFrequency: Math.min(1, tapFrequency / 3), // Normalize to 0-1
          coordination: Math.min(1, Math.max(0, rhythmConsistency)),
          tremor: Math.max(0, Math.min(1, 1 - rhythmConsistency)),
          rhythmConsistency: Math.min(1, Math.max(0, rhythmConsistency)),
        },
        recommendations: [
          tapFrequency >= 2
            ? "Motor function appears normal with good tapping frequency"
            : "Consider follow-up for bradykinesia assessment",
          rhythmConsistency > 0.8
            ? "Good rhythm consistency observed"
            : "Irregular tapping pattern detected",
          "Continue regular physical activity to maintain motor skills",
        ],
        timestamp: new Date(),
      };

      setAnalysisResult(mockResult);
    } catch (error) {
      console.error("Motor analysis failed:", error);
      setError("Analysis failed. Please try again.");
    } finally {
      setIsProcessing(false);
      onProcessingChange(false);
    }
  }, [tapTestState, onProcessingChange]);

  // Reset assessment
  const resetAssessment = useCallback(() => {
    setTapTestState({
      isActive: false,
      countdown: 0,
      tapCount: 0,
      tapTimes: [],
      startTime: 0,
      testDuration: 15,
      isComplete: false,
    });
    setAnalysisResult(null);
    setError(null);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const progress = tapTestState.isActive
    ? ((tapTestState.testDuration - tapTestState.countdown) /
        tapTestState.testDuration) *
      100
    : tapTestState.isComplete
      ? 100
      : 0;
  return (
    <div className="space-y-6">
      {/* Finger Tapping Test */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <h2 className="mb-4 text-[16px] font-semibold text-zinc-100">
          Finger Tapping Test
        </h2>

        <AnimatePresence mode="wait">
          {!tapTestState.isActive &&
          !tapTestState.isComplete &&
          !analysisResult ? (
            <motion.div
              key="instructions"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center"
            >
              <div className="mb-6 rounded-lg border-2 border-dashed border-violet-500/50 bg-violet-500/10 p-8">
                <Hand className="mx-auto mb-4 h-16 w-16 text-violet-400" />
                <h3 className="mb-2 text-xl font-semibold text-zinc-100">
                  Ready to Start
                </h3>
                <p className="mb-4 text-zinc-400">
                  Tap the button below as quickly and consistently as possible
                  for 15 seconds. This test measures your motor function and
                  rhythm consistency.
                </p>
                <div className="mb-4 text-sm text-zinc-500">
                  <p>* Use your index finger</p>
                  <p>* Maintain steady rhythm</p>
                  <p>* Tap as fast as comfortable</p>
                </div>
                <button
                  onClick={startTapTest}
                  className="rounded-lg bg-violet-600 px-8 py-3 text-lg font-medium text-white transition-all duration-200 hover:scale-105 hover:bg-violet-700 inline-flex items-center"
                >
                  <Play className="mr-2 h-5 w-5" />
                  Start Tapping Test
                </button>
              </div>
            </motion.div>
          ) : tapTestState.isActive ? (
            <motion.div
              key="active-test"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center"
            >
              {/* Countdown Display */}
              <div className="mb-6">
                <div className="mx-auto mb-4 flex h-32 w-32 items-center justify-center rounded-full bg-gradient-to-r from-violet-500 to-violet-600 text-4xl font-bold text-white shadow-lg">
                  {tapTestState.countdown}
                </div>
                <div className="mb-2 h-2 rounded-full bg-zinc-800">
                  <div
                    className="h-2 rounded-full bg-gradient-to-r from-violet-500 to-violet-600 transition-all duration-1000"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-sm text-zinc-400">
                  {formatTime(
                    tapTestState.testDuration - tapTestState.countdown,
                  )}{" "}
                  / {formatTime(tapTestState.testDuration)}
                </p>
              </div>

              {/* Tap Button */}
              <div className="mb-6">
                <button
                  onClick={handleTap}
                  className="h-48 w-48 rounded-full bg-gradient-to-r from-violet-500 to-violet-600 text-white shadow-2xl transition-all duration-75 hover:scale-105 active:scale-95 active:shadow-lg"
                >
                  <div className="text-center">
                    <Hand className="mx-auto mb-2 h-12 w-12" />
                    <div className="text-2xl font-bold">
                      {tapTestState.tapCount}
                    </div>
                    <div className="text-sm">Taps</div>
                  </div>
                </button>
              </div>

              {/* Current Stats */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
                  <div className="font-medium text-zinc-200">Tap Count</div>
                  <div className="text-2xl font-bold text-violet-400">
                    {tapTestState.tapCount}
                  </div>
                </div>
                <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
                  <div className="font-medium text-zinc-200">Frequency</div>
                  <div className="text-2xl font-bold text-violet-400">
                    {tapTestState.tapCount > 0
                      ? (
                          (tapTestState.tapCount /
                            (tapTestState.testDuration -
                              tapTestState.countdown)) *
                          60
                        ).toFixed(1)
                      : "0.0"}{" "}
                    /min
                  </div>
                </div>
              </div>

              <button
                onClick={handleStopTapTest}
                className="mt-4 rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 inline-flex items-center"
              >
                <Square className="mr-2 h-4 w-4" />
                Stop Test
              </button>
            </motion.div>
          ) : tapTestState.isComplete && !analysisResult ? (
            <motion.div
              key="complete"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center"
            >
              <div className="mb-6 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-6">
                <CheckCircle className="mx-auto mb-4 h-12 w-12 text-emerald-400" />
                <h3 className="mb-2 text-lg font-semibold text-zinc-100">
                  Test Complete!
                </h3>
                <p className="mb-4 text-zinc-400">
                  You completed {tapTestState.tapCount} taps in{" "}
                  {tapTestState.testDuration} seconds
                </p>

                <div className="mb-4 grid grid-cols-2 gap-4 text-sm">
                  <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
                    <div className="font-medium text-zinc-200">Total Taps</div>
                    <div className="text-xl font-bold text-emerald-400">
                      {tapTestState.tapCount}
                    </div>
                  </div>
                  <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-3">
                    <div className="font-medium text-zinc-200">
                      Average Rate
                    </div>
                    <div className="text-xl font-bold text-emerald-400">
                      {(
                        (tapTestState.tapCount / tapTestState.testDuration) *
                        60
                      ).toFixed(1)}{" "}
                      /min
                    </div>
                  </div>
                </div>

                <div className="flex justify-center space-x-3">
                  <button
                    onClick={processMotorAnalysis}
                    disabled={isProcessing}
                    className="rounded-lg bg-emerald-600 px-6 py-2 font-medium text-white transition-colors hover:bg-emerald-700 disabled:opacity-50 inline-flex items-center"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      "Analyze Results"
                    )}
                  </button>
                  <button
                    onClick={resetAssessment}
                    className="rounded-lg border border-zinc-700 px-6 py-2 font-medium text-zinc-300 transition-colors hover:bg-zinc-800 inline-flex items-center"
                  >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Retry Test
                  </button>
                </div>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-4"
          >
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-5 w-5 text-red-400" />
              <div>
                <p className="font-medium text-zinc-100">Error</p>
                <p className="text-sm text-zinc-400">{error}</p>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Analysis Results */}
      {analysisResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-zinc-900 rounded-lg border border-zinc-800 p-6"
        >
          <div className="mb-6 flex items-center space-x-3">
            <div className="rounded-lg bg-gradient-to-r from-violet-500 to-violet-600 p-2">
              <CheckCircle className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">
                Motor Analysis Complete
              </h2>
              <p className="text-sm text-zinc-400">
                Processed in {analysisResult.processingTime}ms * Confidence:{" "}
                {(analysisResult.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Biomarkers Grid */}
          <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400">
                Tap Frequency
              </h3>
              <p className="text-2xl font-bold text-zinc-100">
                {(analysisResult.biomarkers.tapFrequency * 100).toFixed(1)}%
              </p>
              <div className="mt-2 h-2 rounded-full bg-zinc-700">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-violet-500 to-violet-600"
                  style={{
                    width: `${analysisResult.biomarkers.tapFrequency * 100}%`,
                  }}
                />
              </div>
              <p className="mt-1 text-xs text-zinc-500">
                {tapTestState.tapCount} taps in {tapTestState.testDuration}s
              </p>
            </div>

            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400">
                Coordination
              </h3>
              <p className="text-2xl font-bold text-zinc-100">
                {(analysisResult.biomarkers.coordination * 100).toFixed(1)}%
              </p>
              <div className="mt-2 h-2 rounded-full bg-zinc-700">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-600"
                  style={{
                    width: `${analysisResult.biomarkers.coordination * 100}%`,
                  }}
                />
              </div>
            </div>

            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400">
                Tremor Level
              </h3>
              <p className="text-2xl font-bold text-zinc-100">
                {(analysisResult.biomarkers.tremor * 100).toFixed(1)}%
              </p>
              <div className="mt-2 h-2 rounded-full bg-zinc-700">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-amber-500 to-red-600"
                  style={{
                    width: `${analysisResult.biomarkers.tremor * 100}%`,
                  }}
                />
              </div>
            </div>

            <div className="rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400">
                Rhythm Consistency
              </h3>
              <p className="text-2xl font-bold text-zinc-100">
                {(analysisResult.biomarkers.rhythmConsistency * 100).toFixed(1)}
                %
              </p>
              <div className="mt-2 h-2 rounded-full bg-zinc-700">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-blue-600"
                  style={{
                    width: `${analysisResult.biomarkers.rhythmConsistency * 100}%`,
                  }}
                />
              </div>
            </div>
          </div>

          {/* Risk Assessment */}
          <div className="mb-6 rounded-lg bg-zinc-800/50 border border-zinc-700/50 p-4">
            <h3 className="mb-3 text-sm font-medium text-zinc-400">
              Motor Function Assessment
            </h3>
            <div className="flex items-center space-x-4">
              <div className="flex-1">
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-300">Overall Risk Score</span>
                  <span className="font-medium text-zinc-100">
                    {(analysisResult.riskScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2 h-3 rounded-full bg-zinc-700">
                  <div
                    className={`h-3 rounded-full ${
                      analysisResult.riskScore < 0.3
                        ? "bg-gradient-to-r from-emerald-500 to-emerald-600"
                        : analysisResult.riskScore < 0.7
                          ? "bg-gradient-to-r from-amber-500 to-amber-600"
                          : "bg-gradient-to-r from-red-500 to-red-600"
                    }`}
                    style={{ width: `${analysisResult.riskScore * 100}%` }}
                  />
                </div>
              </div>
              <div
                className={`rounded-full px-3 py-1 text-xs font-medium ${
                  analysisResult.riskScore < 0.3
                    ? "bg-emerald-500/15 text-emerald-400"
                    : analysisResult.riskScore < 0.7
                      ? "bg-amber-500/15 text-amber-400"
                      : "bg-red-500/15 text-red-400"
                }`}
              >
                {analysisResult.riskScore < 0.3
                  ? "Normal"
                  : analysisResult.riskScore < 0.7
                    ? "Monitor"
                    : "Concern"}
              </div>
            </div>
          </div>

          {/* AI Explanation */}
          <ExplanationPanel
            pipeline="motor"
            results={analysisResult}
            patientContext={{ age: 65, sex: "male" }}
            className="mb-6"
          />

          {/* Recommendations */}
          {analysisResult.recommendations.length > 0 && (
            <div className="rounded-lg bg-zinc-900 border border-violet-500/30 p-4">
              <h3 className="mb-3 text-sm font-medium text-zinc-200">
                Recommendations
              </h3>
              <ul className="space-y-2">
                {analysisResult.recommendations.map((recommendation, index) => (
                  <li
                    key={index}
                    className="flex items-start space-x-2 text-sm text-zinc-400"
                  >
                    <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-violet-400" />
                    <span>{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-6 flex justify-between">
            <button
              onClick={resetAssessment}
              className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 inline-flex items-center"
            >
              <RotateCcw className="mr-2 h-4 w-4" />
              New Test
            </button>
            <div className="flex space-x-3">
              <button className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800">
                Export Results
              </button>
              <button className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-700">
                Save to History
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
