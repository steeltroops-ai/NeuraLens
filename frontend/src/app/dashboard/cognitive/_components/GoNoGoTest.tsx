"use client";

/**
 * Go/No-Go Test Component
 *
 * Clinical Basis: Measures response inhibition and impulse control.
 * Used in: ADHD assessment, frontal lobe function, executive control research.
 *
 * Task: Press button on "Go" stimuli (green), withhold on "No-Go" (red).
 * Metrics: Commission errors (false alarms), omission errors, reaction time.
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Play, Square, AlertTriangle } from "lucide-react";
import { TaskResult, TaskEvent } from "../types";
import { motion, AnimatePresence } from "framer-motion";

interface GoNoGoTestProps {
  onComplete: (result: TaskResult) => void;
  onCancel: () => void;
  goRatio?: number; // Proportion of Go trials (default 0.75)
}

const TOTAL_TRIALS = 30;
const STIMULUS_DURATION_MS = 500;
const ISI_MIN_MS = 800;
const ISI_MAX_MS = 1500;

type Phase = "instructions" | "ready" | "stimulus" | "feedback" | "complete";
type StimulusType = "go" | "nogo";

export default function GoNoGoTest({
  onComplete,
  onCancel,
  goRatio = 0.75,
}: GoNoGoTestProps) {
  const [phase, setPhase] = useState<Phase>("instructions");
  const [trial, setTrial] = useState(0);
  const [stimulusType, setStimulusType] = useState<StimulusType>("go");
  const [feedback, setFeedback] = useState<string | null>(null);
  const [feedbackType, setFeedbackType] = useState<"success" | "error">(
    "success",
  );

  // Metrics
  const [stats, setStats] = useState({
    hits: 0,
    commissions: 0,
    omissions: 0,
    correctRejections: 0,
  });

  const events = useRef<TaskEvent[]>([]);
  const startTime = useRef<string>(new Date().toISOString());
  const stimulusOnset = useRef<number>(0);
  const responded = useRef<boolean>(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const trialSequence = useRef<StimulusType[]>([]);

  const logEvent = useCallback(
    (type: string, payload: Record<string, unknown> = {}) => {
      events.current.push({
        timestamp: performance.now(),
        event_type: type,
        payload: { trial, ...payload },
      });
    },
    [trial],
  );

  // Generate balanced trial sequence
  useEffect(() => {
    const goCount = Math.round(TOTAL_TRIALS * goRatio);
    const nogoCount = TOTAL_TRIALS - goCount;

    const sequence: StimulusType[] = [
      ...Array(goCount).fill("go"),
      ...Array(nogoCount).fill("nogo"),
    ];

    // Fisher-Yates shuffle
    for (let i = sequence.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [sequence[i]!, sequence[j]!] = [sequence[j]!, sequence[i]!];
    }

    trialSequence.current = sequence;
  }, [goRatio]);

  const startTest = () => {
    logEvent("test_start", { go_ratio: goRatio, total_trials: TOTAL_TRIALS });
    setPhase("ready");
    setTimeout(() => runTrial(0), 1000);
  };

  const runTrial = useCallback(
    (trialNum: number) => {
      if (trialNum >= TOTAL_TRIALS) {
        finishTest();
        return;
      }

      setTrial(trialNum);
      responded.current = false;
      setFeedback(null);

      const currentStimulus = trialSequence.current[trialNum] || "go";
      setStimulusType(currentStimulus);
      setPhase("stimulus");
      stimulusOnset.current = performance.now();

      logEvent("stimulus_shown", { type: currentStimulus });

      // Stimulus timeout - check for omission
      timeoutRef.current = setTimeout(() => {
        if (!responded.current) {
          if (currentStimulus === "go") {
            // Omission error - failed to respond to Go
            setStats((s) => ({ ...s, omissions: s.omissions + 1 }));
            setFeedback("Too slow!");
            setFeedbackType("error");
            logEvent("trial_result", { result: "omission" });
          } else {
            // Correct rejection - correctly withheld response
            setStats((s) => ({
              ...s,
              correctRejections: s.correctRejections + 1,
            }));
            setFeedback("Correct!");
            setFeedbackType("success");
            logEvent("trial_result", { result: "correct_rejection" });
          }

          setPhase("feedback");
          const isi = ISI_MIN_MS + Math.random() * (ISI_MAX_MS - ISI_MIN_MS);
          setTimeout(() => runTrial(trialNum + 1), isi);
        }
      }, STIMULUS_DURATION_MS);
    },
    [logEvent],
  );

  const handleResponse = useCallback(() => {
    if (phase !== "stimulus" || responded.current) return;

    responded.current = true;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    const rt = performance.now() - stimulusOnset.current;
    logEvent("response_received", { rt, stimulus_type: stimulusType });

    if (stimulusType === "go") {
      // Hit - correct response to Go
      setStats((s) => ({ ...s, hits: s.hits + 1 }));
      setFeedback(`${rt.toFixed(0)}ms`);
      setFeedbackType("success");
      logEvent("trial_result", { result: "hit", rt });
    } else {
      // Commission error - responded to No-Go
      setStats((s) => ({ ...s, commissions: s.commissions + 1 }));
      setFeedback("Don't press!");
      setFeedbackType("error");
      logEvent("trial_result", { result: "commission", rt });
    }

    setPhase("feedback");
    const isi = ISI_MIN_MS + Math.random() * (ISI_MAX_MS - ISI_MIN_MS);
    setTimeout(() => runTrial(trial + 1), isi);
  }, [phase, stimulusType, trial, runTrial, logEvent]);

  const finishTest = () => {
    logEvent("test_end", {
      hits: stats.hits,
      commissions: stats.commissions,
      omissions: stats.omissions,
      correct_rejections: stats.correctRejections,
    });

    setPhase("complete");

    const result: TaskResult = {
      task_id: "go_no_go_v1",
      start_time: startTime.current,
      end_time: new Date().toISOString(),
      events: events.current,
      metadata: { go_ratio: goRatio, total_trials: TOTAL_TRIALS },
    };

    setTimeout(() => onComplete(result), 1500);
  };

  // Keyboard handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && phase === "stimulus") {
        e.preventDefault();
        handleResponse();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [phase, handleResponse]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  // ==========================================================================
  // RENDER: INSTRUCTIONS
  // ==========================================================================
  if (phase === "instructions") {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <h2 className="text-2xl font-bold text-zinc-100">Go / No-Go Test</h2>

        <div className="bg-zinc-800/50 p-6 rounded-lg border border-zinc-700 max-w-md text-center space-y-4">
          <p className="text-zinc-300">
            This test measures your ability to <b>inhibit responses</b>.
          </p>

          <div className="grid grid-cols-2 gap-4 py-4">
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-emerald-500 mb-2" />
              <span className="text-emerald-400 font-medium">GO</span>
              <span className="text-xs text-zinc-500">Press Space</span>
            </div>
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-red-500 mb-2" />
              <span className="text-red-400 font-medium">NO-GO</span>
              <span className="text-xs text-zinc-500">Do NOT press</span>
            </div>
          </div>

          <p className="text-sm text-zinc-400">
            Respond as <b>quickly</b> as possible to green, but <b>withhold</b>{" "}
            on red.
          </p>
        </div>

        <div className="flex gap-4">
          <button
            onClick={startTest}
            className="px-6 py-3 bg-emerald-600 rounded-lg text-white font-medium hover:bg-emerald-700 flex items-center gap-2"
          >
            <Play size={18} /> Start Test
          </button>
          <button
            onClick={onCancel}
            className="px-6 py-3 bg-zinc-700 rounded-lg text-zinc-300 hover:bg-zinc-600"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: COMPLETE
  // ==========================================================================
  if (phase === "complete") {
    const accuracy = (stats.hits + stats.correctRejections) / TOTAL_TRIALS;
    const inhibitionRate =
      stats.correctRejections / (stats.correctRejections + stats.commissions) ||
      0;

    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center h-full space-y-6"
      >
        <h2 className="text-2xl font-bold text-emerald-400">Test Complete!</h2>

        <div className="grid grid-cols-2 gap-4 text-center">
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-3xl font-bold text-zinc-100">
              {(accuracy * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-zinc-500">Overall Accuracy</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-3xl font-bold text-zinc-100">
              {(inhibitionRate * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-zinc-500">Inhibition Rate</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-2xl font-bold text-emerald-400">
              {stats.hits}
            </div>
            <div className="text-xs text-zinc-500">Hits</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-2xl font-bold text-red-400">
              {stats.commissions}
            </div>
            <div className="text-xs text-zinc-500">False Alarms</div>
          </div>
        </div>

        <p className="text-zinc-500 text-sm">Saving results...</p>
      </motion.div>
    );
  }

  // ==========================================================================
  // RENDER: ACTIVE TEST
  // ==========================================================================
  return (
    <div className="flex flex-col items-center justify-center h-full">
      {/* Progress */}
      <div className="absolute top-4 right-4 text-sm text-zinc-500 font-mono">
        Trial {trial + 1} / {TOTAL_TRIALS}
      </div>

      {/* Main Stimulus Area */}
      <div
        onClick={handleResponse}
        className="w-48 h-48 rounded-full flex items-center justify-center cursor-pointer transition-all duration-100 select-none"
        style={{
          backgroundColor:
            phase === "stimulus"
              ? stimulusType === "go"
                ? "#10b981"
                : "#ef4444"
              : "#27272a",
          transform: phase === "stimulus" ? "scale(1)" : "scale(0.9)",
          opacity: phase === "ready" ? 0.3 : 1,
        }}
      >
        {phase === "ready" && (
          <span className="text-zinc-500 text-lg">Get ready...</span>
        )}
        {phase === "feedback" && (
          <span
            className={`text-xl font-bold ${feedbackType === "success" ? "text-white" : "text-white"}`}
          >
            {feedback}
          </span>
        )}
      </div>

      {/* Instructions reminder */}
      <div className="mt-8 text-center text-sm text-zinc-500">
        <span className="text-emerald-400">Green</span> = Press Space |
        <span className="text-red-400 ml-1">Red</span> = Do Nothing
      </div>

      {/* Live Stats */}
      <div className="absolute bottom-4 left-4 text-xs text-zinc-600 font-mono">
        H:{stats.hits} CR:{stats.correctRejections} C:{stats.commissions} O:
        {stats.omissions}
      </div>
    </div>
  );
}
