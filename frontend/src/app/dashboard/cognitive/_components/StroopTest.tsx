"use client";

/**
 * Stroop Test Component
 *
 * Clinical Basis: Measures selective attention and cognitive flexibility.
 * Used in: ADHD assessment, executive dysfunction, frontal lobe evaluation.
 *
 * Task: Name the COLOR of the word, NOT what the word says.
 * Congruent: "RED" in red → Easy
 * Incongruent: "RED" in blue → Requires inhibition
 *
 * Metrics: Stroop effect (RT difference), accuracy, interference score.
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Play } from "lucide-react";
import { TaskResult, TaskEvent } from "../types";
import { motion } from "framer-motion";

interface StroopTestProps {
  onComplete: (result: TaskResult) => void;
  onCancel: () => void;
}

const COLORS = [
  { name: "RED", hex: "#ef4444", key: "r" },
  { name: "GREEN", hex: "#22c55e", key: "g" },
  { name: "BLUE", hex: "#3b82f6", key: "b" },
  { name: "YELLOW", hex: "#eab308", key: "y" },
];

const TOTAL_TRIALS = 40;
const CONGRUENT_RATIO = 0.3; // 30% congruent

interface Trial {
  word: string;
  inkColor: (typeof COLORS)[0];
  isCongruent: boolean;
}

type Phase = "instructions" | "ready" | "stimulus" | "feedback" | "complete";

export default function StroopTest({ onComplete, onCancel }: StroopTestProps) {
  const [phase, setPhase] = useState<Phase>("instructions");
  const [trialIndex, setTrialIndex] = useState(0);
  const [currentTrial, setCurrentTrial] = useState<Trial | null>(null);
  const [feedback, setFeedback] = useState<{
    correct: boolean;
    rt: number;
  } | null>(null);

  // Stats
  const [congruentRTs, setCongruentRTs] = useState<number[]>([]);
  const [incongruentRTs, setIncongruentRTs] = useState<number[]>([]);
  const [errors, setErrors] = useState(0);

  const events = useRef<TaskEvent[]>([]);
  const startTime = useRef<string>(new Date().toISOString());
  const stimulusOnset = useRef<number>(0);
  const trials = useRef<Trial[]>([]);

  const logEvent = useCallback(
    (type: string, payload: Record<string, unknown> = {}) => {
      events.current.push({
        timestamp: performance.now(),
        event_type: type,
        payload: { trial: trialIndex, ...payload },
      });
    },
    [trialIndex],
  );

  // Generate trial sequence
  useEffect(() => {
    const generated: Trial[] = [];
    const congruentCount = Math.round(TOTAL_TRIALS * CONGRUENT_RATIO);

    for (let i = 0; i < TOTAL_TRIALS; i++) {
      const isCongruent = i < congruentCount;
      const wordColorIndex = Math.floor(Math.random() * COLORS.length);
      const wordColor = COLORS[wordColorIndex]!;

      let inkColor: (typeof COLORS)[0];
      if (isCongruent) {
        inkColor = wordColor;
      } else {
        // Pick a different color for ink
        const others = COLORS.filter((c) => c.name !== wordColor.name);
        const otherIndex = Math.floor(Math.random() * others.length);
        inkColor = others[otherIndex]!;
      }

      generated.push({
        word: wordColor.name,
        inkColor,
        isCongruent,
      });
    }

    // Shuffle
    for (let i = generated.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const temp = generated[i]!;
      generated[i] = generated[j]!;
      generated[j] = temp;
    }

    trials.current = generated;
  }, []);

  const startTest = () => {
    logEvent("test_start", {
      total_trials: TOTAL_TRIALS,
      congruent_ratio: CONGRUENT_RATIO,
    });
    setPhase("ready");
    setTimeout(() => showTrial(0), 1000);
  };

  const showTrial = (index: number) => {
    if (index >= TOTAL_TRIALS) {
      finishTest();
      return;
    }

    setTrialIndex(index);
    const trial = trials.current[index];
    if (!trial) return;

    setCurrentTrial(trial);
    setFeedback(null);
    setPhase("stimulus");
    stimulusOnset.current = performance.now();

    logEvent("stimulus_shown", {
      word: trial.word,
      ink_color: trial.inkColor.name,
      congruent: trial.isCongruent,
    });
  };

  const handleResponse = useCallback(
    (colorKey: string) => {
      if (phase !== "stimulus" || !currentTrial) return;

      const rt = performance.now() - stimulusOnset.current;
      const selectedColor = COLORS.find((c) => c.key === colorKey);

      if (!selectedColor) return;

      const correct = selectedColor.name === currentTrial.inkColor.name;

      logEvent("response", {
        response: selectedColor.name,
        correct_answer: currentTrial.inkColor.name,
        correct,
        rt,
        congruent: currentTrial.isCongruent,
      });

      if (correct) {
        if (currentTrial.isCongruent) {
          setCongruentRTs((rts) => [...rts, rt]);
        } else {
          setIncongruentRTs((rts) => [...rts, rt]);
        }
      } else {
        setErrors((e) => e + 1);
      }

      setFeedback({ correct, rt });
      setPhase("feedback");

      setTimeout(() => showTrial(trialIndex + 1), 500);
    },
    [phase, currentTrial, trialIndex, logEvent],
  );

  const finishTest = () => {
    const avgCongruent =
      congruentRTs.length > 0
        ? congruentRTs.reduce((a, b) => a + b, 0) / congruentRTs.length
        : 0;
    const avgIncongruent =
      incongruentRTs.length > 0
        ? incongruentRTs.reduce((a, b) => a + b, 0) / incongruentRTs.length
        : 0;
    const stroopEffect = avgIncongruent - avgCongruent;

    logEvent("test_end", {
      avg_congruent_rt: avgCongruent,
      avg_incongruent_rt: avgIncongruent,
      stroop_effect_ms: stroopEffect,
      errors,
    });

    setPhase("complete");

    const result: TaskResult = {
      task_id: "stroop_v1",
      start_time: startTime.current,
      end_time: new Date().toISOString(),
      events: events.current,
      metadata: { total_trials: TOTAL_TRIALS },
    };

    setTimeout(() => onComplete(result), 2000);
  };

  // Keyboard handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (phase !== "stimulus") return;

      const key = e.key.toLowerCase();
      if (["r", "g", "b", "y"].includes(key)) {
        handleResponse(key);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [phase, handleResponse]);

  // ==========================================================================
  // RENDER: INSTRUCTIONS
  // ==========================================================================
  if (phase === "instructions") {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <h2 className="text-2xl font-bold text-zinc-100">Stroop Test</h2>

        <div className="bg-zinc-800/50 p-6 rounded-lg border border-zinc-700 max-w-lg text-center space-y-4">
          <p className="text-zinc-300">
            Name the <b>COLOR of the ink</b>, NOT what the word says!
          </p>

          <div className="py-4 space-y-3">
            <div className="flex items-center justify-center gap-4">
              <span className="text-2xl font-bold" style={{ color: "#ef4444" }}>
                RED
              </span>
              <span className="text-zinc-500">→ Press</span>
              <span className="px-3 py-1 bg-red-500 text-white rounded font-bold">
                R
              </span>
            </div>
            <div className="flex items-center justify-center gap-4">
              <span className="text-2xl font-bold" style={{ color: "#3b82f6" }}>
                GREEN
              </span>
              <span className="text-zinc-500">→ Press</span>
              <span className="px-3 py-1 bg-blue-500 text-white rounded font-bold">
                B
              </span>
              <span className="text-xs text-zinc-400">(ink is Blue!)</span>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2 pt-4 border-t border-zinc-700">
            {COLORS.map((color) => (
              <div key={color.key} className="flex flex-col items-center">
                <div
                  className="w-8 h-8 rounded-full mb-1"
                  style={{ backgroundColor: color.hex }}
                />
                <span className="text-xs text-zinc-400 uppercase">
                  {color.key}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex gap-4">
          <button
            onClick={startTest}
            className="px-6 py-3 bg-rose-600 rounded-lg text-white font-medium hover:bg-rose-700 flex items-center gap-2"
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
    const avgCongruent =
      congruentRTs.length > 0
        ? congruentRTs.reduce((a, b) => a + b, 0) / congruentRTs.length
        : 0;
    const avgIncongruent =
      incongruentRTs.length > 0
        ? incongruentRTs.reduce((a, b) => a + b, 0) / incongruentRTs.length
        : 0;
    const stroopEffect = avgIncongruent - avgCongruent;
    const accuracy = ((TOTAL_TRIALS - errors) / TOTAL_TRIALS) * 100;

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
              {stroopEffect.toFixed(0)}ms
            </div>
            <div className="text-xs text-zinc-500">Stroop Effect</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-3xl font-bold text-emerald-400">
              {accuracy.toFixed(0)}%
            </div>
            <div className="text-xs text-zinc-500">Accuracy</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-2xl font-bold text-zinc-300">
              {avgCongruent.toFixed(0)}ms
            </div>
            <div className="text-xs text-zinc-500">Congruent RT</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-2xl font-bold text-zinc-300">
              {avgIncongruent.toFixed(0)}ms
            </div>
            <div className="text-xs text-zinc-500">Incongruent RT</div>
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
        Trial {trialIndex + 1} / {TOTAL_TRIALS}
      </div>

      {/* Stimulus */}
      <div className="text-center mb-8">
        {phase === "ready" && (
          <div className="text-2xl text-zinc-500">Get ready...</div>
        )}
        {(phase === "stimulus" || phase === "feedback") && currentTrial && (
          <motion.div
            key={trialIndex}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-7xl font-black select-none"
            style={{ color: currentTrial.inkColor.hex }}
          >
            {currentTrial.word}
          </motion.div>
        )}
        {feedback && (
          <div
            className={`mt-4 text-lg ${feedback.correct ? "text-emerald-400" : "text-red-400"}`}
          >
            {feedback.correct ? `${feedback.rt.toFixed(0)}ms` : "Wrong!"}
          </div>
        )}
      </div>

      {/* Response Buttons */}
      <div className="grid grid-cols-4 gap-4">
        {COLORS.map((color) => (
          <button
            key={color.key}
            onClick={() => handleResponse(color.key)}
            className="w-16 h-16 rounded-xl font-bold text-white text-xl transition-transform hover:scale-105 active:scale-95"
            style={{ backgroundColor: color.hex }}
          >
            {color.key.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Keyboard hint */}
      <div className="mt-6 text-sm text-zinc-500">Press R, G, B, or Y</div>

      {/* Live Stats */}
      <div className="absolute bottom-4 left-4 text-xs text-zinc-600 font-mono">
        Errors: {errors}
      </div>
    </div>
  );
}
