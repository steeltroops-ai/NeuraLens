"use client";

/**
 * Digit Symbol Substitution Test (DSST) / Symbol Digit Modalities Test
 *
 * Clinical Basis: Measures processing speed, visual scanning, and sustained attention.
 * Used in: Cognitive aging research, dementia screening, medication effects assessment.
 *
 * Task: Match symbols to digits using a reference key as quickly as possible.
 * Metrics: Items completed, accuracy, response times.
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Play, ArrowRight } from "lucide-react";
import { TaskResult, TaskEvent } from "../types";
import { motion, AnimatePresence } from "framer-motion";

interface DigitSymbolTestProps {
  onComplete: (result: TaskResult) => void;
  onCancel: () => void;
  duration?: number; // seconds
}

// Symbol set (using Unicode symbols for visual distinctness)
const SYMBOLS = ["◆", "★", "●", "▲", "■", "◈", "✦", "○", "△"];

const TEST_DURATION_MS = 90000; // 90 seconds
const ITEMS_COUNT = 50;

type Phase = "instructions" | "running" | "complete";

interface TrialItem {
  symbol: string;
  correctDigit: number;
  position: number;
}

export default function DigitSymbolTest({
  onComplete,
  onCancel,
  duration = 90,
}: DigitSymbolTestProps) {
  const [phase, setPhase] = useState<Phase>("instructions");
  const [items, setItems] = useState<TrialItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(duration);
  const [responses, setResponses] = useState<
    { correct: boolean; rt: number }[]
  >([]);
  const [lastFeedback, setLastFeedback] = useState<"correct" | "wrong" | null>(
    null,
  );

  const events = useRef<TaskEvent[]>([]);
  const startTime = useRef<string>(new Date().toISOString());
  const itemOnsetTime = useRef<number>(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  const logEvent = useCallback(
    (type: string, payload: Record<string, unknown> = {}) => {
      events.current.push({
        timestamp: performance.now(),
        event_type: type,
        payload,
      });
    },
    [],
  );

  // Generate randomized items
  useEffect(() => {
    const generated: TrialItem[] = [];
    for (let i = 0; i < ITEMS_COUNT; i++) {
      const digitIndex = Math.floor(Math.random() * 9);
      generated.push({
        symbol: SYMBOLS[digitIndex]!,
        correctDigit: digitIndex + 1,
        position: i,
      });
    }
    setItems(generated);
  }, []);

  const startTest = () => {
    logEvent("test_start", {
      duration_seconds: duration,
      items_count: ITEMS_COUNT,
    });
    setPhase("running");
    setTimeRemaining(duration);
    itemOnsetTime.current = performance.now();

    // Countdown timer
    countdownRef.current = setInterval(() => {
      setTimeRemaining((t) => {
        if (t <= 1) {
          finishTest();
          return 0;
        }
        return t - 1;
      });
    }, 1000);
  };

  const handleDigitPress = useCallback(
    (digit: number) => {
      if (phase !== "running" || currentIndex >= items.length) return;

      const currentItem = items[currentIndex];
      if (!currentItem) return;

      const rt = performance.now() - itemOnsetTime.current;
      const correct = digit === currentItem.correctDigit;

      logEvent("response", {
        item_index: currentIndex,
        symbol: currentItem.symbol,
        response: digit,
        correct_answer: currentItem.correctDigit,
        correct,
        rt,
      });

      setResponses((r) => [...r, { correct, rt }]);
      setLastFeedback(correct ? "correct" : "wrong");

      // Move to next item
      if (currentIndex + 1 >= items.length) {
        finishTest();
      } else {
        setCurrentIndex((i) => i + 1);
        itemOnsetTime.current = performance.now();
        setTimeout(() => setLastFeedback(null), 200);
      }
    },
    [phase, currentIndex, items, logEvent],
  );

  const finishTest = useCallback(() => {
    if (countdownRef.current) clearInterval(countdownRef.current);

    const correctCount =
      responses.filter((r) => r.correct).length +
      (lastFeedback === "correct" ? 1 : 0);
    const avgRt =
      responses.length > 0
        ? responses.reduce((sum, r) => sum + r.rt, 0) / responses.length
        : 0;

    logEvent("test_end", {
      items_completed: currentIndex + 1,
      correct_count: correctCount,
      avg_rt_ms: avgRt,
    });

    setPhase("complete");

    const result: TaskResult = {
      task_id: "digit_symbol_v1",
      start_time: startTime.current,
      end_time: new Date().toISOString(),
      events: events.current,
      metadata: { duration_seconds: duration },
    };

    setTimeout(() => onComplete(result), 2000);
  }, [responses, currentIndex, lastFeedback, duration, onComplete, logEvent]);

  // Keyboard handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (phase !== "running") return;

      const digit = parseInt(e.key);
      if (digit >= 1 && digit <= 9) {
        handleDigitPress(digit);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [phase, handleDigitPress]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, []);

  const currentItem = items[currentIndex];
  const correctCount = responses.filter((r) => r.correct).length;

  // ==========================================================================
  // RENDER: INSTRUCTIONS
  // ==========================================================================
  if (phase === "instructions") {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <h2 className="text-2xl font-bold text-zinc-100">Digit Symbol Test</h2>

        <div className="bg-zinc-800/50 p-6 rounded-lg border border-zinc-700 max-w-lg text-center space-y-4">
          <p className="text-zinc-300">
            Match each <b>symbol</b> to its corresponding <b>number</b> using
            the key below.
          </p>

          {/* Reference Key */}
          <div className="bg-zinc-900 p-4 rounded-lg border border-zinc-600">
            <div className="grid grid-cols-9 gap-2 text-center">
              {SYMBOLS.map((symbol, i) => (
                <div key={i} className="flex flex-col items-center">
                  <span className="text-2xl mb-1">{symbol}</span>
                  <span className="text-sm font-bold text-blue-400">
                    {i + 1}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <p className="text-sm text-zinc-400">
            Press the number key (1-9) that matches the symbol shown.
            <br />
            Work as <b>quickly and accurately</b> as possible!
          </p>

          <p className="text-xs text-zinc-500">
            Time limit: {duration} seconds
          </p>
        </div>

        <div className="flex gap-4">
          <button
            onClick={startTest}
            className="px-6 py-3 bg-purple-600 rounded-lg text-white font-medium hover:bg-purple-700 flex items-center gap-2"
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
    const accuracy =
      responses.length > 0 ? (correctCount / responses.length) * 100 : 0;
    const avgRt =
      responses.length > 0
        ? responses.reduce((sum, r) => sum + r.rt, 0) / responses.length
        : 0;

    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center h-full space-y-6"
      >
        <h2 className="text-2xl font-bold text-emerald-400">Test Complete!</h2>

        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-3xl font-bold text-zinc-100">
              {currentIndex + 1}
            </div>
            <div className="text-xs text-zinc-500">Items Completed</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-3xl font-bold text-emerald-400">
              {accuracy.toFixed(0)}%
            </div>
            <div className="text-xs text-zinc-500">Accuracy</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-3xl font-bold text-zinc-100">
              {(avgRt / 1000).toFixed(2)}s
            </div>
            <div className="text-xs text-zinc-500">Avg Response</div>
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
    <div className="flex flex-col items-center h-full pt-4">
      {/* Timer and Progress */}
      <div className="absolute top-4 left-4 right-4 flex justify-between items-center">
        <div
          className={`text-lg font-mono font-bold ${timeRemaining <= 10 ? "text-red-400" : "text-zinc-300"}`}
        >
          {Math.floor(timeRemaining / 60)}:
          {(timeRemaining % 60).toString().padStart(2, "0")}
        </div>
        <div className="text-sm text-zinc-500">
          Item {currentIndex + 1} | Correct: {correctCount}
        </div>
      </div>

      {/* Reference Key (always visible) */}
      <div className="bg-zinc-900 p-3 rounded-lg border border-zinc-700 mb-6 mt-8">
        <div className="grid grid-cols-9 gap-3 text-center">
          {SYMBOLS.map((symbol, i) => (
            <div key={i} className="flex flex-col items-center px-2">
              <span className="text-xl">{symbol}</span>
              <span className="text-xs font-bold text-blue-400">{i + 1}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Current Symbol */}
      <div
        className={`w-32 h-32 rounded-xl flex items-center justify-center text-6xl transition-all ${
          lastFeedback === "correct"
            ? "bg-emerald-500/20 border-emerald-500"
            : lastFeedback === "wrong"
              ? "bg-red-500/20 border-red-500"
              : "bg-zinc-800 border-zinc-600"
        } border-2`}
      >
        {currentItem?.symbol}
      </div>

      {/* Number Buttons */}
      <div className="grid grid-cols-9 gap-2 mt-8">
        {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => (
          <button
            key={digit}
            onClick={() => handleDigitPress(digit)}
            className="w-12 h-12 rounded-lg bg-zinc-700 text-zinc-100 font-bold text-lg hover:bg-zinc-600 active:bg-zinc-500 transition-colors"
          >
            {digit}
          </button>
        ))}
      </div>

      {/* Instructions */}
      <div className="mt-6 text-sm text-zinc-500">
        Press 1-9 on keyboard or click buttons
      </div>
    </div>
  );
}
