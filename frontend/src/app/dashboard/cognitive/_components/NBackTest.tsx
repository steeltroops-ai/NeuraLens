"use client";

import React, { useState, useRef, useEffect } from "react";
import { Play } from "lucide-react";
import { TaskResult, TaskEvent } from "../types";

interface NBackTestProps {
  onComplete: (result: TaskResult) => void;
  onCancel: () => void;
  n?: number; // default 2
}

export default function NBackTest({
  onComplete,
  onCancel,
  n = 2,
}: NBackTestProps) {
  // 1-Back for simplicity in this demo, but logic supports N
  const [testPhase, setTestPhase] = useState<"instruct" | "running" | "end">(
    "instruct",
  );
  const [currentLetter, setCurrentLetter] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [score, setScore] = useState({ hits: 0, fps: 0, misses: 0 }); // Debug feedback
  const [feedback, setFeedback] = useState<string | null>(null); // "Correct!", "Missed!"

  const events = useRef<TaskEvent[]>([]);
  const startTime = useRef<string>(new Date().toISOString());

  const LETTERS = ["A", "B", "C", "D", "E", "H", "K", "L", "M"];
  const TOTAL_STEPS = 20; // Short sequence for demo
  const INTERVAL_MS = 2500;

  const stepRef = useRef(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const respondedRef = useRef(false);

  const logEvent = (type: string, payload: any = {}) => {
    events.current.push({
      timestamp: performance.now(),
      event_type: type,
      payload,
    });
  };

  const startTest = () => {
    setTestPhase("running");
    logEvent("test_start", { n_back: n });
    nextStep();
  };

  const nextStep = () => {
    if (stepRef.current >= TOTAL_STEPS) {
      finishTest();
      return;
    }

    respondedRef.current = false;
    setFeedback(null);

    // Generate letter (weighted to ensure some matches occur)
    const randomIndex = Math.floor(Math.random() * LETTERS.length);
    let letter = LETTERS[randomIndex] || "A";

    // Force match approx 30% of time
    if (history.length >= n && Math.random() < 0.3) {
      const matchLetter = history[history.length - n];
      if (matchLetter) {
        letter = matchLetter;
      }
    }

    setCurrentLetter(letter);
    setHistory((prev) => [...prev, letter]);
    logEvent("stimulus_shown", { letter, step: stepRef.current });

    stepRef.current += 1;

    timerRef.current = setTimeout(nextStep, INTERVAL_MS);
  };

  const handleMatchClaim = () => {
    if (respondedRef.current) return;
    respondedRef.current = true;

    const isMatch =
      history.length > n && history[history.length - 1 - n] === currentLetter;

    logEvent("user_response", {
      claimed_match: true,
      is_actual_match: isMatch,
    });

    if (isMatch) {
      setFeedback("Hit!");
      setScore((s) => ({ ...s, hits: s.hits + 1 }));
      logEvent("trial_result", { result: "hit" });
    } else {
      setFeedback("False Alarm");
      setScore((s) => ({ ...s, fps: s.fps + 1 }));
      logEvent("trial_result", { result: "false_alarm" });
    }
  };

  const finishTest = () => {
    // Logic to log misses (omission errors) requires post-processing or tracking un-responded matches
    setTestPhase("end");
    const result: TaskResult = {
      task_id: `n_back_${n}`,
      start_time: startTime.current,
      end_time: new Date().toISOString(),
      events: events.current,
      metadata: { n },
    };
    onComplete(result);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  if (testPhase === "instruct") {
    return (
      <div className="flex flex-col items-center justify-center space-y-6 h-[400px]">
        <h2 className="text-2xl font-bold text-zinc-100">
          {n}-Back Memory Test
        </h2>
        <div className="bg-zinc-800/50 p-6 rounded-lg text-center border border-zinc-700">
          <p className="text-zinc-300 mb-4">
            Memorize the sequence of letters.
            <br />
            Press <span className="text-amber-400 font-bold">MATCH</span>{" "}
            whenever the current letter is the same as the one shown{" "}
            <b>{n} steps ago</b>.
          </p>
          <div className="flex justify-center gap-2 font-mono text-sm text-zinc-500">
            <span>A</span> → <span>B</span> →{" "}
            <span className="text-green-500 font-bold underline">A</span> (Match
            for 2-back)
          </div>
        </div>
        <div className="flex gap-4">
          <button
            onClick={startTest}
            className="px-6 py-3 bg-amber-600 rounded-lg text-white hover:bg-amber-700 flex items-center gap-2"
          >
            <Play size={18} /> Start
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

  return (
    <div className="flex flex-col items-center justify-center h-[500px]">
      {/* Stimulus Card */}
      <div className="w-64 h-64 bg-zinc-800 rounded-xl flex items-center justify-center border border-zinc-700 shadow-xl mb-8 relative">
        <span className="text-9xl font-bold text-zinc-200">
          {currentLetter}
        </span>
        {feedback && (
          <div
            className={`absolute -top-12 text-lg font-bold ${feedback === "Hit!" ? "text-emerald-400" : "text-red-400"}`}
          >
            {feedback}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-col items-center space-y-4">
        <button
          onMouseDown={handleMatchClaim}
          className="w-64 py-6 bg-zinc-700 rounded-xl border-b-4 border-zinc-900 active:border-b-0 active:translate-y-1 hover:bg-zinc-600 text-xl font-semibold text-zinc-200 transition-all"
        >
          MATCH (Space)
        </button>

        <div className="text-zinc-500 text-sm font-mono mt-4">
          Trial {stepRef.current}/{TOTAL_STEPS}
        </div>
      </div>
    </div>
  );
}
