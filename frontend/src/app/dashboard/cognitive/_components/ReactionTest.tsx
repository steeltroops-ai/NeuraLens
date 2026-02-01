"use client";

import React, { useState, useRef, useEffect } from "react";
import { Play, RotateCcw, CheckCircle } from "lucide-react";
import { TaskResult, TaskEvent } from "../types";

interface ReactionTestProps {
  onComplete: (result: TaskResult) => void;
  onCancel: () => void;
}

type TestState =
  | "instructions"
  | "waiting"
  | "stimulus"
  | "feedback"
  | "complete";

export default function ReactionTest({
  onComplete,
  onCancel,
}: ReactionTestProps) {
  const [state, setState] = useState<TestState>("instructions");
  const [trial, setTrial] = useState(0);
  const [reactionTime, setReactionTime] = useState<number | null>(null);
  const [message, setMessage] = useState("");

  // Data logging
  const events = useRef<TaskEvent[]>([]);
  const startTime = useRef<string>(new Date().toISOString());
  const stimTime = useRef<number>(0);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const TOTAL_TRIALS = 5;

  const logEvent = (type: string, payload: any = {}) => {
    events.current.push({
      timestamp: performance.now(),
      event_type: type,
      payload,
    });
  };

  const startTrial = () => {
    setState("waiting");
    setMessage("Wait for green...");
    setReactionTime(null);

    const delay = 2000 + Math.random() * 3000; // 2-5s random delay

    timeoutRef.current = setTimeout(() => {
      setState("stimulus");
      setMessage("CLICK NOW!");
      stimTime.current = performance.now();
      logEvent("stimulus_shown", { trial });
    }, delay);
  };

  const handleClick = () => {
    if (state === "waiting") {
      // False start
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      setMessage("Too early! Wait for green.");
      logEvent("response_early", { trial });
      setState("feedback");
      setTimeout(startTrial, 1500);
      return;
    }

    if (state === "stimulus") {
      const now = performance.now();
      const rt = now - stimTime.current;
      setReactionTime(rt);
      logEvent("response_received", { trial, rt });

      setState("feedback");

      if (trial + 1 >= TOTAL_TRIALS) {
        setTimeout(finishTest, 1000);
      } else {
        setTrial((t) => t + 1);
        setTimeout(startTrial, 1000); // 1s cooldown
      }
    }
  };

  const finishTest = () => {
    setState("complete");
    const result: TaskResult = {
      task_id: "reaction_time",
      start_time: startTime.current,
      end_time: new Date().toISOString(),
      events: events.current,
      metadata: { userAgent: navigator.userAgent },
    };
    onComplete(result);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  if (state === "instructions") {
    return (
      <div className="flex flex-col items-center justify-center space-y-6 h-[400px]">
        <h2 className="text-2xl font-bold text-zinc-100">
          Simple Reaction Time
        </h2>
        <p className="text-zinc-400 max-w-md text-center">
          Wait for the box to turn{" "}
          <span className="text-emerald-400 font-bold">GREEN</span>. Click
          anywhere or press any key as fast as possible.
        </p>
        <div className="flex gap-4">
          <button
            onClick={() => {
              logEvent("test_start");
              startTrial();
            }}
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

  return (
    <div
      className={`
            flex flex-col items-center justify-center h-[500px] w-full rounded-xl cursor-pointer select-none transition-colors duration-200
            ${state === "waiting" ? "bg-zinc-800" : ""}
            ${state === "stimulus" ? "bg-emerald-500" : ""}
            ${state === "feedback" ? "bg-zinc-900" : ""}
        `}
      onMouseDown={handleClick}
      // Also support keyboard
      tabIndex={0}
      onKeyDown={() => handleClick()}
    >
      {state === "waiting" && (
        <div className="text-zinc-500 text-xl font-medium tracking-wider animate-pulse">
          WAIT...
        </div>
      )}

      {state === "stimulus" && (
        <div className="text-white text-4xl font-black tracking-widest scale-110 transformation">
          CLICK!
        </div>
      )}

      {state === "feedback" && reactionTime !== null && (
        <div className="flex flex-col items-center">
          <div className="text-3xl text-zinc-100 font-mono mb-2">
            {reactionTime.toFixed(0)} ms
          </div>
          <div className="text-zinc-500 text-sm">Valid Response</div>
        </div>
      )}

      {state === "feedback" && reactionTime === null && (
        <div className="text-amber-500 text-xl font-bold">{message}</div>
      )}

      <div className="absolute top-4 right-4 text-zinc-500 font-mono text-sm">
        Trial {trial + 1}/{TOTAL_TRIALS}
      </div>
    </div>
  );
}
