"use client";

/**
 * Trail Making Test (TMT) Component
 *
 * Clinical Basis: Measures visual attention, task switching, and processing speed.
 * Used in: Neuropsychological assessment, dementia screening, brain injury evaluation.
 *
 * Part A: Connect numbers in sequence (1-2-3-4...)
 * Part B: Alternate between numbers and letters (1-A-2-B-3-C...)
 *
 * Metrics: Completion time, errors, path efficiency.
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Play, RotateCcw } from "lucide-react";
import { TaskResult, TaskEvent } from "../types";
import { motion } from "framer-motion";

interface TrailMakingTestProps {
  onComplete: (result: TaskResult) => void;
  onCancel: () => void;
  part?: "A" | "B";
}

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  order: number;
}

const PART_A_COUNT = 15;
const PART_B_COUNT = 16; // 8 numbers + 8 letters

type Phase = "instructions" | "running" | "complete";

export default function TrailMakingTest({
  onComplete,
  onCancel,
  part = "A",
}: TrailMakingTestProps) {
  const [phase, setPhase] = useState<Phase>("instructions");
  const [nodes, setNodes] = useState<Node[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [path, setPath] = useState<string[]>([]);
  const [errors, setErrors] = useState(0);
  const [errorFlash, setErrorFlash] = useState<string | null>(null);

  const events = useRef<TaskEvent[]>([]);
  const startTime = useRef<string>(new Date().toISOString());
  const testStartMs = useRef<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);

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

  // Generate nodes with non-overlapping positions
  const generateNodes = useCallback(() => {
    const nodeList: Node[] = [];
    const padding = 60;
    const minDistance = 80;

    const count = part === "A" ? PART_A_COUNT : PART_B_COUNT;

    // Generate labels
    const labels: string[] = [];
    if (part === "A") {
      for (let i = 1; i <= count; i++) {
        labels.push(i.toString());
      }
    } else {
      // Part B: 1, A, 2, B, 3, C, ...
      for (let i = 0; i < count / 2; i++) {
        labels.push((i + 1).toString());
        labels.push(String.fromCharCode(65 + i)); // A, B, C, ...
      }
    }

    // Place nodes with collision detection
    const containerWidth = 600;
    const containerHeight = 400;

    for (let i = 0; i < labels.length; i++) {
      let attempts = 0;
      let x: number, y: number;
      let valid = false;

      while (!valid && attempts < 100) {
        x = padding + Math.random() * (containerWidth - 2 * padding);
        y = padding + Math.random() * (containerHeight - 2 * padding);

        valid = true;
        for (const existing of nodeList) {
          const dx = existing.x - x;
          const dy = existing.y - y;
          if (Math.sqrt(dx * dx + dy * dy) < minDistance) {
            valid = false;
            break;
          }
        }
        attempts++;
      }

      nodeList.push({
        id: `node_${i}`,
        label: labels[i]!,
        x: x!,
        y: y!,
        order: i,
      });
    }

    return nodeList;
  }, [part]);

  useEffect(() => {
    setNodes(generateNodes());
  }, [generateNodes]);

  const startTest = () => {
    setNodes(generateNodes());
    setCurrentIndex(0);
    setPath([]);
    setErrors(0);
    testStartMs.current = performance.now();
    logEvent("test_start", { part, node_count: nodes.length });
    setPhase("running");
  };

  const handleNodeClick = (node: Node) => {
    if (phase !== "running") return;

    const clickTime = performance.now() - testStartMs.current;

    if (node.order === currentIndex) {
      // Correct
      logEvent("node_selected", {
        node_id: node.id,
        label: node.label,
        correct: true,
        time_ms: clickTime,
      });

      setPath((p) => [...p, node.id]);
      setCurrentIndex((i) => i + 1);

      // Check completion
      if (currentIndex + 1 >= nodes.length) {
        finishTest(clickTime);
      }
    } else {
      // Error
      logEvent("node_selected", {
        node_id: node.id,
        label: node.label,
        correct: false,
        expected: nodes[currentIndex]?.label,
        time_ms: clickTime,
      });

      setErrors((e) => e + 1);
      setErrorFlash(node.id);
      setTimeout(() => setErrorFlash(null), 300);
    }
  };

  const finishTest = (completionTime: number) => {
    logEvent("test_end", {
      completion_time_ms: completionTime,
      errors,
      part,
    });

    setPhase("complete");

    const result: TaskResult = {
      task_id: `trail_making_${part.toLowerCase()}`,
      start_time: startTime.current,
      end_time: new Date().toISOString(),
      events: events.current,
      metadata: { part, node_count: nodes.length },
    };

    setTimeout(() => onComplete(result), 2000);
  };

  const getExpectedLabel = () => {
    if (currentIndex >= nodes.length) return "";
    return nodes.find((n) => n.order === currentIndex)?.label || "";
  };

  // ==========================================================================
  // RENDER: INSTRUCTIONS
  // ==========================================================================
  if (phase === "instructions") {
    return (
      <div className="flex flex-col items-center justify-center h-full space-y-6">
        <h2 className="text-2xl font-bold text-zinc-100">
          Trail Making Test - Part {part}
        </h2>

        <div className="bg-zinc-800/50 p-6 rounded-lg border border-zinc-700 max-w-lg text-center space-y-4">
          {part === "A" ? (
            <>
              <p className="text-zinc-300">
                Connect the <b>numbers</b> in ascending order as quickly as
                possible.
              </p>
              <div className="flex justify-center items-center gap-2 py-4 text-lg font-mono">
                <span className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                  1
                </span>
                <span className="text-zinc-500">→</span>
                <span className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                  2
                </span>
                <span className="text-zinc-500">→</span>
                <span className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                  3
                </span>
                <span className="text-zinc-500">→</span>
                <span className="text-zinc-400">...</span>
              </div>
            </>
          ) : (
            <>
              <p className="text-zinc-300">
                Alternate between <b>numbers</b> and <b>letters</b> in ascending
                order.
              </p>
              <div className="flex justify-center items-center gap-2 py-4 text-lg font-mono">
                <span className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                  1
                </span>
                <span className="text-zinc-500">→</span>
                <span className="w-8 h-8 rounded-full bg-amber-500 flex items-center justify-center text-white">
                  A
                </span>
                <span className="text-zinc-500">→</span>
                <span className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                  2
                </span>
                <span className="text-zinc-500">→</span>
                <span className="w-8 h-8 rounded-full bg-amber-500 flex items-center justify-center text-white">
                  B
                </span>
              </div>
            </>
          )}

          <p className="text-sm text-zinc-400">
            Click each node in the correct order. Work as fast as you can!
          </p>
        </div>

        <div className="flex gap-4">
          <button
            onClick={startTest}
            className="px-6 py-3 bg-blue-600 rounded-lg text-white font-medium hover:bg-blue-700 flex items-center gap-2"
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
    const completionTime =
      (events.current.find((e) => e.event_type === "test_end")?.payload
        ?.completion_time_ms as number) || 0;

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
              {(completionTime / 1000).toFixed(1)}s
            </div>
            <div className="text-xs text-zinc-500">Completion Time</div>
          </div>
          <div className="bg-zinc-800 p-4 rounded-lg">
            <div className="text-3xl font-bold text-zinc-100">{errors}</div>
            <div className="text-xs text-zinc-500">Errors</div>
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
      {/* Header */}
      <div className="absolute top-4 left-4 right-4 flex justify-between items-center">
        <div className="text-sm text-zinc-400">
          Part {part} | Next:{" "}
          <span className="font-bold text-blue-400">{getExpectedLabel()}</span>
        </div>
        <div className="text-sm text-zinc-500 font-mono">
          Progress: {currentIndex}/{nodes.length} | Errors: {errors}
        </div>
      </div>

      {/* Trail Area */}
      <div
        ref={containerRef}
        className="relative w-[600px] h-[400px] bg-zinc-900 rounded-lg border border-zinc-700"
      >
        {/* Connection lines */}
        <svg className="absolute inset-0 pointer-events-none">
          {path.map((nodeId, i) => {
            if (i === 0) return null;
            const prevNode = nodes.find((n) => n.id === path[i - 1]);
            const currNode = nodes.find((n) => n.id === nodeId);
            if (!prevNode || !currNode) return null;

            return (
              <line
                key={`line_${i}`}
                x1={prevNode.x}
                y1={prevNode.y}
                x2={currNode.x}
                y2={currNode.y}
                stroke="#3b82f6"
                strokeWidth="2"
                strokeLinecap="round"
              />
            );
          })}
        </svg>

        {/* Nodes */}
        {nodes.map((node) => {
          const isCompleted = path.includes(node.id);
          const isCurrent = node.order === currentIndex;
          const isError = errorFlash === node.id;
          const isLetter = isNaN(parseInt(node.label));

          return (
            <motion.div
              key={node.id}
              onClick={() => handleNodeClick(node)}
              className={`absolute w-10 h-10 rounded-full flex items-center justify-center cursor-pointer font-bold text-sm select-none transition-all ${
                isCompleted
                  ? "bg-emerald-500 text-white"
                  : isError
                    ? "bg-red-500 text-white scale-110"
                    : isCurrent
                      ? "bg-blue-500 text-white ring-4 ring-blue-500/30"
                      : isLetter
                        ? "bg-amber-600 text-white hover:bg-amber-500"
                        : "bg-blue-600 text-white hover:bg-blue-500"
              }`}
              style={{
                left: node.x - 20,
                top: node.y - 20,
              }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              {node.label}
            </motion.div>
          );
        })}
      </div>

      {/* Instructions */}
      <div className="mt-4 text-sm text-zinc-500">
        Click the nodes in order:{" "}
        {part === "A" ? "1 → 2 → 3 → ..." : "1 → A → 2 → B → ..."}
      </div>
    </div>
  );
}
