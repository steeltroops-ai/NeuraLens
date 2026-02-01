"use client";

/**
 * Cognitive Assessment Component - Production Grade
 *
 * Complete test battery with research-grade cognitive assessments:
 * - Reaction Time (PVT): Processing speed, sustained attention
 * - N-Back: Working memory capacity
 * - Go/No-Go: Response inhibition, impulse control
 * - Trail Making (A & B): Visual attention, task switching
 * - Digit Symbol: Processing speed, visual scanning
 * - Stroop: Selective attention, cognitive flexibility
 */

import React from "react";
import {
  Brain,
  Activity,
  Clock,
  TrendingUp,
  CheckCircle,
  AlertTriangle,
  Loader2,
  RefreshCw,
  XCircle,
  Zap,
  Target,
  GitBranch,
  Sparkles,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Test Components
import ReactionTest from "./ReactionTest";
import NBackTest from "./NBackTest";
import GoNoGoTest from "./GoNoGoTest";
import TrailMakingTest from "./TrailMakingTest";
import DigitSymbolTest from "./DigitSymbolTest";
import StroopTest from "./StroopTest";
import ResultsPanel from "./ResultsPanel";
import { useCognitiveSession } from "./useCognitiveSession";
import { TaskResult } from "../types";

interface CognitiveAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

type TestType =
  | "none"
  | "reaction"
  | "memory"
  | "go_no_go"
  | "trail_a"
  | "trail_b"
  | "digit_symbol"
  | "stroop";

interface TestInfo {
  id: Exclude<TestType, "none">;
  name: string;
  description: string;
  duration: string;
  domain: string;
  icon: React.ReactNode;
  color: string;
  available: boolean;
}

const TEST_BATTERY: TestInfo[] = [
  {
    id: "reaction",
    name: "Reaction Time",
    description:
      "Psychomotor vigilance task measuring processing speed and sustained attention.",
    duration: "~1 min",
    domain: "Processing Speed",
    icon: <Zap size={20} />,
    color: "blue",
    available: true,
  },
  {
    id: "memory",
    name: "N-Back Memory",
    description:
      "2-Back working memory task assessing short-term memory updating.",
    duration: "~2 min",
    domain: "Working Memory",
    icon: <Brain size={20} />,
    color: "amber",
    available: true,
  },
  {
    id: "go_no_go",
    name: "Go / No-Go",
    description:
      "Response inhibition task measuring impulse control and executive function.",
    duration: "~2 min",
    domain: "Inhibition",
    icon: <Target size={20} />,
    color: "emerald",
    available: true,
  },
  {
    id: "stroop",
    name: "Stroop Test",
    description:
      "Color-word interference task assessing selective attention and cognitive flexibility.",
    duration: "~2 min",
    domain: "Attention",
    icon: <Sparkles size={20} />,
    color: "rose",
    available: true,
  },
  {
    id: "trail_a",
    name: "Trail Making A",
    description:
      "Connect numbers in sequence to assess visual attention and processing speed.",
    duration: "~1 min",
    domain: "Visual Attention",
    icon: <GitBranch size={20} />,
    color: "cyan",
    available: true,
  },
  {
    id: "trail_b",
    name: "Trail Making B",
    description:
      "Alternate between numbers and letters to assess task switching ability.",
    duration: "~2 min",
    domain: "Executive Function",
    icon: <TrendingUp size={20} />,
    color: "purple",
    available: true,
  },
  {
    id: "digit_symbol",
    name: "Digit Symbol",
    description:
      "Symbol-digit matching under time pressure to assess processing speed.",
    duration: "~90 sec",
    domain: "Processing Speed",
    icon: <Clock size={20} />,
    color: "indigo",
    available: true,
  },
];

const colorClasses: Record<
  string,
  { bg: string; hover: string; icon: string; border: string }
> = {
  blue: {
    bg: "bg-blue-500/15",
    hover: "hover:border-blue-500/50",
    icon: "text-blue-400",
    border: "border-blue-500/30",
  },
  amber: {
    bg: "bg-amber-500/15",
    hover: "hover:border-amber-500/50",
    icon: "text-amber-400",
    border: "border-amber-500/30",
  },
  emerald: {
    bg: "bg-emerald-500/15",
    hover: "hover:border-emerald-500/50",
    icon: "text-emerald-400",
    border: "border-emerald-500/30",
  },
  rose: {
    bg: "bg-rose-500/15",
    hover: "hover:border-rose-500/50",
    icon: "text-rose-400",
    border: "border-rose-500/30",
  },
  cyan: {
    bg: "bg-cyan-500/15",
    hover: "hover:border-cyan-500/50",
    icon: "text-cyan-400",
    border: "border-cyan-500/30",
  },
  purple: {
    bg: "bg-purple-500/15",
    hover: "hover:border-purple-500/50",
    icon: "text-purple-400",
    border: "border-purple-500/30",
  },
  indigo: {
    bg: "bg-indigo-500/15",
    hover: "hover:border-indigo-500/50",
    icon: "text-indigo-400",
    border: "border-indigo-500/30",
  },
};

export default function CognitiveAssessment({
  onProcessingChange,
}: CognitiveAssessmentProps) {
  const { state, actions } = useCognitiveSession();
  const [activeTest, setActiveTest] = React.useState<TestType>("none");

  // Sync processing state with parent
  React.useEffect(() => {
    onProcessingChange(state.state === "submitting");
  }, [state.state, onProcessingChange]);

  const handleTestComplete = (result: TaskResult) => {
    actions.completeTest(result);
    setActiveTest("none");
  };

  const handleSubmit = async () => {
    await actions.submitSession();
  };

  const handleStartTest = (type: TestType) => {
    actions.startTest();
    setActiveTest(type);
  };

  const getCompletedTestIds = (): Exclude<TestType, "none">[] => {
    return state.tasks
      .map((t) => {
        // Match standardized task IDs (without version suffix)
        if (t.task_id === "reaction_time" || t.task_id.startsWith("reaction"))
          return "reaction" as const;
        if (t.task_id === "n_back" || t.task_id.includes("n_back"))
          return "memory" as const;
        if (t.task_id === "go_no_go" || t.task_id.includes("go_no_go"))
          return "go_no_go" as const;
        if (
          t.task_id === "trail_making_a" ||
          t.task_id.includes("trail_making_a")
        )
          return "trail_a" as const;
        if (
          t.task_id === "trail_making_b" ||
          t.task_id.includes("trail_making_b")
        )
          return "trail_b" as const;
        if (t.task_id === "digit_symbol" || t.task_id.includes("digit_symbol"))
          return "digit_symbol" as const;
        if (t.task_id === "stroop" || t.task_id.includes("stroop"))
          return "stroop" as const;
        return null;
      })
      .filter((id): id is Exclude<TestType, "none"> => id !== null);
  };

  const completedTests = getCompletedTestIds();

  // ==========================================================================
  // RENDER: ACTIVE TEST
  // ==========================================================================
  if (activeTest !== "none") {
    return (
      <div className="min-h-[600px] bg-black rounded-lg border border-zinc-800 p-8 flex flex-col relative overflow-hidden">
        <AnimatePresence mode="wait">
          {activeTest === "reaction" && (
            <motion.div
              key="reaction"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ReactionTest
                onComplete={handleTestComplete}
                onCancel={() => {
                  actions.cancelTest();
                  setActiveTest("none");
                }}
              />
            </motion.div>
          )}
          {activeTest === "memory" && (
            <motion.div
              key="memory"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <NBackTest
                n={2}
                onComplete={handleTestComplete}
                onCancel={() => {
                  actions.cancelTest();
                  setActiveTest("none");
                }}
              />
            </motion.div>
          )}
          {activeTest === "go_no_go" && (
            <motion.div
              key="go_no_go"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <GoNoGoTest
                onComplete={handleTestComplete}
                onCancel={() => {
                  actions.cancelTest();
                  setActiveTest("none");
                }}
              />
            </motion.div>
          )}
          {activeTest === "trail_a" && (
            <motion.div
              key="trail_a"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <TrailMakingTest
                part="A"
                onComplete={handleTestComplete}
                onCancel={() => {
                  actions.cancelTest();
                  setActiveTest("none");
                }}
              />
            </motion.div>
          )}
          {activeTest === "trail_b" && (
            <motion.div
              key="trail_b"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <TrailMakingTest
                part="B"
                onComplete={handleTestComplete}
                onCancel={() => {
                  actions.cancelTest();
                  setActiveTest("none");
                }}
              />
            </motion.div>
          )}
          {activeTest === "digit_symbol" && (
            <motion.div
              key="digit_symbol"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <DigitSymbolTest
                onComplete={handleTestComplete}
                onCancel={() => {
                  actions.cancelTest();
                  setActiveTest("none");
                }}
              />
            </motion.div>
          )}
          {activeTest === "stroop" && (
            <motion.div
              key="stroop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <StroopTest
                onComplete={handleTestComplete}
                onCancel={() => {
                  actions.cancelTest();
                  setActiveTest("none");
                }}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: RESULTS
  // ==========================================================================
  if (state.state === "success" || state.state === "partial") {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <ResultsPanel response={state.response!} onReset={actions.reset} />
      </div>
    );
  }

  // ==========================================================================
  // RENDER: SUBMITTING
  // ==========================================================================
  if (state.state === "submitting") {
    return (
      <div className="min-h-[400px] bg-zinc-900 rounded-lg border border-zinc-800 p-8 flex flex-col items-center justify-center">
        <Loader2 size={48} className="text-amber-400 animate-spin mb-4" />
        <h2 className="text-xl font-semibold text-zinc-100 mb-2">
          Analyzing Results
        </h2>
        <p className="text-zinc-400 text-sm">
          Processing {state.tasks.length} task(s)...
        </p>
        <div className="mt-6 flex gap-2">
          {["Validating", "Extracting", "Scoring", "Complete"].map(
            (stage, i) => (
              <div
                key={stage}
                className={`px-3 py-1 rounded text-xs ${i === 0 ? "bg-amber-500/20 text-amber-400 animate-pulse" : "bg-zinc-800 text-zinc-500"}`}
              >
                {stage}
              </div>
            ),
          )}
        </div>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: ERROR
  // ==========================================================================
  if (state.state === "error") {
    return (
      <div className="min-h-[400px] bg-zinc-900 rounded-lg border border-red-500/30 p-8 flex flex-col items-center justify-center">
        <XCircle size={48} className="text-red-400 mb-4" />
        <h2 className="text-xl font-semibold text-zinc-100 mb-2">
          Analysis Failed
        </h2>
        <p className="text-red-400 text-sm mb-6 text-center max-w-md">
          {state.error}
        </p>
        <div className="flex gap-4">
          {state.retryCount < 3 && (
            <button
              onClick={actions.submitSession}
              className="px-6 py-2 bg-amber-600 rounded-lg text-white hover:bg-amber-700 flex items-center gap-2"
            >
              <RefreshCw size={16} /> Retry
            </button>
          )}
          <button
            onClick={actions.reset}
            className="px-6 py-2 bg-zinc-700 rounded-lg text-zinc-200 hover:bg-zinc-600"
          >
            Start Over
          </button>
        </div>
        <p className="mt-4 text-xs text-zinc-500">Retry {state.retryCount}/3</p>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: IDLE (TEST SELECTION)
  // ==========================================================================
  return (
    <div className="space-y-6">
      {/* Session Status Banner */}
      {state.tasks.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-emerald-900/20 border border-emerald-500/30 p-4 rounded-lg flex justify-between items-center"
        >
          <div className="flex items-center gap-3">
            <CheckCircle className="text-emerald-400" size={20} />
            <div>
              <h3 className="text-emerald-200 font-medium">
                {state.tasks.length} Test(s) Completed
              </h3>
              <p className="text-xs text-emerald-400/70">
                {state.tasks.map((t) => t.task_id).join(", ")}
              </p>
            </div>
          </div>
          <button
            onClick={handleSubmit}
            className="px-4 py-2 bg-emerald-600 rounded text-xs font-bold text-white uppercase tracking-wider hover:bg-emerald-700"
          >
            Analyze Results
          </button>
        </motion.div>
      )}

      {/* Test Battery Selection */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-[16px] font-semibold text-zinc-100">
              Cognitive Test Battery
            </h2>
            <p className="text-xs text-zinc-500 mt-1">
              Select tests to include in your assessment
            </p>
          </div>
          <div className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">
            {completedTests.length}/{TEST_BATTERY.length} Complete
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {TEST_BATTERY.map((test) => {
            const colors = colorClasses[test.color]!;
            const isCompleted = completedTests.includes(test.id);

            return (
              <div
                key={test.id}
                onClick={() => test.available && handleStartTest(test.id)}
                className={`relative cursor-pointer rounded-lg border p-4 transition-all ${
                  test.available
                    ? `border-zinc-700 ${colors.hover} bg-zinc-800/50 hover:bg-zinc-800`
                    : "border-zinc-800 bg-zinc-900/50 cursor-not-allowed opacity-50"
                } ${isCompleted ? `${colors.border} ${colors.bg}` : ""}`}
              >
                {/* Completed badge */}
                {isCompleted && (
                  <div className="absolute top-2 right-2">
                    <CheckCircle size={16} className="text-emerald-400" />
                  </div>
                )}

                {/* Header */}
                <div className="mb-3 flex items-center gap-3">
                  <div
                    className={`p-2 rounded-lg ${colors.bg} transition-colors`}
                  >
                    <span className={colors.icon}>{test.icon}</span>
                  </div>
                  <div>
                    <h3 className="text-[14px] font-medium text-zinc-100">
                      {test.name}
                    </h3>
                    <span className={`text-[10px] ${colors.icon}`}>
                      {test.domain}
                    </span>
                  </div>
                </div>

                {/* Description */}
                <p className="mb-3 text-[11px] text-zinc-400 leading-relaxed line-clamp-2">
                  {test.description}
                </p>

                {/* Footer */}
                <div className="flex justify-between items-center text-[10px] text-zinc-500">
                  <span>{test.duration}</span>
                  {!test.available && (
                    <span className="text-zinc-600">Coming Soon</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Clinical Disclaimer */}
      <div className="bg-zinc-800/30 rounded-lg p-4 border border-zinc-800 flex items-start gap-4">
        <AlertTriangle
          className="text-zinc-500 mt-0.5 flex-shrink-0"
          size={16}
        />
        <div className="text-[11px] text-zinc-500">
          <strong className="text-zinc-400">Clinical Disclaimer:</strong> This
          cognitive screening tool is designed for research and wellness
          monitoring purposes. It is NOT a diagnostic device and should not
          replace professional neuropsychological evaluation. Results may be
          affected by fatigue, medication, environmental factors, or device
          latency. Consult a qualified healthcare professional for clinical
          interpretation.
        </div>
      </div>
    </div>
  );
}
