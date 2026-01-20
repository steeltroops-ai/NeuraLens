"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Lightbulb,
  ChevronRight,
  Star,
  Target,
} from "lucide-react";

interface AIInsight {
  id: string;
  type: "recommendation" | "alert" | "trend" | "achievement";
  title: string;
  description: string;
  confidence: number;
  priority: "high" | "medium" | "low";
  actionable: boolean;
  timestamp: string;
}

interface AIInsightsPanelProps {
  insights?: AIInsight[];
  maxInsights?: number;
}

export default function AIInsightsPanel({
  insights,
  maxInsights = 6,
}: AIInsightsPanelProps) {
  const [selectedInsight, setSelectedInsight] = useState<string | null>(null);

  const defaultInsights: AIInsight[] = [
    {
      id: "1",
      type: "recommendation",
      title: "Optimize Speech Assessment Frequency",
      description:
        "Based on patient patterns, increasing speech assessments to twice weekly could improve early detection by 23%.",
      confidence: 94,
      priority: "high",
      actionable: true,
      timestamp: "2025-08-23T14:30:00Z",
    },
    {
      id: "2",
      type: "trend",
      title: "Cognitive Scores Trending Upward",
      description:
        "Patient cognitive assessment scores have improved 15% over the last month, indicating positive treatment response.",
      confidence: 87,
      priority: "medium",
      actionable: false,
      timestamp: "2025-08-23T14:15:00Z",
    },
    {
      id: "3",
      type: "alert",
      title: "Motor Function Variance Detected",
      description:
        "Unusual variance in motor function tests suggests need for additional evaluation or equipment calibration.",
      confidence: 91,
      priority: "high",
      actionable: true,
      timestamp: "2025-08-23T14:00:00Z",
    },
    {
      id: "4",
      type: "achievement",
      title: "Assessment Accuracy Milestone",
      description:
        "Your assessment accuracy has reached 98.2%, placing you in the top 5% of healthcare professionals.",
      confidence: 99,
      priority: "low",
      actionable: false,
      timestamp: "2025-08-23T13:45:00Z",
    },
    {
      id: "5",
      type: "recommendation",
      title: "Retinal Imaging Protocol Update",
      description:
        "New AI model suggests adjusting retinal imaging parameters for 12% improvement in diagnostic accuracy.",
      confidence: 89,
      priority: "medium",
      actionable: true,
      timestamp: "2025-08-23T13:30:00Z",
    },
    {
      id: "6",
      type: "trend",
      title: "Multi-Modal Fusion Efficiency",
      description:
        "NRI fusion processing time has decreased by 18% while maintaining accuracy, indicating system optimization success.",
      confidence: 92,
      priority: "low",
      actionable: false,
      timestamp: "2025-08-23T13:15:00Z",
    },
  ];

  const insightList = (insights || defaultInsights).slice(0, maxInsights);

  const getInsightIcon = (type: string) => {
    switch (type) {
      case "recommendation":
        return <Lightbulb size={14} strokeWidth={1.5} />;
      case "alert":
        return <AlertTriangle size={14} strokeWidth={1.5} />;
      case "trend":
        return <TrendingUp size={14} strokeWidth={1.5} />;
      case "achievement":
        return <Star size={14} strokeWidth={1.5} />;
      default:
        return <Brain size={14} strokeWidth={1.5} />;
    }
  };

  const getInsightColor = (type: string, priority: string) => {
    if (priority === "high") return "#ef4444";

    switch (type) {
      case "recommendation":
        return "#3b82f6";
      case "alert":
        return "#f59e0b";
      case "trend":
        return "#22c55e";
      case "achievement":
        return "#eab308";
      default:
        return "#64748b";
    }
  };

  const getPriorityBadge = (priority: string) => {
    const colors = {
      high: { bg: "#fee2e2", text: "#991b1b" },
      medium: { bg: "#fef3c7", text: "#92400e" },
      low: { bg: "#dcfce7", text: "#166534" },
    };
    const color = colors[priority as keyof typeof colors];

    return (
      <span
        className="rounded px-1.5 py-0.5 text-[10px] font-medium"
        style={{ backgroundColor: color.bg, color: color.text }}
      >
        {priority.toUpperCase()}
      </span>
    );
  };

  return (
    <div className="space-y-2">
      {insightList.map((insight, index) => (
        <motion.div
          key={insight.id}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15, delay: index * 0.03 }}
          className={`cursor-pointer rounded-lg border p-3 transition-colors duration-150 ${
            selectedInsight === insight.id
              ? "bg-zinc-800 border-zinc-600"
              : "bg-zinc-800/50 border-zinc-700/50 hover:bg-zinc-800"
          }`}
          onClick={() =>
            setSelectedInsight(
              selectedInsight === insight.id ? null : insight.id,
            )
          }
        >
          <div className="flex items-start gap-2.5">
            {/* Icon */}
            <div
              className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md"
              style={{
                backgroundColor: `${getInsightColor(insight.type, insight.priority)}15`,
              }}
            >
              {React.cloneElement(
                getInsightIcon(insight.type) as React.ReactElement,
                {
                  style: {
                    color: getInsightColor(insight.type, insight.priority),
                  },
                },
              )}
            </div>

            {/* Content */}
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 mb-0.5">
                <h4 className="text-[12px] font-medium text-zinc-200 truncate">
                  {insight.title}
                </h4>
                {getPriorityBadge(insight.priority)}
              </div>
              <p className="text-[11px] text-zinc-400 line-clamp-2">
                {insight.description}
              </p>
              <div className="mt-1.5 flex items-center gap-3">
                <span className="text-[10px] text-zinc-500">
                  {insight.confidence}% confidence
                </span>
                {insight.actionable && (
                  <span className="text-[10px] font-medium text-[#3b82f6]">
                    Actionable
                  </span>
                )}
              </div>
            </div>

            {/* Expand Icon */}
            <motion.div
              animate={{ rotate: selectedInsight === insight.id ? 90 : 0 }}
              transition={{ duration: 0.15 }}
            >
              <ChevronRight size={14} className="text-zinc-500" />
            </motion.div>
          </div>

          {/* Expanded Content */}
          <AnimatePresence>
            {selectedInsight === insight.id && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.15 }}
                className="mt-3 pt-3 border-t border-zinc-700"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1">
                    <Target size={12} className="text-zinc-500" />
                    <span className="text-[10px] text-zinc-500">
                      {new Date(insight.timestamp).toLocaleString()}
                    </span>
                  </div>
                  {insight.actionable && (
                    <button className="px-2.5 py-1 text-[11px] font-medium text-white bg-[#3b82f6] hover:bg-[#2563eb] rounded-md transition-colors">
                      Take Action
                    </button>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      ))}
    </div>
  );
}
