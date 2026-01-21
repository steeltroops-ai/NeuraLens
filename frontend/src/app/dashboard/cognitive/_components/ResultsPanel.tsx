"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  RefreshCw,
  Brain,
  Activity,
  Zap,
  BarChart3,
  ChevronRight,
  Info,
} from "lucide-react";
import {
  CognitiveResponse,
  RiskLevel,
  DomainRiskDetail,
  ClinicalRecommendation,
} from "../types";

interface ResultsPanelProps {
  response: CognitiveResponse;
  onReset: () => void;
}

// =============================================================================
// HELPERS
// =============================================================================

const riskColors: Record<
  RiskLevel,
  { bg: string; text: string; border: string }
> = {
  low: {
    bg: "bg-emerald-500/10",
    text: "text-emerald-400",
    border: "border-emerald-500/30",
  },
  moderate: {
    bg: "bg-amber-500/10",
    text: "text-amber-400",
    border: "border-amber-500/30",
  },
  high: {
    bg: "bg-orange-500/10",
    text: "text-orange-400",
    border: "border-orange-500/30",
  },
  critical: {
    bg: "bg-red-500/10",
    text: "text-red-400",
    border: "border-red-500/30",
  },
};

const priorityColors: Record<string, string> = {
  low: "text-zinc-400",
  medium: "text-amber-400",
  high: "text-orange-400",
  critical: "text-red-400",
};

const domainIcons: Record<string, React.ReactNode> = {
  memory: <Brain size={16} />,
  processing_speed: <Zap size={16} />,
  inhibition: <Activity size={16} />,
  attention: <BarChart3 size={16} />,
  general: <Info size={16} />,
};

// =============================================================================
// COMPONENTS
// =============================================================================

function RiskGauge({ score, level }: { score: number; level: RiskLevel }) {
  const percentage = Math.round(score * 100);
  const colors = riskColors[level];

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        {/* Background circle */}
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="64"
            cy="64"
            r="56"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-zinc-800"
          />
          <circle
            cx="64"
            cy="64"
            r="56"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            strokeDasharray={`${percentage * 3.52} 352`}
            strokeLinecap="round"
            className={colors.text}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${colors.text}`}>
            {percentage}
          </span>
          <span className="text-xs text-zinc-500">Risk Score</span>
        </div>
      </div>
      <div
        className={`mt-2 px-3 py-1 rounded-full ${colors.bg} ${colors.border} border`}
      >
        <span
          className={`text-xs font-medium uppercase tracking-wider ${colors.text}`}
        >
          {level}
        </span>
      </div>
    </div>
  );
}

function DomainCard({
  domain,
  detail,
}: {
  domain: string;
  detail: DomainRiskDetail;
}) {
  const colors = riskColors[detail.risk_level];
  const icon = domainIcons[domain] || <Info size={16} />;
  const percentage = Math.round(detail.score * 100);

  return (
    <div className={`p-4 rounded-lg border ${colors.border} ${colors.bg}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={colors.text}>{icon}</span>
          <span className="text-sm font-medium text-zinc-200 capitalize">
            {domain.replace("_", " ")}
          </span>
        </div>
        <span
          className={`text-xs px-2 py-0.5 rounded ${colors.bg} ${colors.text} border ${colors.border}`}
        >
          {detail.risk_level}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden mb-2">
        <div
          className={`h-full ${colors.text.replace("text-", "bg-")} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      <div className="flex justify-between text-xs">
        <span className="text-zinc-500">Risk: {percentage}%</span>
        {detail.percentile && (
          <span className="text-zinc-500">Percentile: {detail.percentile}</span>
        )}
      </div>

      {detail.contributing_factors.length > 0 && (
        <div className="mt-3 pt-3 border-t border-zinc-700/50">
          <ul className="space-y-1">
            {detail.contributing_factors.map((factor, i) => (
              <li
                key={i}
                className="text-xs text-zinc-400 flex items-start gap-1"
              >
                <ChevronRight size={12} className="mt-0.5 flex-shrink-0" />
                {factor}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function RecommendationCard({ rec }: { rec: ClinicalRecommendation }) {
  const color = priorityColors[rec.priority] || "text-zinc-400";

  return (
    <div className="p-4 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 ${color}`}>
          {rec.priority === "critical" || rec.priority === "high" ? (
            <AlertTriangle size={16} />
          ) : (
            <CheckCircle size={16} />
          )}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-xs uppercase tracking-wider ${color}`}>
              {rec.priority}
            </span>
            <span className="text-xs text-zinc-500 capitalize">
              {rec.category}
            </span>
          </div>
          <p className="text-sm text-zinc-300">{rec.description}</p>
        </div>
      </div>
    </div>
  );
}

function StageTimeline({ stages }: { stages: CognitiveResponse["stages"] }) {
  return (
    <div className="flex items-center gap-2 p-4 bg-zinc-900 rounded-lg border border-zinc-800">
      {stages.map((stage, i) => (
        <React.Fragment key={i}>
          <div className="flex items-center gap-2">
            {stage.stage === "complete" ? (
              <CheckCircle size={14} className="text-emerald-400" />
            ) : stage.stage === "failed" ? (
              <XCircle size={14} className="text-red-400" />
            ) : (
              <div className="w-3.5 h-3.5 rounded-full bg-zinc-600 animate-pulse" />
            )}
            <span className="text-xs text-zinc-400">{stage.message}</span>
            {stage.duration_ms && (
              <span className="text-xs text-zinc-600">
                ({stage.duration_ms.toFixed(0)}ms)
              </span>
            )}
          </div>
          {i < stages.length - 1 && <div className="flex-1 h-px bg-zinc-700" />}
        </React.Fragment>
      ))}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function ResultsPanel({ response, onReset }: ResultsPanelProps) {
  const { risk_assessment, features, recommendations, explainability, stages } =
    response;

  if (!risk_assessment) {
    return (
      <div className="p-8 text-center">
        <XCircle size={48} className="mx-auto text-red-400 mb-4" />
        <h2 className="text-xl font-semibold text-zinc-100 mb-2">
          Analysis Failed
        </h2>
        <p className="text-zinc-400 mb-6">
          {response.error_message || "Unknown error occurred"}
        </p>
        <button
          onClick={onReset}
          className="px-6 py-2 bg-zinc-700 rounded-lg text-zinc-200 hover:bg-zinc-600 flex items-center gap-2 mx-auto"
        >
          <RefreshCw size={16} /> Try Again
        </button>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100">
            Assessment Results
          </h2>
          <p className="text-sm text-zinc-500">
            Session: {response.session_id} | Processed in{" "}
            {response.processing_time_ms.toFixed(0)}ms
          </p>
        </div>
        <button
          onClick={onReset}
          className="px-4 py-2 bg-zinc-800 rounded-lg text-zinc-300 hover:bg-zinc-700 flex items-center gap-2"
        >
          <RefreshCw size={14} /> New Assessment
        </button>
      </div>

      {/* Pipeline Stages */}
      <StageTimeline stages={stages} />

      {/* Main Results Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Gauge */}
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 flex flex-col items-center justify-center">
          <RiskGauge
            score={risk_assessment.overall_risk_score}
            level={risk_assessment.risk_level}
          />
          <div className="mt-4 text-center">
            <p className="text-xs text-zinc-500">
              Confidence: {Math.round(risk_assessment.confidence_score * 100)}%
            </p>
            <p className="text-xs text-zinc-600">
              95% CI: [
              {Math.round(risk_assessment.confidence_interval[0] * 100)},{" "}
              {Math.round(risk_assessment.confidence_interval[1] * 100)}]
            </p>
          </div>
        </div>

        {/* Explainability */}
        {explainability && (
          <div className="lg:col-span-2 bg-zinc-900 rounded-lg border border-zinc-800 p-6">
            <h3 className="text-sm font-medium text-zinc-200 mb-3">Summary</h3>
            <p className="text-sm text-zinc-400 mb-4">
              {explainability.summary}
            </p>

            {explainability.key_factors.length > 0 && (
              <>
                <h4 className="text-xs font-medium text-zinc-400 mb-2 uppercase tracking-wider">
                  Key Factors
                </h4>
                <ul className="space-y-1 mb-4">
                  {explainability.key_factors.map((factor, i) => (
                    <li
                      key={i}
                      className="text-sm text-zinc-300 flex items-center gap-2"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                      {factor}
                    </li>
                  ))}
                </ul>
              </>
            )}

            <p className="text-xs text-zinc-600 italic">
              {explainability.methodology_note}
            </p>
          </div>
        )}
      </div>

      {/* Domain Scores */}
      {Object.keys(risk_assessment.domain_risks).length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-zinc-200 mb-4">
            Domain Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(risk_assessment.domain_risks).map(
              ([domain, detail]) => (
                <DomainCard key={domain} domain={domain} detail={detail} />
              ),
            )}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-zinc-200 mb-4">
            Recommendations
          </h3>
          <div className="space-y-3">
            {recommendations.map((rec, i) => (
              <RecommendationCard key={i} rec={rec} />
            ))}
          </div>
        </div>
      )}

      {/* Task Metrics (Collapsible) */}
      {features && features.raw_metrics.length > 0 && (
        <details className="bg-zinc-900 rounded-lg border border-zinc-800">
          <summary className="p-4 cursor-pointer text-sm font-medium text-zinc-300 hover:text-zinc-100">
            Task Details ({features.raw_metrics.length} tasks)
          </summary>
          <div className="p-4 pt-0 space-y-3">
            {features.raw_metrics.map((metric, i) => (
              <div key={i} className="p-3 bg-zinc-800/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-mono text-zinc-300">
                    {metric.task_id}
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      metric.validity_flag
                        ? "bg-emerald-500/20 text-emerald-400"
                        : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {metric.completion_status}
                  </span>
                </div>
                <div className="text-xs text-zinc-500">
                  Score: {metric.performance_score.toFixed(1)} |
                  {Object.entries(metric.parameters)
                    .slice(0, 3)
                    .map(([k, v]) => (
                      <span key={k}>
                        {" "}
                        {k}: {typeof v === "number" ? v.toFixed(2) : v}
                      </span>
                    ))}
                </div>
                {metric.quality_warnings.length > 0 && (
                  <div className="mt-2 text-xs text-amber-400">
                    {metric.quality_warnings.join(", ")}
                  </div>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Clinical Disclaimer */}
      <div className="p-4 bg-zinc-800/30 rounded-lg border border-zinc-800 flex items-start gap-3">
        <AlertTriangle
          size={16}
          className="text-zinc-500 mt-0.5 flex-shrink-0"
        />
        <p className="text-xs text-zinc-500">
          <strong className="text-zinc-400">Clinical Disclaimer:</strong> This
          tool is a screening aid and not a diagnostic device. Results should be
          interpreted by a qualified healthcare professional.
        </p>
      </div>
    </motion.div>
  );
}
