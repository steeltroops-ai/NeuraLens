"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  CheckCircle,
  AlertTriangle,
  AlertCircle,
  Clock,
  FileAudio,
  Download,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Brain,
  Activity,
  Info,
  Sparkles,
} from "lucide-react";
import { cn } from "@/utils/cn";
import { BiomarkerCard } from "./BiomarkerCard";
import { RiskBadge, getRiskLevelFromScore } from "@/components/ui/RiskBadge";
import type {
  EnhancedSpeechAnalysisResponse,
  BiomarkerDisplayConfig,
  ConditionRisk,
} from "@/types/speech-enhanced";
import {
  PRIMARY_BIOMARKER_CONFIGS,
  SECONDARY_BIOMARKER_CONFIGS,
  RESEARCH_BIOMARKER_CONFIGS,
  formatConditionName,
  getRiskLevelColor,
  CONDITION_EXPLANATIONS,
} from "@/types/speech-enhanced";

interface SpeechResultsPanelProps {
  results: EnhancedSpeechAnalysisResponse;
  onReset: () => void;
}

interface ConditionRiskCardProps {
  risk: ConditionRisk;
}

const ConditionRiskCard: React.FC<ConditionRiskCardProps> = ({ risk }) => {
  const [expanded, setExpanded] = React.useState(false);
  const explanation = CONDITION_EXPLANATIONS[risk.condition];
  const color = getRiskLevelColor(risk.risk_level);

  return (
    <div
      className="border border-[#e2e8f0] rounded-lg overflow-hidden"
      style={{ borderLeftColor: color, borderLeftWidth: "3px" }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-[#f8fafc] transition-colors"
      >
        <div className="flex items-center gap-3">
          <Brain className="h-4 w-4" style={{ color }} />
          <div className="text-left">
            <div className="text-[13px] font-medium text-[#0f172a]">
              {formatConditionName(risk.condition)}
            </div>
            <div className="text-[11px] text-[#64748b]">
              {(risk.probability * 100).toFixed(0)}% probability •{" "}
              {risk.risk_level} risk
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div
            className="px-2 py-0.5 rounded text-[10px] font-medium text-white"
            style={{ backgroundColor: color }}
          >
            {risk.risk_level.toUpperCase()}
          </div>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-[#64748b]" />
          ) : (
            <ChevronDown className="h-4 w-4 text-[#64748b]" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {expanded && explanation && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="px-3 pb-3 border-t border-[#e2e8f0]"
          >
            <div className="pt-3 space-y-2">
              <p className="text-[12px] text-[#475569]">
                {explanation.description}
              </p>
              <div className="text-[11px] text-[#64748b]">
                <strong>Key indicators:</strong>{" "}
                {explanation.indicators.join(", ")}
              </div>
              {risk.contributing_factors.length > 0 && (
                <div className="text-[11px] text-[#64748b]">
                  <strong>Your factors:</strong>{" "}
                  {risk.contributing_factors.join(", ")}
                </div>
              )}
              <div className="mt-2 p-2 bg-[#f1f5f9] rounded text-[11px] text-[#334155]">
                <strong>Recommended:</strong> {explanation.action}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export const SpeechResultsPanel: React.FC<SpeechResultsPanelProps> = ({
  results,
  onReset,
}) => {
  const [showAllBiomarkers, setShowAllBiomarkers] = React.useState(false);
  const [showResearch, setShowResearch] = React.useState(false);
  const [showConditions, setShowConditions] = React.useState(true);

  const riskScore = Math.round(results.risk_score * 100);
  const riskLevel = getRiskLevelFromScore(riskScore);
  const confidence = Math.round(results.confidence * 100);
  const qualityScore = Math.round(results.quality_score * 100);

  const getBiomarkerValue = (config: BiomarkerDisplayConfig) => {
    const key = config.key as keyof typeof results.biomarkers;
    return (
      results.biomarkers[key] ||
      results.extended_biomarkers?.[
        key as keyof typeof results.extended_biomarkers
      ]
    );
  };

  const getBaselineComparison = (key: string) => {
    return results.baseline_comparisons?.find((b) => b.biomarker_name === key);
  };

  // Filter condition risks to show only significant ones
  const significantConditions =
    results.condition_risks?.filter((c) => c.probability > 0.1) || [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Summary Card */}
      <div className="bg-white rounded-lg border border-[#e2e8f0] p-5">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[#dcfce7]">
              <CheckCircle className="h-5 w-5 text-[#22c55e]" />
            </div>
            <div>
              <h3 className="text-[14px] font-semibold text-[#0f172a]">
                Analysis Complete
              </h3>
              <p className="text-[12px] text-[#64748b]">
                Processed in {results.processing_time.toFixed(2)}s
              </p>
            </div>
          </div>
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-[#64748b] hover:text-[#334155] hover:bg-[#f1f5f9] rounded-md transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            New Analysis
          </button>
        </div>

        {/* Risk Score Display */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="text-center p-4 bg-[#f8fafc] rounded-lg">
            <div className="text-[32px] font-bold text-[#0f172a] leading-none mb-1">
              {riskScore}
            </div>
            <div className="text-[11px] text-[#64748b] mb-2">Risk Score</div>
            <RiskBadge level={riskLevel} size="sm" />
          </div>

          <div className="text-center p-4 bg-[#f8fafc] rounded-lg">
            <div className="text-[32px] font-bold text-[#3b82f6] leading-none mb-1">
              {confidence}%
            </div>
            <div className="text-[11px] text-[#64748b]">Confidence</div>
            {results.confidence_interval && (
              <div className="text-[10px] text-[#94a3b8] mt-1">
                95% CI: {results.confidence_interval[0].toFixed(0)}-
                {results.confidence_interval[1].toFixed(0)}
              </div>
            )}
          </div>

          <div className="text-center p-4 bg-[#f8fafc] rounded-lg">
            <div className="text-[32px] font-bold text-[#22c55e] leading-none mb-1">
              {qualityScore}%
            </div>
            <div className="text-[11px] text-[#64748b]">Audio Quality</div>
          </div>
        </div>

        {/* Clinical Notes */}
        {results.clinical_notes && (
          <div className="p-3 bg-[#eff6ff] border border-[#bfdbfe] rounded-lg mb-4">
            <div className="flex items-start gap-2">
              <Info className="h-4 w-4 text-[#3b82f6] mt-0.5 flex-shrink-0" />
              <p className="text-[12px] text-[#1e40af]">
                {results.clinical_notes}
              </p>
            </div>
          </div>
        )}

        {/* Review Warning */}
        {results.requires_review && (
          <div className="p-3 bg-[#fef3c7] border border-[#fcd34d] rounded-lg mb-4">
            <div className="flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-[#d97706] mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-[12px] font-medium text-[#92400e]">
                  Clinical Review Recommended
                </p>
                {results.review_reason && (
                  <p className="text-[11px] text-[#a16207] mt-0.5">
                    {results.review_reason}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* File Info */}
        {results.file_info && (
          <div className="flex items-center gap-4 p-3 bg-[#f8fafc] rounded-lg text-[12px]">
            <FileAudio className="h-4 w-4 text-[#64748b]" />
            <div className="flex-1 flex items-center gap-4 text-[#64748b]">
              {results.file_info.duration && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {results.file_info.duration.toFixed(1)}s
                </span>
              )}
              {results.file_info.sample_rate && (
                <span>{results.file_info.sample_rate} Hz</span>
              )}
              {results.file_info.size && (
                <span>{(results.file_info.size / 1024).toFixed(1)} KB</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Condition Risks */}
      {significantConditions.length > 0 && (
        <div className="bg-white rounded-lg border border-[#e2e8f0]">
          <button
            onClick={() => setShowConditions(!showConditions)}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-[#f8fafc] transition-colors"
          >
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#ef4444]" />
              <span className="text-[14px] font-semibold text-[#0f172a]">
                Condition Risk Assessment
              </span>
              <span className="px-2 py-0.5 bg-[#fef2f2] text-[#dc2626] text-[10px] font-medium rounded">
                {significantConditions.length} detected
              </span>
            </div>
            {showConditions ? (
              <ChevronUp className="h-4 w-4 text-[#64748b]" />
            ) : (
              <ChevronDown className="h-4 w-4 text-[#64748b]" />
            )}
          </button>

          <AnimatePresence>
            {showConditions && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="px-4 pb-4 space-y-2"
              >
                {significantConditions.map((risk) => (
                  <ConditionRiskCard key={risk.condition} risk={risk} />
                ))}
                <p className="text-[10px] text-[#94a3b8] mt-2 italic">
                  These are screening indicators only, not diagnoses. Consult a
                  healthcare provider for proper evaluation.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Primary Biomarkers */}
      <div className="bg-white rounded-lg border border-[#e2e8f0] p-5">
        <h3 className="text-[14px] font-semibold text-[#0f172a] mb-4">
          Key Biomarkers
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {PRIMARY_BIOMARKER_CONFIGS.map((config) => {
            const biomarker = getBiomarkerValue(config);
            if (!biomarker) return null;
            return (
              <BiomarkerCard
                key={config.key}
                name={config.key}
                label={config.label}
                description={config.description}
                biomarker={biomarker}
                iconKey={config.icon}
                higherIsBetter={config.higherIsBetter}
                formatValue={config.formatValue}
                baseline={getBaselineComparison(config.key)}
              />
            );
          })}
        </div>
      </div>

      {/* Secondary Biomarkers (Collapsible) */}
      <div className="bg-white rounded-lg border border-[#e2e8f0]">
        <button
          onClick={() => setShowAllBiomarkers(!showAllBiomarkers)}
          className="w-full flex items-center justify-between p-4 text-left hover:bg-[#f8fafc] transition-colors"
        >
          <span className="text-[14px] font-semibold text-[#0f172a]">
            Additional Biomarkers
          </span>
          {showAllBiomarkers ? (
            <ChevronUp className="h-4 w-4 text-[#64748b]" />
          ) : (
            <ChevronDown className="h-4 w-4 text-[#64748b]" />
          )}
        </button>

        <AnimatePresence>
          {showAllBiomarkers && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="px-5 pb-5"
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {SECONDARY_BIOMARKER_CONFIGS.map((config) => {
                  const biomarker = getBiomarkerValue(config);
                  if (!biomarker) return null;
                  return (
                    <BiomarkerCard
                      key={config.key}
                      name={config.key}
                      label={config.label}
                      description={config.description}
                      biomarker={biomarker}
                      iconKey={config.icon}
                      higherIsBetter={config.higherIsBetter}
                      formatValue={config.formatValue}
                      baseline={getBaselineComparison(config.key)}
                    />
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Research Biomarkers */}
      {results.extended_biomarkers && (
        <div className="bg-white rounded-lg border border-[#e2e8f0]">
          <button
            onClick={() => setShowResearch(!showResearch)}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-[#f8fafc] transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-[#8b5cf6]" />
              <span className="text-[14px] font-semibold text-[#0f172a]">
                Research-Grade Biomarkers
              </span>
              <span className="px-2 py-0.5 bg-[#f3e8ff] text-[#7c3aed] text-[10px] font-medium rounded">
                EXPERIMENTAL
              </span>
            </div>
            {showResearch ? (
              <ChevronUp className="h-4 w-4 text-[#64748b]" />
            ) : (
              <ChevronDown className="h-4 w-4 text-[#64748b]" />
            )}
          </button>

          <AnimatePresence>
            {showResearch && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="px-5 pb-5"
              >
                <p className="text-[11px] text-[#64748b] mb-3">
                  These novel biomarkers are under active research and may
                  provide additional clinical insights. Results should be
                  interpreted with caution.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {RESEARCH_BIOMARKER_CONFIGS.map((config) => {
                    const key =
                      config.key as keyof typeof results.extended_biomarkers;
                    const biomarker = results.extended_biomarkers?.[key];
                    if (!biomarker) return null;
                    return (
                      <BiomarkerCard
                        key={config.key}
                        name={config.key}
                        label={config.label}
                        description={config.description}
                        biomarker={biomarker}
                        iconKey={config.icon}
                        higherIsBetter={config.higherIsBetter}
                        formatValue={config.formatValue}
                        baseline={getBaselineComparison(config.key)}
                        isResearch
                      />
                    );
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Recommendations */}
      {results.recommendations && results.recommendations.length > 0 && (
        <div className="bg-white rounded-lg border border-[#e2e8f0] p-5">
          <h3 className="text-[14px] font-semibold text-[#0f172a] mb-3">
            Clinical Recommendations
          </h3>
          <ul className="space-y-2">
            {results.recommendations.map((rec, index) => (
              <li
                key={index}
                className="flex items-start gap-2 text-[13px] text-[#475569]"
              >
                <span className="text-[#3b82f6] mt-0.5">•</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Disclaimer */}
      <div className="p-4 bg-[#f8fafc] border border-[#e2e8f0] rounded-lg">
        <div className="flex items-start gap-2">
          <AlertCircle className="h-4 w-4 text-[#64748b] mt-0.5 flex-shrink-0" />
          <p className="text-[11px] text-[#64748b] leading-relaxed">
            <strong>Disclaimer:</strong> This analysis is for informational
            screening purposes only and is NOT a medical diagnosis. Voice
            biomarkers can be affected by recording conditions, fatigue,
            medications, and temporary illness. Always consult a qualified
            healthcare provider for medical advice, diagnosis, or treatment.
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-end gap-3">
        <button className="flex items-center gap-2 px-4 py-2 text-[13px] font-medium text-[#334155] bg-white border border-[#e2e8f0] rounded-md hover:bg-[#f1f5f9] transition-colors">
          <Download className="h-4 w-4" />
          Export Report
        </button>
      </div>
    </motion.div>
  );
};

export default SpeechResultsPanel;
