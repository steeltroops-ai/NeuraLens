'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
    CheckCircle,
    AlertTriangle,
    Clock,
    FileAudio,
    Download,
    RefreshCw,
    ChevronDown,
    ChevronUp,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { BiomarkerCard } from './BiomarkerCard';
import {
    RiskBadge,
    getRiskLevelFromScore,
} from '@/components/ui/RiskBadge';
import type {
    EnhancedSpeechAnalysisResponse,
    BiomarkerDisplayConfig,
} from '@/types/speech-enhanced';
import { BIOMARKER_CONFIGS } from '@/types/speech-enhanced';

interface SpeechResultsPanelProps {
    results: EnhancedSpeechAnalysisResponse;
    onReset: () => void;
}

export const SpeechResultsPanel: React.FC<SpeechResultsPanelProps> = ({
    results,
    onReset,
}) => {
    const [showAllBiomarkers, setShowAllBiomarkers] = React.useState(false);

    const riskScore = Math.round(results.risk_score * 100);
    const riskLevel = getRiskLevelFromScore(riskScore);
    const confidence = Math.round(results.confidence * 100);
    const qualityScore = Math.round(results.quality_score * 100);

    // Get primary biomarkers (first 4) and secondary (rest)
    const primaryBiomarkers = BIOMARKER_CONFIGS.slice(0, 4);
    const secondaryBiomarkers = BIOMARKER_CONFIGS.slice(4);

    const getBiomarkerValue = (config: BiomarkerDisplayConfig) => {
        return results.biomarkers[config.key];
    };

    const getBaselineComparison = (key: string) => {
        return results.baseline_comparisons?.find((b) => b.biomarker_name === key);
    };

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
                    </div>

                    <div className="text-center p-4 bg-[#f8fafc] rounded-lg">
                        <div className="text-[32px] font-bold text-[#22c55e] leading-none mb-1">
                            {qualityScore}%
                        </div>
                        <div className="text-[11px] text-[#64748b]">Audio Quality</div>
                    </div>
                </div>

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

            {/* Primary Biomarkers */}
            <div className="bg-white rounded-lg border border-[#e2e8f0] p-5">
                <h3 className="text-[14px] font-semibold text-[#0f172a] mb-4">
                    Key Biomarkers
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {primaryBiomarkers.map((config) => {
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

                {showAllBiomarkers && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="px-5 pb-5"
                    >
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                            {secondaryBiomarkers.map((config) => {
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
            </div>

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

            {/* Actions */}
            <div className="flex items-center justify-end gap-3">
                <button
                    className="flex items-center gap-2 px-4 py-2 text-[13px] font-medium text-[#334155] bg-white border border-[#e2e8f0] rounded-md hover:bg-[#f1f5f9] transition-colors"
                >
                    <Download className="h-4 w-4" />
                    Export Report
                </button>
            </div>
        </motion.div>
    );
};

export default SpeechResultsPanel;
