/**
 * Enhanced Speech Results Display Component
 * 
 * Displays all 9 biomarkers with values, confidence indicators,
 * estimated value markers, and baseline comparison.
 * 
 * Feature: speech-pipeline-fix
 * **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
 */

'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
    Activity,
    AlertCircle,
    CheckCircle,
    TrendingUp,
    TrendingDown,
    Minus,
    Info,
    HelpCircle,
} from 'lucide-react';

/**
 * Individual biomarker result with metadata
 */
export interface BiomarkerResult {
    value: number;
    unit: string;
    normalRange: [number, number];
    isEstimated: boolean;
    confidence?: number;
}

/**
 * Complete speech analysis result structure
 */
export interface EnhancedSpeechAnalysisResult {
    sessionId: string;
    processingTime: number;
    timestamp: string;
    confidence: number;
    riskScore: number;
    qualityScore: number;

    biomarkers: {
        jitter: BiomarkerResult;
        shimmer: BiomarkerResult;
        hnr: BiomarkerResult;
        speechRate: BiomarkerResult;
        pauseRatio: BiomarkerResult;
        fluencyScore: BiomarkerResult;
        voiceTremor: BiomarkerResult;
        articulationClarity: BiomarkerResult;
        prosodyVariation: BiomarkerResult;
    };

    recommendations: string[];
    baselineComparison?: Record<string, number>;
}

/**
 * Props for the EnhancedSpeechResults component
 */
interface EnhancedSpeechResultsProps {
    result: EnhancedSpeechAnalysisResult;
    className?: string;
}

/**
 * Biomarker display configuration
 */
interface BiomarkerConfig {
    key: keyof EnhancedSpeechAnalysisResult['biomarkers'];
    label: string;
    description: string;
    icon: React.ReactNode;
    colorClass: string;
    formatValue: (value: number) => string;
}


/**
 * All 9 biomarker configurations with display metadata
 */
const BIOMARKER_CONFIGS: BiomarkerConfig[] = [
    {
        key: 'jitter',
        label: 'Jitter',
        description: 'Fundamental frequency variation - measures pitch stability',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-blue-600 bg-blue-50',
        formatValue: (v) => `${(v * 100).toFixed(2)}%`,
    },
    {
        key: 'shimmer',
        label: 'Shimmer',
        description: 'Amplitude variation - measures volume stability',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-indigo-600 bg-indigo-50',
        formatValue: (v) => `${(v * 100).toFixed(2)}%`,
    },
    {
        key: 'hnr',
        label: 'HNR',
        description: 'Harmonics-to-Noise Ratio - measures voice clarity',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-purple-600 bg-purple-50',
        formatValue: (v) => `${v.toFixed(1)} dB`,
    },
    {
        key: 'speechRate',
        label: 'Speech Rate',
        description: 'Syllables per second - measures speaking pace',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-teal-600 bg-teal-50',
        formatValue: (v) => `${v.toFixed(1)} syl/s`,
    },
    {
        key: 'pauseRatio',
        label: 'Pause Ratio',
        description: 'Proportion of silence - measures speech continuity',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-cyan-600 bg-cyan-50',
        formatValue: (v) => `${(v * 100).toFixed(1)}%`,
    },
    {
        key: 'fluencyScore',
        label: 'Fluency Score',
        description: 'Overall speech fluency - measures smoothness',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-green-600 bg-green-50',
        formatValue: (v) => `${(v * 100).toFixed(1)}%`,
    },
    {
        key: 'voiceTremor',
        label: 'Voice Tremor',
        description: 'Tremor intensity - measures voice steadiness',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-orange-600 bg-orange-50',
        formatValue: (v) => `${(v * 100).toFixed(1)}%`,
    },
    {
        key: 'articulationClarity',
        label: 'Articulation Clarity',
        description: 'Clarity of speech sounds - measures pronunciation',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-rose-600 bg-rose-50',
        formatValue: (v) => `${(v * 100).toFixed(1)}%`,
    },
    {
        key: 'prosodyVariation',
        label: 'Prosody Variation',
        description: 'Pitch and rhythm variation - measures expressiveness',
        icon: <Activity className="h-4 w-4" />,
        colorClass: 'text-amber-600 bg-amber-50',
        formatValue: (v) => `${(v * 100).toFixed(1)}%`,
    },
];

/**
 * Get confidence level label and color
 */
function getConfidenceLevel(confidence: number): { label: string; colorClass: string } {
    if (confidence >= 0.9) {
        return { label: 'High', colorClass: 'text-green-600 bg-green-100' };
    } else if (confidence >= 0.7) {
        return { label: 'Medium', colorClass: 'text-yellow-600 bg-yellow-100' };
    } else {
        return { label: 'Low', colorClass: 'text-red-600 bg-red-100' };
    }
}

/**
 * Check if a value is within normal range
 */
function isWithinNormalRange(value: number, range: [number, number]): boolean {
    return value >= range[0] && value <= range[1];
}

/**
 * Get delta direction indicator
 */
function getDeltaIndicator(delta: number): React.ReactNode {
    if (delta > 0.05) {
        return <TrendingUp className="h-4 w-4 text-green-600" />;
    } else if (delta < -0.05) {
        return <TrendingDown className="h-4 w-4 text-red-600" />;
    }
    return <Minus className="h-4 w-4 text-gray-400" />;
}

/**
 * Format delta value for display
 */
function formatDelta(delta: number): string {
    const sign = delta >= 0 ? '+' : '';
    return `${sign}${(delta * 100).toFixed(1)}%`;
}


/**
 * Estimated Value Tooltip Component
 * Shows explanation when a biomarker value is estimated
 */
function EstimatedTooltip(): React.ReactNode {
    return (
        <div className="group relative inline-flex items-center ml-1">
            <HelpCircle className="h-3.5 w-3.5 text-amber-500 cursor-help" />
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 whitespace-nowrap z-10">
                <div className="font-medium mb-1">Estimated Value</div>
                <div className="text-gray-300">
                    This value was estimated using clinically-validated defaults
                    <br />
                    due to insufficient data in the audio sample.
                </div>
                <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 border-4 border-transparent border-t-gray-900" />
            </div>
        </div>
    );
}

/**
 * Individual Biomarker Card Component
 */
interface BiomarkerCardProps {
    config: BiomarkerConfig;
    biomarker: BiomarkerResult;
    baselineDelta?: number;
}

function BiomarkerCard({ config, biomarker, baselineDelta }: BiomarkerCardProps): React.ReactNode {
    const isNormal = isWithinNormalRange(biomarker.value, biomarker.normalRange);
    const confidenceInfo = biomarker.confidence !== undefined
        ? getConfidenceLevel(biomarker.confidence)
        : null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm hover:shadow-md transition-shadow"
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <div className={`p-1.5 rounded-md ${config.colorClass}`}>
                        {config.icon}
                    </div>
                    <span className="font-medium text-gray-900">{config.label}</span>
                    {biomarker.isEstimated && <EstimatedTooltip />}
                </div>
                {confidenceInfo && (
                    <span className={`text-xs px-2 py-0.5 rounded-full ${confidenceInfo.colorClass}`}>
                        {confidenceInfo.label}
                    </span>
                )}
            </div>

            {/* Value */}
            <div className="flex items-baseline gap-2 mb-2">
                <span className="text-2xl font-bold text-gray-900">
                    {config.formatValue(biomarker.value)}
                </span>
                {baselineDelta !== undefined && (
                    <div className="flex items-center gap-1 text-sm">
                        {getDeltaIndicator(baselineDelta)}
                        <span className={baselineDelta >= 0 ? 'text-green-600' : 'text-red-600'}>
                            {formatDelta(baselineDelta)}
                        </span>
                    </div>
                )}
            </div>

            {/* Normal Range Indicator */}
            <div className="flex items-center gap-2 text-xs text-gray-500">
                {isNormal ? (
                    <CheckCircle className="h-3.5 w-3.5 text-green-500" />
                ) : (
                    <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
                )}
                <span>
                    Normal: {config.formatValue(biomarker.normalRange[0])} - {config.formatValue(biomarker.normalRange[1])}
                </span>
            </div>

            {/* Description */}
            <p className="mt-2 text-xs text-gray-500">{config.description}</p>
        </motion.div>
    );
}


/**
 * Overall Score Summary Component
 */
interface ScoreSummaryProps {
    confidence: number;
    riskScore: number;
    qualityScore: number;
}

function ScoreSummary({ confidence, riskScore, qualityScore }: ScoreSummaryProps): React.ReactNode {
    return (
        <div className="grid grid-cols-3 gap-4 mb-6">
            {/* Confidence Score */}
            <div className="text-center p-4 rounded-lg bg-blue-50 border border-blue-100">
                <div className="text-3xl font-bold text-blue-600">
                    {Math.round(confidence * 100)}%
                </div>
                <div className="text-sm text-blue-700 font-medium">Confidence</div>
                <div className="text-xs text-blue-500 mt-1">Analysis certainty</div>
            </div>

            {/* Risk Score */}
            <div className="text-center p-4 rounded-lg bg-purple-50 border border-purple-100">
                <div className="text-3xl font-bold text-purple-600">
                    {Math.round(riskScore * 100)}
                </div>
                <div className="text-sm text-purple-700 font-medium">Risk Score</div>
                <div className="text-xs text-purple-500 mt-1">Neurological risk index</div>
            </div>

            {/* Quality Score */}
            <div className="text-center p-4 rounded-lg bg-teal-50 border border-teal-100">
                <div className="text-3xl font-bold text-teal-600">
                    {Math.round(qualityScore * 100)}%
                </div>
                <div className="text-sm text-teal-700 font-medium">Quality</div>
                <div className="text-xs text-teal-500 mt-1">Audio quality score</div>
            </div>
        </div>
    );
}

/**
 * Baseline Comparison Summary Component
 */
interface BaselineComparisonProps {
    baselineComparison: Record<string, number>;
}

function BaselineComparisonSummary({ baselineComparison }: BaselineComparisonProps): React.ReactNode {
    const improvements = Object.entries(baselineComparison).filter(([, delta]) => delta > 0.05);
    const declines = Object.entries(baselineComparison).filter(([, delta]) => delta < -0.05);
    const stable = Object.entries(baselineComparison).filter(
        ([, delta]) => delta >= -0.05 && delta <= 0.05
    );

    return (
        <div className="mb-6 p-4 rounded-lg bg-gray-50 border border-gray-200">
            <div className="flex items-center gap-2 mb-3">
                <Info className="h-5 w-5 text-gray-600" />
                <h4 className="font-semibold text-gray-900">Baseline Comparison</h4>
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-green-600" />
                    <span className="text-gray-700">
                        <span className="font-medium text-green-600">{improvements.length}</span> improved
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <Minus className="h-4 w-4 text-gray-400" />
                    <span className="text-gray-700">
                        <span className="font-medium text-gray-600">{stable.length}</span> stable
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <TrendingDown className="h-4 w-4 text-red-600" />
                    <span className="text-gray-700">
                        <span className="font-medium text-red-600">{declines.length}</span> declined
                    </span>
                </div>
            </div>
        </div>
    );
}

/**
 * Estimated Values Warning Component
 */
interface EstimatedWarningProps {
    estimatedCount: number;
}

function EstimatedWarning({ estimatedCount }: EstimatedWarningProps): React.ReactNode {
    if (estimatedCount === 0) return null;

    return (
        <div className="mb-4 p-3 rounded-lg bg-amber-50 border border-amber-200 flex items-start gap-2">
            <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
                <p className="text-sm text-amber-800 font-medium">
                    {estimatedCount} biomarker{estimatedCount > 1 ? 's' : ''} estimated
                </p>
                <p className="text-xs text-amber-700 mt-1">
                    Some values were estimated using clinically-validated defaults due to
                    insufficient data in the audio sample. These are marked with a help icon.
                </p>
            </div>
        </div>
    );
}


/**
 * Main Enhanced Speech Results Component
 * 
 * Displays all 9 biomarkers with values, confidence indicators,
 * estimated value markers, and baseline comparison.
 * 
 * **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
 */
export function EnhancedSpeechResults({
    result,
    className = '',
}: EnhancedSpeechResultsProps): React.ReactNode {
    // Count estimated biomarkers
    const estimatedCount = Object.values(result.biomarkers).filter(
        (b) => b.isEstimated
    ).length;

    return (
        <div className={`rounded-xl bg-white shadow-lg ${className}`}>
            {/* Header */}
            <div className="border-b border-gray-200 p-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold text-gray-900">
                            Speech Analysis Results
                        </h2>
                        <p className="mt-1 text-sm text-gray-600">
                            Session: {result.sessionId}
                        </p>
                        <p className="text-xs text-gray-500">
                            Completed: {new Date(result.timestamp).toLocaleString()}
                            {' • '}
                            Processing time: {(result.processingTime / 1000).toFixed(2)}s
                        </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <Activity className="h-6 w-6 text-blue-600" />
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="p-6">
                {/* Score Summary */}
                <ScoreSummary
                    confidence={result.confidence}
                    riskScore={result.riskScore}
                    qualityScore={result.qualityScore}
                />

                {/* Baseline Comparison Summary */}
                {result.baselineComparison && Object.keys(result.baselineComparison).length > 0 && (
                    <BaselineComparisonSummary baselineComparison={result.baselineComparison} />
                )}

                {/* Estimated Values Warning */}
                <EstimatedWarning estimatedCount={estimatedCount} />

                {/* Biomarkers Section Header */}
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">
                        Voice Biomarkers
                    </h3>
                    <span className="text-sm text-gray-500">
                        {BIOMARKER_CONFIGS.length} metrics analyzed
                    </span>
                </div>

                {/* Biomarker Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {BIOMARKER_CONFIGS.map((config) => {
                        const biomarker = result.biomarkers[config.key];
                        const baselineDelta = result.baselineComparison?.[config.key];

                        return (
                            <BiomarkerCard
                                key={config.key}
                                config={config}
                                biomarker={biomarker}
                                baselineDelta={baselineDelta}
                            />
                        );
                    })}
                </div>

                {/* Recommendations */}
                {result.recommendations && result.recommendations.length > 0 && (
                    <div className="mt-6 p-4 rounded-lg bg-gray-50 border border-gray-200">
                        <h4 className="font-semibold text-gray-900 mb-3">
                            Recommendations
                        </h4>
                        <ul className="space-y-2">
                            {result.recommendations.map((recommendation, index) => (
                                <li key={index} className="flex items-start gap-2">
                                    <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                                    <span className="text-sm text-gray-700">{recommendation}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}

/**
 * Utility function to check if all required biomarkers are present
 * Used for property testing
 */
export function hasAllBiomarkers(result: EnhancedSpeechAnalysisResult): boolean {
    const requiredKeys: (keyof EnhancedSpeechAnalysisResult['biomarkers'])[] = [
        'jitter',
        'shimmer',
        'hnr',
        'speechRate',
        'pauseRatio',
        'fluencyScore',
        'voiceTremor',
        'articulationClarity',
        'prosodyVariation',
    ];

    return requiredKeys.every((key) => {
        const biomarker = result.biomarkers[key];
        return (
            biomarker !== undefined &&
            typeof biomarker.value === 'number' &&
            typeof biomarker.unit === 'string' &&
            Array.isArray(biomarker.normalRange) &&
            biomarker.normalRange.length === 2 &&
            typeof biomarker.isEstimated === 'boolean'
        );
    });
}

/**
 * Utility function to check if confidence indicators are present
 * Used for property testing
 */
export function hasConfidenceIndicators(result: EnhancedSpeechAnalysisResult): boolean {
    return (
        typeof result.confidence === 'number' &&
        result.confidence >= 0 &&
        result.confidence <= 1
    );
}

/**
 * Utility function to check if estimated biomarkers are properly marked
 * Used for property testing
 */
export function hasEstimatedMarkers(result: EnhancedSpeechAnalysisResult): boolean {
    return Object.values(result.biomarkers).every(
        (biomarker) => typeof biomarker.isEstimated === 'boolean'
    );
}

/**
 * Get the count of biomarkers displayed
 * Used for property testing
 */
export function getBiomarkerCount(): number {
    return BIOMARKER_CONFIGS.length;
}

export default EnhancedSpeechResults;
