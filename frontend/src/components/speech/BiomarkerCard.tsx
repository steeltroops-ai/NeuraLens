'use client';

import React from 'react';
import {
    Activity,
    Waves,
    MessageCircle,
    Gauge,
    Zap,
    BarChart2,
    Volume2,
    Pause,
    Music,
    AlertCircle,
    CheckCircle,
    TrendingUp,
    TrendingDown,
    Minus,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import type { BiomarkerResult, BaselineComparison } from '@/types/speech-enhanced';

interface BiomarkerCardProps {
    name: string;
    label: string;
    description: string;
    biomarker: BiomarkerResult;
    iconKey: string;
    higherIsBetter: boolean;
    formatValue: (value: number, unit: string) => string;
    baseline?: BaselineComparison;
    compact?: boolean;
}

const iconMap: Record<string, React.ElementType> = {
    activity: Activity,
    waves: Waves,
    'message-circle': MessageCircle,
    gauge: Gauge,
    zap: Zap,
    'bar-chart-2': BarChart2,
    'volume-2': Volume2,
    pause: Pause,
    music: Music,
};

export const BiomarkerCard: React.FC<BiomarkerCardProps> = ({
    label,
    description,
    biomarker,
    iconKey,
    higherIsBetter,
    formatValue,
    baseline,
    compact = false,
}) => {
    const Icon = iconMap[iconKey] || Activity;
    const [min, max] = biomarker.normal_range;
    const isNormal = biomarker.value >= min && biomarker.value <= max;

    // Calculate position on the range bar (0-100%)
    const rangeSpan = max - min;
    const extendedMin = min - rangeSpan * 0.5;
    const extendedMax = max + rangeSpan * 0.5;
    const totalSpan = extendedMax - extendedMin;
    const position = Math.max(0, Math.min(100, ((biomarker.value - extendedMin) / totalSpan) * 100));
    const normalStart = ((min - extendedMin) / totalSpan) * 100;
    const normalEnd = ((max - extendedMin) / totalSpan) * 100;

    // Status styling
    const getStatusColor = () => {
        if (isNormal) return 'text-[#22c55e]';
        const deviation = biomarker.value < min
            ? (min - biomarker.value) / min
            : (biomarker.value - max) / max;
        if (deviation > 0.3) return 'text-[#ef4444]';
        return 'text-[#f59e0b]';
    };

    const getStatusBg = () => {
        if (isNormal) return 'bg-[#dcfce7]';
        const deviation = biomarker.value < min
            ? (min - biomarker.value) / min
            : (biomarker.value - max) / max;
        if (deviation > 0.3) return 'bg-[#fee2e2]';
        return 'bg-[#fef3c7]';
    };

    if (compact) {
        return (
            <div className="flex items-center justify-between py-2 border-b border-[#f0f0f0] last:border-0">
                <div className="flex items-center gap-2">
                    <Icon size={14} className="text-[#64748b]" strokeWidth={1.5} />
                    <span className="text-[13px] text-[#334155]">{label}</span>
                    {biomarker.is_estimated && (
                        <span className="text-[10px] text-[#94a3b8] bg-[#f1f5f9] px-1.5 py-0.5 rounded">Est.</span>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    <span className={cn("text-[13px] font-medium", getStatusColor())}>
                        {formatValue(biomarker.value, biomarker.unit)}
                    </span>
                    {isNormal ? (
                        <CheckCircle size={14} className="text-[#22c55e]" />
                    ) : (
                        <AlertCircle size={14} className={getStatusColor()} />
                    )}
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-lg border border-[#e2e8f0] p-4">
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                    <div className={cn("p-1.5 rounded-md", getStatusBg())}>
                        <Icon size={16} className={getStatusColor()} strokeWidth={1.5} />
                    </div>
                    <div>
                        <h4 className="text-[13px] font-medium text-[#0f172a]">{label}</h4>
                        <p className="text-[11px] text-[#64748b]">{description}</p>
                    </div>
                </div>
                {biomarker.is_estimated && (
                    <span className="text-[10px] text-[#94a3b8] bg-[#f1f5f9] px-1.5 py-0.5 rounded">
                        Estimated
                    </span>
                )}
            </div>

            {/* Value */}
            <div className="mb-3">
                <span className={cn("text-xl font-semibold", getStatusColor())}>
                    {formatValue(biomarker.value, biomarker.unit)}
                </span>
                {biomarker.confidence !== null && (
                    <span className="text-[11px] text-[#94a3b8] ml-2">
                        {(biomarker.confidence * 100).toFixed(0)}% conf.
                    </span>
                )}
            </div>

            {/* Range visualization */}
            <div className="relative h-2 bg-[#f1f5f9] rounded-full mb-2">
                {/* Normal range indicator */}
                <div
                    className="absolute h-full bg-[#dcfce7] rounded-full"
                    style={{ left: `${normalStart}%`, width: `${normalEnd - normalStart}%` }}
                />
                {/* Current value marker */}
                <div
                    className={cn(
                        "absolute w-3 h-3 rounded-full -top-0.5 transform -translate-x-1/2 border-2 border-white shadow-sm",
                        isNormal ? "bg-[#22c55e]" : biomarker.value < min ? "bg-[#f59e0b]" : "bg-[#ef4444]"
                    )}
                    style={{ left: `${position}%` }}
                />
            </div>

            {/* Range labels */}
            <div className="flex justify-between text-[10px] text-[#94a3b8]">
                <span>{min.toFixed(2)}</span>
                <span className="text-[#64748b]">Normal Range</span>
                <span>{max.toFixed(2)}</span>
            </div>

            {/* Baseline comparison */}
            {baseline && (
                <div className="mt-3 pt-3 border-t border-[#f0f0f0]">
                    <div className="flex items-center justify-between text-[11px]">
                        <span className="text-[#64748b]">vs Baseline</span>
                        <div className="flex items-center gap-1">
                            {baseline.direction === 'improved' && (
                                <>
                                    <TrendingUp size={12} className="text-[#22c55e]" />
                                    <span className="text-[#22c55e]">+{baseline.delta_percent.toFixed(1)}%</span>
                                </>
                            )}
                            {baseline.direction === 'worsened' && (
                                <>
                                    <TrendingDown size={12} className="text-[#ef4444]" />
                                    <span className="text-[#ef4444]">{baseline.delta_percent.toFixed(1)}%</span>
                                </>
                            )}
                            {baseline.direction === 'stable' && (
                                <>
                                    <Minus size={12} className="text-[#64748b]" />
                                    <span className="text-[#64748b]">Stable</span>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default BiomarkerCard;
