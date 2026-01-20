"use client";

import React from "react";
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
  Brain,
  GitBranch,
  Wind,
  Music2,
  Maximize2,
  AudioWaveform,
  Sparkles,
} from "lucide-react";
import { cn } from "@/utils/cn";
import type {
  BiomarkerResult,
  BaselineComparison,
} from "@/types/speech-enhanced";

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
  isResearch?: boolean;
}

const iconMap: Record<string, React.ElementType> = {
  activity: Activity,
  waves: Waves,
  "message-circle": MessageCircle,
  gauge: Gauge,
  zap: Zap,
  "bar-chart-2": BarChart2,
  "volume-2": Volume2,
  pause: Pause,
  music: Music,
  brain: Brain,
  "git-branch": GitBranch,
  wind: Wind,
  "music-2": Music2,
  "maximize-2": Maximize2,
  "audio-waveform": AudioWaveform,
  sparkles: Sparkles,
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
  isResearch = false,
}) => {
  const Icon = iconMap[iconKey] || Activity;
  const [min, max] = biomarker.normal_range;
  const isNormal = biomarker.value >= min && biomarker.value <= max;

  // Calculate position on the range bar (0-100%)
  const rangeSpan = max - min;
  const extendedMin = min - rangeSpan * 0.5;
  const extendedMax = max + rangeSpan * 0.5;
  const totalSpan = extendedMax - extendedMin;
  const position = Math.max(
    0,
    Math.min(100, ((biomarker.value - extendedMin) / totalSpan) * 100),
  );
  const normalStart = ((min - extendedMin) / totalSpan) * 100;
  const normalEnd = ((max - extendedMin) / totalSpan) * 100;

  // Status styling - Dark theme colors
  const getStatusColor = () => {
    if (isNormal) return "text-emerald-400";
    const deviation =
      biomarker.value < min
        ? (min - biomarker.value) / min
        : (biomarker.value - max) / max;
    if (deviation > 0.3) return "text-red-400";
    return "text-amber-400";
  };

  const getStatusBg = () => {
    if (isNormal) return "bg-emerald-500/15";
    const deviation =
      biomarker.value < min
        ? (min - biomarker.value) / min
        : (biomarker.value - max) / max;
    if (deviation > 0.3) return "bg-red-500/15";
    return "bg-amber-500/15";
  };

  if (compact) {
    return (
      <div className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
        <div className="flex items-center gap-2">
          <Icon size={14} className="text-zinc-500" strokeWidth={1.5} />
          <span className="text-[13px] text-zinc-300">{label}</span>
          {biomarker.is_estimated && (
            <span className="text-[10px] text-zinc-500 bg-zinc-800 px-1.5 py-0.5 rounded">
              Est.
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={cn("text-[13px] font-medium", getStatusColor())}>
            {formatValue(biomarker.value, biomarker.unit)}
          </span>
          {isNormal ? (
            <CheckCircle size={14} className="text-emerald-400" />
          ) : (
            <AlertCircle size={14} className={getStatusColor()} />
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "bg-zinc-800/50 rounded-lg border p-4",
        isResearch ? "border-violet-500/30" : "border-zinc-700",
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "p-1.5 rounded-md",
              isResearch ? "bg-violet-500/15" : getStatusBg(),
            )}
          >
            <Icon
              size={16}
              className={isResearch ? "text-violet-400" : getStatusColor()}
              strokeWidth={1.5}
            />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h4 className="text-[13px] font-medium text-zinc-200">{label}</h4>
              {isResearch && <Sparkles size={10} className="text-violet-400" />}
            </div>
            <p className="text-[11px] text-zinc-500">{description}</p>
          </div>
        </div>
        {biomarker.is_estimated && (
          <span className="text-[10px] text-zinc-500 bg-zinc-800 px-1.5 py-0.5 rounded">
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
          <span className="text-[11px] text-zinc-500 ml-2">
            {(biomarker.confidence * 100).toFixed(0)}% conf.
          </span>
        )}
      </div>

      {/* Range visualization */}
      <div className="relative h-2 bg-zinc-700 rounded-full mb-2">
        {/* Normal range indicator */}
        <div
          className="absolute h-full bg-emerald-500/30 rounded-full"
          style={{
            left: `${normalStart}%`,
            width: `${normalEnd - normalStart}%`,
          }}
        />
        {/* Current value marker */}
        <div
          className={cn(
            "absolute w-3 h-3 rounded-full -top-0.5 transform -translate-x-1/2 border-2 border-zinc-900 shadow-sm",
            isNormal
              ? "bg-emerald-400"
              : biomarker.value < min
                ? "bg-amber-400"
                : "bg-red-400",
          )}
          style={{ left: `${position}%` }}
        />
      </div>

      {/* Range labels */}
      <div className="flex justify-between text-[10px] text-zinc-500">
        <span>{min.toFixed(2)}</span>
        <span className="text-zinc-400">Normal Range</span>
        <span>{max.toFixed(2)}</span>
      </div>

      {/* Baseline comparison */}
      {baseline && (
        <div className="mt-3 pt-3 border-t border-zinc-700">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-zinc-500">vs Baseline</span>
            <div className="flex items-center gap-1">
              {baseline.direction === "improved" && (
                <>
                  <TrendingUp size={12} className="text-emerald-400" />
                  <span className="text-emerald-400">
                    +{baseline.delta_percent.toFixed(1)}%
                  </span>
                </>
              )}
              {baseline.direction === "worsened" && (
                <>
                  <TrendingDown size={12} className="text-red-400" />
                  <span className="text-red-400">
                    {baseline.delta_percent.toFixed(1)}%
                  </span>
                </>
              )}
              {baseline.direction === "stable" && (
                <>
                  <Minus size={12} className="text-zinc-500" />
                  <span className="text-zinc-500">Stable</span>
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
