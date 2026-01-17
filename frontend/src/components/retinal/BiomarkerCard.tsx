/**
 * BiomarkerCard Component
 * 
 * Display card for individual biomarker values with reference ranges.
 * 
 * Features:
 * - Value display with units
 * - Reference range visualization
 * - Status indicator (normal/abnormal)
 * - Confidence score
 * - Trend indicator
 * 
 * @module components/retinal/BiomarkerCard
 */

'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus,
  CheckCircle2,
  AlertCircle,
  Info
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

interface BiomarkerCardProps {
  /** Biomarker display name */
  name: string;
  /** Current value */
  value: number;
  /** Unit of measurement */
  unit: string;
  /** Normal range [min, max] */
  normalRange?: [number, number];
  /** Confidence score (0-1) */
  confidence?: number;
  /** Icon to display */
  icon?: React.ReactNode;
  /** Previous value for trend */
  previousValue?: number;
  /** Description text */
  description?: string;
  /** Compact layout */
  compact?: boolean;
  /** Custom status override */
  status?: 'normal' | 'low' | 'high' | 'critical';
}

// ============================================================================
// Helpers
// ============================================================================

function getStatus(
  value: number, 
  normalRange?: [number, number]
): 'normal' | 'low' | 'high' {
  if (!normalRange) return 'normal';
  if (value < normalRange[0]) return 'low';
  if (value > normalRange[1]) return 'high';
  return 'normal';
}

function getStatusColor(status: string): { bg: string; text: string; border: string } {
  switch (status) {
    case 'low':
      return { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' };
    case 'high':
      return { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' };
    case 'critical':
      return { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' };
    default:
      return { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200' };
  }
}

function getTrend(value: number, previousValue?: number): 'up' | 'down' | 'stable' | null {
  if (previousValue === undefined) return null;
  const diff = value - previousValue;
  const threshold = Math.abs(previousValue) * 0.05; // 5% threshold
  if (diff > threshold) return 'up';
  if (diff < -threshold) return 'down';
  return 'stable';
}

// ============================================================================
// Component
// ============================================================================

export function BiomarkerCard({
  name,
  value,
  unit,
  normalRange,
  confidence,
  icon,
  previousValue,
  description,
  compact = false,
  status: statusOverride,
}: BiomarkerCardProps) {
  const status = statusOverride || getStatus(value, normalRange);
  const statusColors = getStatusColor(status);
  const trend = getTrend(value, previousValue);

  // Progress bar position within range
  const rangeProgress = useMemo(() => {
    if (!normalRange) return 50;
    const [min, max] = normalRange;
    const range = max - min;
    const padding = range * 0.5; // Show some padding outside range
    const fullMin = min - padding;
    const fullMax = max + padding;
    const fullRange = fullMax - fullMin;
    return Math.min(100, Math.max(0, ((value - fullMin) / fullRange) * 100));
  }, [value, normalRange]);

  if (compact) {
    return (
      <div className={`rounded-lg border ${statusColors.border} ${statusColors.bg} p-3`}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-zinc-700">{name}</span>
          <div className="flex items-center gap-1.5">
            <span className={`text-lg font-bold ${statusColors.text}`}>
              {value.toFixed(2)}
            </span>
            <span className="text-xs text-zinc-500">{unit}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          {icon && (
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-100">
              {icon}
            </div>
          )}
          <div>
            <h3 className="text-sm font-semibold text-zinc-900">{name}</h3>
            {description && (
              <p className="text-xs text-zinc-500">{description}</p>
            )}
          </div>
        </div>
        
        {/* Status Badge */}
        <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full ${statusColors.bg}`}>
          {status === 'normal' ? (
            <CheckCircle2 className={`h-3 w-3 ${statusColors.text}`} />
          ) : (
            <AlertCircle className={`h-3 w-3 ${statusColors.text}`} />
          )}
          <span className={`text-xs font-medium capitalize ${statusColors.text}`}>
            {status}
          </span>
        </div>
      </div>

      {/* Value Display */}
      <div className="flex items-baseline gap-2 mb-3">
        <span className="text-3xl font-bold text-zinc-900">
          {value.toFixed(2)}
        </span>
        <span className="text-sm text-zinc-500">{unit}</span>
        
        {/* Trend Indicator */}
        {trend && (
          <div className={`flex items-center gap-1 ml-2 ${
            trend === 'up' ? 'text-amber-600' : 
            trend === 'down' ? 'text-blue-600' : 
            'text-zinc-400'
          }`}>
            {trend === 'up' && <TrendingUp className="h-4 w-4" />}
            {trend === 'down' && <TrendingDown className="h-4 w-4" />}
            {trend === 'stable' && <Minus className="h-4 w-4" />}
          </div>
        )}
      </div>

      {/* Reference Range Bar */}
      {normalRange && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-zinc-500">
            <span>Reference Range</span>
            <span>{normalRange[0]} - {normalRange[1]} {unit}</span>
          </div>
          <div className="relative h-2 rounded-full bg-zinc-100 overflow-hidden">
            {/* Normal range indicator */}
            <div 
              className="absolute h-full bg-green-100"
              style={{ 
                left: '25%', 
                right: '25%',
              }}
            />
            
            {/* Value indicator */}
            <motion.div
              className={`absolute top-0 h-full w-1 rounded-full ${
                status === 'normal' ? 'bg-green-500' : 
                status === 'low' ? 'bg-blue-500' : 
                'bg-amber-500'
              }`}
              initial={{ left: 0 }}
              animate={{ left: `${rangeProgress}%` }}
              transition={{ duration: 0.5, delay: 0.2 }}
              style={{ transform: 'translateX(-50%)' }}
            />
          </div>
        </div>
      )}

      {/* Confidence Score */}
      {confidence !== undefined && (
        <div className="flex items-center gap-2 mt-3 pt-3 border-t border-zinc-100">
          <Info className="h-3.5 w-3.5 text-zinc-400" />
          <span className="text-xs text-zinc-500">
            Model Confidence: {(confidence * 100).toFixed(0)}%
          </span>
          <div className="flex-1 h-1 rounded-full bg-zinc-100 overflow-hidden">
            <motion.div
              className="h-full bg-cyan-500"
              initial={{ width: 0 }}
              animate={{ width: `${confidence * 100}%` }}
              transition={{ duration: 0.5, delay: 0.3 }}
            />
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default BiomarkerCard;
