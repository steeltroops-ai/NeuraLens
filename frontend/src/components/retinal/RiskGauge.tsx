/**
 * RiskGauge Component
 * 
 * Visual gauge displaying risk score with color-coded segments.
 * 
 * Features:
 * - Animated gauge needle
 * - Color-coded risk segments
 * - Confidence interval display
 * - Category label
 * 
 * @module components/retinal/RiskGauge
 */

'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { RiskCategory, RISK_CATEGORY_COLORS } from '@/types/retinal-analysis';

// ============================================================================
// Types
// ============================================================================

interface RiskGaugeProps {
  /** Risk score (0-100) */
  score: number;
  /** Risk category */
  category: RiskCategory;
  /** Confidence interval [lower, upper] */
  confidenceInterval?: [number, number];
  /** Size of gauge */
  size?: 'sm' | 'md' | 'lg';
  /** Show category label */
  showLabel?: boolean;
  /** Show score value */
  showScore?: boolean;
}

// ============================================================================
// Size Configurations
// ============================================================================

const SIZE_CONFIG = {
  sm: {
    width: 160,
    height: 100,
    strokeWidth: 12,
    labelSize: 'text-xs',
    scoreSize: 'text-lg',
  },
  md: {
    width: 240,
    height: 140,
    strokeWidth: 16,
    labelSize: 'text-sm',
    scoreSize: 'text-2xl',
  },
  lg: {
    width: 320,
    height: 180,
    strokeWidth: 20,
    labelSize: 'text-base',
    scoreSize: 'text-3xl',
  },
};

// ============================================================================
// Risk Segments for the gauge
// ============================================================================

const RISK_SEGMENTS = [
  { category: 'minimal' as const, min: 0, max: 25, color: '#22c55e' },
  { category: 'low' as const, min: 25, max: 40, color: '#84cc16' },
  { category: 'moderate' as const, min: 40, max: 55, color: '#eab308' },
  { category: 'elevated' as const, min: 55, max: 70, color: '#f97316' },
  { category: 'high' as const, min: 70, max: 85, color: '#ef4444' },
  { category: 'critical' as const, min: 85, max: 100, color: '#991b1b' },
];

// ============================================================================
// Component
// ============================================================================

export function RiskGauge({
  score,
  category,
  confidenceInterval,
  size = 'md',
  showLabel = true,
  showScore = true,
}: RiskGaugeProps) {
  const config = SIZE_CONFIG[size];
  
  // Calculate gauge geometry
  const geometry = useMemo(() => {
    const cx = config.width / 2;
    const cy = config.height - 10;
    const radius = Math.min(cx, cy) - config.strokeWidth / 2 - 10;
    
    // Gauge spans from -180 to 0 degrees (semicircle)
    const startAngle = -180;
    const endAngle = 0;
    const angleRange = endAngle - startAngle;
    
    // Score to angle conversion
    const scoreToAngle = (s: number) => startAngle + (s / 100) * angleRange;
    const needleAngle = scoreToAngle(Math.min(100, Math.max(0, score)));
    
    return { cx, cy, radius, startAngle, endAngle, angleRange, needleAngle };
  }, [config, score]);

  // Generate arc path for SVG
  const createArcPath = (startPercent: number, endPercent: number) => {
    const { cx, cy, radius, startAngle, angleRange } = geometry;
    
    const start = startAngle + (startPercent / 100) * angleRange;
    const end = startAngle + (endPercent / 100) * angleRange;
    
    const startRad = (start * Math.PI) / 180;
    const endRad = (end * Math.PI) / 180;
    
    const x1 = cx + radius * Math.cos(startRad);
    const y1 = cy + radius * Math.sin(startRad);
    const x2 = cx + radius * Math.cos(endRad);
    const y2 = cy + radius * Math.sin(endRad);
    
    const largeArc = endPercent - startPercent > 50 ? 1 : 0;
    
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
  };

  // Calculate needle position
  const needleEnd = useMemo(() => {
    const { cx, cy, radius, needleAngle } = geometry;
    const rad = (needleAngle * Math.PI) / 180;
    return {
      x: cx + (radius - 10) * Math.cos(rad),
      y: cy + (radius - 10) * Math.sin(rad),
    };
  }, [geometry]);

  const categoryColor = RISK_CATEGORY_COLORS[category] || '#6b7280';

  return (
    <div className="flex flex-col items-center">
      {/* SVG Gauge */}
      <svg 
        width={config.width} 
        height={config.height} 
        viewBox={`0 0 ${config.width} ${config.height}`}
      >
        {/* Background track */}
        <path
          d={createArcPath(0, 100)}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={config.strokeWidth}
          strokeLinecap="round"
        />

        {/* Colored segments */}
        {RISK_SEGMENTS.map((segment, index) => (
          <path
            key={segment.category}
            d={createArcPath(segment.min, segment.max)}
            fill="none"
            stroke={segment.color}
            strokeWidth={config.strokeWidth - 2}
            strokeLinecap="butt"
            opacity={0.85}
          />
        ))}

        {/* Confidence interval arc (if provided) */}
        {confidenceInterval && (
          <motion.path
            d={createArcPath(confidenceInterval[0], confidenceInterval[1])}
            fill="none"
            stroke={categoryColor}
            strokeWidth={4}
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            opacity={0.4}
          />
        )}

        {/* Needle */}
        <motion.line
          x1={geometry.cx}
          y1={geometry.cy}
          x2={needleEnd.x}
          y2={needleEnd.y}
          stroke="#1f2937"
          strokeWidth={3}
          strokeLinecap="round"
          initial={{ rotate: -90 }}
          animate={{ rotate: 0 }}
          transition={{ 
            type: 'spring', 
            stiffness: 60, 
            damping: 12,
            duration: 1 
          }}
          style={{ transformOrigin: `${geometry.cx}px ${geometry.cy}px` }}
        />

        {/* Needle center dot */}
        <circle
          cx={geometry.cx}
          cy={geometry.cy}
          r={8}
          fill="#1f2937"
        />
        <circle
          cx={geometry.cx}
          cy={geometry.cy}
          r={4}
          fill="white"
        />

        {/* Score display */}
        {showScore && (
          <text
            x={geometry.cx}
            y={geometry.cy - 25}
            textAnchor="middle"
            className={`${config.scoreSize} font-bold fill-zinc-900`}
          >
            {score.toFixed(0)}
          </text>
        )}
      </svg>

      {/* Category Label */}
      {showLabel && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex flex-col items-center mt-2"
        >
          <span
            className={`px-3 py-1 rounded-full ${config.labelSize} font-semibold capitalize`}
            style={{ 
              backgroundColor: `${categoryColor}20`,
              color: categoryColor 
            }}
          >
            {category} Risk
          </span>

          {/* Confidence Interval */}
          {confidenceInterval && (
            <span className="mt-1 text-xs text-zinc-500">
              95% CI: {confidenceInterval[0].toFixed(0)} - {confidenceInterval[1].toFixed(0)}
            </span>
          )}
        </motion.div>
      )}
    </div>
  );
}

export default RiskGauge;
