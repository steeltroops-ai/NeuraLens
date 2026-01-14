'use client';

import React from 'react';
import { cn } from '@/utils/cn';

/**
 * NRI Risk Level Types
 * Based on MediLens Design System NRI Risk Gradient
 */
export type RiskLevel = 'minimal' | 'low' | 'moderate' | 'elevated' | 'high' | 'critical';

/**
 * Risk level score ranges
 */
export const RISK_LEVEL_RANGES = {
    minimal: { min: 0, max: 25 },
    low: { min: 26, max: 40 },
    moderate: { min: 41, max: 55 },
    elevated: { min: 56, max: 70 },
    high: { min: 71, max: 85 },
    critical: { min: 86, max: 100 },
} as const;

/**
 * NRI Risk Gradient Colors from MediLens Design System
 */
export const NRI_COLORS = {
    minimal: '#34C759',   // 0-25: Minimal risk
    low: '#30D158',       // 26-40: Low risk
    moderate: '#FFD60A',  // 41-55: Moderate risk
    elevated: '#FF9F0A',  // 56-70: Elevated risk
    high: '#FF6B6B',      // 71-85: High risk
    critical: '#FF3B30',  // 86-100: Critical risk
} as const;

/**
 * Risk level labels
 */
export const RISK_LABELS = {
    minimal: 'Minimal Risk',
    low: 'Low Risk',
    moderate: 'Moderate Risk',
    elevated: 'Elevated Risk',
    high: 'High Risk',
    critical: 'Critical Risk',
} as const;

/**
 * Get risk level from NRI score
 */
export function getRiskLevelFromScore(score: number): RiskLevel {
    if (score <= 25) return 'minimal';
    if (score <= 40) return 'low';
    if (score <= 55) return 'moderate';
    if (score <= 70) return 'elevated';
    if (score <= 85) return 'high';
    return 'critical';
}

/**
 * Get risk color from score
 */
export function getRiskColorFromScore(score: number): string {
    return NRI_COLORS[getRiskLevelFromScore(score)];
}

interface RiskBadgeProps {
    /** Risk level to display */
    level: RiskLevel;
    /** Optional custom label (defaults to standard risk label) */
    label?: string;
    /** Size variant */
    size?: 'sm' | 'md' | 'lg';
    /** Additional CSS classes */
    className?: string;
    /** Test ID for testing */
    testId?: string;
}

/**
 * MediLens Design System RiskBadge Component
 * 
 * Displays risk level indicators using the NRI gradient color palette.
 * Follows MediLens Design System specifications:
 * - rounded-full, px-3 py-1 styling
 * - NRI gradient colors (minimal through critical)
 */
export const RiskBadge: React.FC<RiskBadgeProps> = ({
    level,
    label,
    size = 'md',
    className,
    testId,
}) => {
    // MediLens NRI gradient color classes
    const colorClasses: Record<RiskLevel, string> = {
        minimal: 'bg-nri-minimal/10 text-nri-minimal',
        low: 'bg-nri-low/10 text-nri-low',
        moderate: 'bg-nri-moderate/10 text-nri-moderate',
        elevated: 'bg-nri-elevated/10 text-nri-elevated',
        high: 'bg-nri-high/10 text-nri-high',
        critical: 'bg-nri-critical/10 text-nri-critical',
    };

    // Size classes
    const sizeClasses = {
        sm: 'px-2 py-0.5 text-xs',
        md: 'px-3 py-1 text-sm',
        lg: 'px-4 py-1.5 text-base',
    };

    const displayLabel = label || RISK_LABELS[level];

    return (
        <span
            className={cn(
                'inline-flex items-center justify-center',
                'rounded-full',
                'font-medium',
                'transition-colors duration-200',
                colorClasses[level],
                sizeClasses[size],
                className,
            )}
            data-testid={testId}
            role="status"
            aria-label={`Risk level: ${displayLabel}`}
        >
            {displayLabel}
        </span>
    );
};

/**
 * RiskBadge with score - automatically determines risk level from score
 */
interface RiskBadgeWithScoreProps extends Omit<RiskBadgeProps, 'level'> {
    /** NRI score (0-100) */
    score: number;
    /** Show score in badge */
    showScore?: boolean;
}

export const RiskBadgeWithScore: React.FC<RiskBadgeWithScoreProps> = ({
    score,
    showScore = false,
    label,
    ...props
}) => {
    const level = getRiskLevelFromScore(score);
    const displayLabel = showScore
        ? `${score} - ${label || RISK_LABELS[level]}`
        : label || RISK_LABELS[level];

    return <RiskBadge level={level} label={displayLabel} {...props} />;
};

/**
 * Risk indicator dot - compact version for inline use
 */
interface RiskDotProps {
    level: RiskLevel;
    className?: string;
    testId?: string;
}

export const RiskDot: React.FC<RiskDotProps> = ({ level, className, testId }) => {
    const colorClasses: Record<RiskLevel, string> = {
        minimal: 'bg-nri-minimal',
        low: 'bg-nri-low',
        moderate: 'bg-nri-moderate',
        elevated: 'bg-nri-elevated',
        high: 'bg-nri-high',
        critical: 'bg-nri-critical',
    };

    return (
        <span
            className={cn(
                'inline-block w-2 h-2 rounded-full',
                colorClasses[level],
                className,
            )}
            data-testid={testId}
            role="presentation"
            aria-hidden="true"
        />
    );
};

export default RiskBadge;
